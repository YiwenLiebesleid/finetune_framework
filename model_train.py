"""
    model training module
    get the processed DS as input and train the model
    save to desired path
"""
from frame_config import *
from debug_util import debug_prt
import os 
import sys
import torch
import numpy as np
import pandas as pd
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, EarlyStoppingCallback, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import load_dataset, DatasetDict , Dataset, Audio
from peft import prepare_model_for_int8_training
from tqdm import tqdm
import evaluate
from peft import LoraConfig, LoraConfig, get_peft_model
import datasets
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import jiwer
import gc
import transformers
import wandb
from wandb import AlertLevel

from isat.wer_utils import wer_from_counts


def model_training():

    problem_name = config_dic["problem_name"]

    # setup wandb
    wandb_project = config_dic["wandb_project"]
    wandb_name = config_dic["wandb_name"]
    wandb.init(project=wandb_project, name=wandb_name)


    # basic configuration
    init_from_hub_path = config_dic["init_from_hub_path"]
    task = config_dic["task"]
    lang = config_dic["lang"]
    OUT_DIR = OUT_DIR_PATH
    USE_OTHER_MODEL = model_config_dic["use_other_pretrained_model"]
    pretrained_model_path = model_config_dic["pretrained_model_path"]

    MERGE_LORA_MODEL = model_config_dic["merge_lora_model"]
    merge_lora_adapter_path = model_config_dic["merge_lora_adapter_path"]
    merged_lora_model_save_path = model_config_dic["merged_lora_model_save_path"]

    VALIDATE_ALL = model_config_dic["validate_all"]
    VALIDATE_RANGE = model_config_dic["validate_range"]
    VAL_DATASET_NAME = model_config_dic["val_dataset_name"]

    USE_INT8 = model_config_dic["use_int8"]
    USE_LORA = model_config_dic["use_lora"]
    LORA_TARGET = model_config_dic["lora_target"]

    START_FROM_CHECKPOINT = model_config_dic["start_from_checkpoint"]
    checkpoint_path = model_config_dic["checkpoint_path"]

    debug_prt(problem_name, "problem_name")
    debug_prt(init_from_hub_path, "init_from_hub_path")
    debug_prt(OUT_DIR, "OUT_DIR")
    debug_prt(MERGE_LORA_MODEL, "MERGE_LORA_MODEL")
    if MERGE_LORA_MODEL:
        debug_prt(merge_lora_adapter_path, "merge_lora_adapter_path")
        debug_prt(merged_lora_model_save_path, "merged_lora_model_save_path")
    debug_prt(USE_OTHER_MODEL, "USE_OTHER_MODEL")
    if USE_OTHER_MODEL:
        debug_prt(pretrained_model_path, "pretrained_model_path")
    debug_prt(VALIDATE_ALL, "VALIDATE_ALL")
    if not VALIDATE_ALL:
        debug_prt(VALIDATE_RANGE, "VALIDATE_RANGE")
    debug_prt(VAL_DATASET_NAME, "VAL_DATASET_NAME")
    debug_prt(START_FROM_CHECKPOINT, "START_FROM_CHECKPOINT")
    if START_FROM_CHECKPOINT:
        debug_prt(checkpoint_path, "checkpoint_path")

    data_root = config_dic["data_root"]
    wav_col_name = config_dic["wav_col"]
    trans_col_name = config_dic["transcript_col"]

    debug_prt(data_root, "data_root")
    debug_prt(wav_col_name, "wav_col_name")
    debug_prt(trans_col_name, "trans_col_name")

    # get training data and validation data
    if model_config_dic["use_exist_dataset"] is True:
        # direct load dataset from HF
        ds_name, ds_subset = model_config_dic["exist_dataset_path"]
        DS = load_dataset(ds_name, ds_subset)

        debug_prt("Use HF dataset", "[STATUS]")
        debug_prt(ds_name, "ds_name")
        debug_prt(ds_subset, "ds_subset")
    else:
        # use local dataset
        ds_name, ds_subset = model_config_dic["exist_dataset_path"]
        train_ds_path = model_config_dic["local_dataset_train_path"]
        val_ds_path = model_config_dic["local_dataset_val_path"]
        DS = DatasetDict({
            "train": 
                DataSet_from_manifest(train_ds_path, data_root, wav_col_name, trans_col_name, 'train') if train_ds_path != "default" 
                else load_dataset(ds_name, ds_subset)["train"],
            VAL_DATASET_NAME: 
                DataSet_from_manifest(val_ds_path, data_root, wav_col_name, trans_col_name, VAL_DATASET_NAME) if val_ds_path != "default"
                else load_dataset(ds_name, ds_subset)[VAL_DATASET_NAME],
            }
        )
        
        debug_prt("Use local dataset", "[STATUS]")
        debug_prt(data_root, "data_root")
        if train_ds_path == "default":
            debug_prt(f"(default) {ds_name}", "train_ds_path")
        else:
            debug_prt(train_ds_path, "train_ds_path")
        if val_ds_path == "default":
            debug_prt(f"(default) {ds_name}", "val_ds_path")
        else:
            debug_prt(val_ds_path, "val_ds_path")

    if not VALIDATE_ALL:
        inds = np.array(range(0, DS[VAL_DATASET_NAME].num_rows))
        np.random.shuffle(inds)
        inds = inds[0: VALIDATE_RANGE]
        DS[VAL_DATASET_NAME] = datasets.arrow_dataset.Dataset.from_dict(DS[VAL_DATASET_NAME][inds])

    sampling_rate = config_dic["sampling_rate"]
    DS = DS.cast_column("audio", Audio(sampling_rate=sampling_rate))



    # other modules and data collator
    from transformers import WhisperFeatureExtractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(init_from_hub_path)
    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained(init_from_hub_path, language=lang, task=task)
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(init_from_hub_path, language=lang, task=task)
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    def prepare_dataset(batch):
        audio = batch[wav_col_name]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch[trans_col_name]).input_ids
        return batch

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            features = [prepare_dataset(feature) for feature in features] 
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)



    # evaluation metrics
    metric = evaluate.load("wer")

    def compute_metrics(pred, do_normalize_eval=True):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]
            pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
            label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        wandb.log({'wer': wer})
        wandb.alert(
            title=f"WER update",
            text=f"new WER: {wer}",
            level=AlertLevel.INFO
            )
        return {"wer": wer}

    # manually set temperature to 0
    def compute_metrics_temperature_0(pred, do_normalize_eval=True):
        EVAL_BATCH_SIZE = 16
        eval_dataloader = DataLoader(DS["validation"], batch_size=EVAL_BATCH_SIZE, collate_fn=data_collator)
        model.eval()
        meas={k:0 for k in ('substitutions','deletions','insertions','word_counts')}
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    generated_tokens = (
                        model.generate(
                            input_features=batch["input_features"].to("cuda"),
                            max_new_tokens=112,
                            temperature = 0,
                        )
                        .cpu()
                        .numpy()
                    )
                    labels = batch["labels"].cpu().numpy()
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    
                    decoded_preds = [normalizer(pred) for pred in decoded_preds]
                    decoded_labels= [normalizer(label) for label in decoded_labels]
                    # filtering step to only evaluate the samples that correspond to non-zero references:
                    decoded_preds = [decoded_preds[i] for i in range(len(decoded_preds)) if len(decoded_labels[i]) > 0]
                    decoded_labels = [decoded_labels[i] for i in range(len(decoded_labels)) if len(decoded_labels[i]) > 0]
                    # get s, del, ins direct from JIWER
                    wer_meas = [jiwer.compute_measures(ref,hyp) for ref,hyp in zip(decoded_labels, decoded_preds)]
                    meas['substitutions'] += sum([res['substitutions'] for res in wer_meas])
                    meas['deletions'] += sum([res['deletions'] for res in wer_meas])
                    meas['insertions'] += sum([res['insertions'] for res in wer_meas])
                    meas['word_counts'] += sum([len(ref.split(' ')) for ref in decoded_labels])
            del generated_tokens, labels, batch, wer_meas
            gc.collect()
        wer_meas = wer_from_counts(meas['word_counts'],
        meas['substitutions'],
        meas['deletions'],
        meas['insertions'] )
        for key,val in wer_meas.items():
            print((f"{key} = {val:.4f}" ))
        wer = 100 * wer_meas["wer"]
        wandb.log({'wer': wer})
        wandb.alert(
            title=f"WER update",
            text=f"new WER: {wer}",
            level=AlertLevel.INFO
            )
        return {"wer": wer}


    # merge lora model and get back base model with lora weights adapted
    if MERGE_LORA_MODEL:
        from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
        from peft import prepare_model_for_int8_training
        peft_config = PeftConfig.from_pretrained(merge_lora_adapter_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            init_from_hub_path, 
            load_in_8bit=USE_INT8,
            device_map={'': 0}, 
            use_cache=False,
        )
        base_model.config.forced_decoder_ids = None
        base_model.config.suppress_tokens = []
        base_model.config.use_cache = False
        base_model.enable_input_require_grads()
        base_model = prepare_model_for_int8_training(base_model)
        model_to_merge = PeftModel.from_pretrained(
            WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path).to("cuda"),
        merge_lora_adapter_path)
        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained(merged_lora_model_save_path)
        del merged_model, base_model


    # model parameters
    if USE_OTHER_MODEL:
        load_model_path = pretrained_model_path
    elif MERGE_LORA_MODEL:
        load_model_path = merged_lora_model_save_path
    else:
        load_model_path = init_from_hub_path
    debug_prt(load_model_path, "load_model_path")

    if USE_LORA:
        model = WhisperForConditionalGeneration.from_pretrained(
            load_model_path, 
            load_in_8bit=USE_INT8,
            device_map={'': 0}, 
            use_cache=False,
        )

        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.config.use_cache = False
        model.enable_input_require_grads()
        
        model = prepare_model_for_int8_training(model)

        if LORA_TARGET == "all":
            import re
            pattern = r'\((\w+)\): Linear'
            linear_layers = re.findall(pattern, str(model.modules))
            target_modules = list(set(linear_layers))
        else:
            target_modules = ["q_proj", "v_proj"]

        config = LoraConfig(
            r = model_config_dic["lora_r"],
            lora_alpha = model_config_dic["lora_alpha"],
            target_modules = target_modules, 
            lora_dropout = model_config_dic["lora_dropout"], 
            bias = "none")
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        debug_prt("use lora", "[STATUS]")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            load_model_path, 
        ).cuda()

        debug_prt("not use lora", "[STATUS]")


    # PEFT callback
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


    # setup trainer
    train_args = training_args
    early_stopping_patience = model_config_dic["early_stopping_patience"]
    set_temperature_0 = model_config_dic["set_temperature_0"]

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset = DS["train"],
        eval_dataset = DS[VAL_DATASET_NAME],
        compute_metrics=compute_metrics if not set_temperature_0 else compute_metrics_temperature_0,
        tokenizer=feature_extractor,
        callbacks=[SavePeftModelCallback, EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
                ] if USE_LORA else [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    processor.save_pretrained(train_args.output_dir)
    model.config.use_cache = False
    model.config.to_json_file(os.path.join(OUT_DIR, "config.json"))

    debug_prt(early_stopping_patience, "early_stopping_patience")
    debug_prt(set_temperature_0, "set_temperature_0")


    # model training
    debug_prt("start training", "[STATUS]")

    wandb.watch(model, log="all")
    wandb.alert(
        title="Model Training Start",
        text=f"{problem_name} Model Training start",
        level=AlertLevel.INFO
    )

    transformers.logging.set_verbosity_info()
    with torch.autocast("cuda"):
        if not START_FROM_CHECKPOINT:
            debug_prt("not from checkpoint", "[STATUS]")
            trainer.train()
        else:
            debug_prt("start from checkpoint", "[STATUS]")
            debug_prt(checkpoint_path, "checkpoint_path")
            trainer.train(resume_from_checkpoint = checkpoint_path)
        
        if USE_LORA:
            save_model_path = os.path.join(OUT_DIR, 'final/adapter_model')
        else:
            save_model_path = os.path.join(OUT_DIR, 'final')
    trainer.save_model(save_model_path)

    debug_prt("end training", "[STATUS]")
    debug_prt(save_model_path, "save_model_path")



if __name__ == "__main__":
    model_training()