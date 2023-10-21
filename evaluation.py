"""
    evaluation module
    get the processed DS as input and evaluate the model
    save the predicted ASR to desired path
"""
from frame_config import *
from debug_util import debug_prt
import os
import torch
import numpy as np
import pandas as pd
from transformers import WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict , Dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import jiwer
from isat.wer_utils import format_text_for_wer, wer_from_counts
from num2words import num2words
import re
from decimal import InvalidOperation
import string


def strip_punct(instr):
    newstr = ''
    for word in instr.split():
	    # delete punct
        word = word.strip(string.punctuation)

        # delete commas inside numbers
        m = re.match(r'(\d*),(\d)', word)
        if m != None:
            word = word.replace(',', '')

        # commas inside words become space
        word = re.sub(",", " ", word)

        # hyphens inside words become space
        word = re.sub("-", " ", word)
        word = word.strip()

        newstr += ' ' + word
    newstr = newstr.strip()
    return newstr

def caught_num2words(text):
    # first do currency replacements
    if '$' in text:
        text = re.sub('\$([0-9]+)', '\g<1> dollars', text)
    # strip punctuation 
    text=strip_punct(text)
    # catch strings that might be converted to infinity or NaN and return as is... 
    naughty_words = ['INF','Inf','inf','NAN','NaN', 'nan', 'NONE','None','none','Infinity','infinity']
    if text in naughty_words:
        return text
    try:
        if len(text.split()) > 1:
            return ' '.join([caught_num2words(word) for word in text.split()])
        else:
            return num2words(text)
    except (InvalidOperation, ValueError) as error:
        return text

def process_num2words(text):
    if isinstance(text,list):
        text = ' '.join(text)
    text = text.replace('\n',' ') # replace newline with space
    text = ' '.join([caught_num2words(str) for str in text.split(' ')]) # spell out numbers
    text = strip_punct(text)
    text = text.lower()
    text = re.sub('\s+',' ',text) # replace multiple space with single
    return text


def do_evaluation():

    # basic setups
    problem_name = config_dic["problem_name"]
    OUT_DIR = OUT_DIR_PATH

    EVAL_BATCH_SIZE = evaluate_config_dic["eval_batch_size"]
    model_path = os.path.join(evaluate_config_dic["model_dir"], evaluate_config_dic["model_base_name"])
    if model_path[-1] == "/":
        model_path = model_path[0:-1]
    init_from_hub_path = config_dic["init_from_hub_path"]
    task = config_dic["task"]
    lang = config_dic["lang"]

    debug_prt(problem_name, "problem_name")
    debug_prt(EVAL_BATCH_SIZE, "EVAL_BATCH_SIZE")
    debug_prt(model_path, "model_path")
    debug_prt(init_from_hub_path, "init_from_hub_path")
    debug_prt(OUT_DIR, "OUT_DIR")

    data_root = config_dic["data_root"]
    wav_col_name = config_dic["wav_col"]
    trans_col_name = config_dic["transcript_col"]

    debug_prt(data_root, "data_root")
    debug_prt(wav_col_name, "wav_col_name")
    debug_prt(trans_col_name, "trans_col_name")

    # test data setup
    if evaluate_config_dic["use_exist_dataset"] is True:
        # direct load dataset from HF
        ds_name, ds_subset, ds_split = evaluate_config_dic["exist_dataset_path"]
        DS_test = load_dataset(ds_name, ds_subset, split=ds_split)

        debug_prt("Use HF dataset", "[STATUS]")
        debug_prt(ds_name, "ds_name")
        debug_prt(ds_subset, "ds_subset")
        debug_prt(ds_split, "ds_split")
    else:
        # use local dataset
        local_data_path = evaluate_config_dic["local_dataset_path"]

        DS_test = DatasetDict({
            "test": DataSet_from_manifest(local_data_path, data_root, wav_col_name, trans_col_name, 'test'),
            }
        )["test"]

        debug_prt("Use local dataset", "[STATUS]")
        debug_prt(local_data_path, "local_data_path")

    sampling_rate = config_dic["sampling_rate"]
    DS_test = DS_test.cast_column("audio", Audio(sampling_rate=sampling_rate))


    # load model
    USE_INT8 = evaluate_config_dic["use_int8"]
    USE_LORA = evaluate_config_dic["use_lora"]
    lora_config_path = evaluate_config_dic["lora_config_path"]

    debug_prt(USE_INT8, "USE_INT8")
    debug_prt(USE_LORA, "USE_LORA")

    if USE_LORA:
        peft_config = PeftConfig.from_pretrained(lora_config_path)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, 
            load_in_8bit = USE_INT8,
            device_map = {'': 0}, 
            use_cache = False,
        )
        model = PeftModel.from_pretrained(model, lora_config_path)
        debug_prt(lora_config_path, "lora_config_path")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_path).cuda()



    # other modules and data collator
    from transformers import WhisperFeatureExtractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(init_from_hub_path)
    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained(init_from_hub_path, language=lang, task=task)
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(init_from_hub_path, language=lang, task=task)
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    # define data collator
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



    # do evaluation
    do_eval = evaluate_config_dic["do_eval"]
    eval_from_csv = evaluate_config_dic["eval_from_csv"]

    debug_prt(do_eval, "do_eval")
    debug_prt(eval_from_csv, "eval_from_csv")

    do_normalize_eval = evaluate_config_dic["do_normalize_eval"]
    do_isat_normalize = evaluate_config_dic["do_isat_normalize"]
    do_num2words = evaluate_config_dic["do_num2words"]
    max_new_tokens = evaluate_config_dic["max_new_tokens"]
    set_temperature_0 = evaluate_config_dic["set_temperature_0"]

    debug_prt(do_normalize_eval, "do_normalize_eval")
    debug_prt(do_isat_normalize, "do_isat_normalize")
    debug_prt(do_num2words, "do_num2words")
    debug_prt(max_new_tokens, "max_new_tokens")
    debug_prt(set_temperature_0, "set_temperature_0")

    if do_eval:
        debug_prt("eval from data", "[STATUS]")

        eval_dataloader = DataLoader(DS_test, batch_size=EVAL_BATCH_SIZE, collate_fn=data_collator)
        model.eval()
        meas={k:0 for k in ('substitutions','deletions','insertions','word_counts')}
        asr_results=[]
        bcount=0

        for step, batch in enumerate(tqdm(eval_dataloader)):
            bcount+=len(batch['input_features'])

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    if set_temperature_0 is True:
                        generated_tokens = (
                            model.generate(
                                input_features=batch["input_features"].to("cuda"),
                                max_new_tokens = max_new_tokens,
                                temperature = 0,
                            )
                            .cpu()
                            .numpy()
                        )
                    else:
                        generated_tokens = (
                            model.generate(
                                input_features=batch["input_features"].to("cuda"),
                                max_new_tokens = max_new_tokens,
                            )
                            .cpu()
                            .numpy()
                        )
                    labels = batch["labels"].cpu().numpy()
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    decoded_preds_pre_norm=decoded_preds
                    decoded_labels_pre_norm=decoded_labels

                    if do_isat_normalize:
                        decoded_preds = [format_text_for_wer(pred) for pred in decoded_preds]
                        decoded_labels= [format_text_for_wer(label) for label in decoded_labels]
                    if do_normalize_eval:
                        decoded_preds = [normalizer(pred) for pred in decoded_preds]
                        decoded_labels= [normalizer(label) for label in decoded_labels]
                    if do_num2words:
                        decoded_preds = [process_num2words(pred) for pred in decoded_preds]
                        decoded_labels= [process_num2words(label) for label in decoded_labels]

                    asr_results.extend( [
                        {
                        'ref':ref,
                        'hyp':hyp,
                        'ref_norm':ref_norm,
                        'hyp_norm':hyp_norm}
                        for ix, (ref,hyp,ref_norm,hyp_norm) in enumerate(zip(decoded_labels_pre_norm,decoded_preds_pre_norm,decoded_labels, decoded_preds))
                    ] )

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

        # save ASR results
        model_base_name = evaluate_config_dic["model_base_name"]
        ASR_dir = os.path.join(OUT_DIR, 'ASR_output')
        os.makedirs(ASR_dir, exist_ok=True)
        ASR_file = f"ASR_{problem_name}_{model_base_name}.csv"
        ASR_file_path = os.path.join(ASR_dir, f"{ASR_file}")
        pd.DataFrame.from_records(asr_results).to_csv(ASR_file_path)

        debug_prt(ASR_dir, "ASR_dir")
        debug_prt(ASR_file_path, "ASR_file_path")

        # compute WERs
        wer_meas = wer_from_counts(meas['word_counts'],
            meas['substitutions'],
            meas['deletions'],
            meas['insertions'] )
        for key,val in wer_meas.items():
            print((f"{key} = {val:.4f}" ))

    def wer_from_csv(csv_path, refcol='ref', hypcol='hyp', return_alignments=False):
        res = pd.read_csv(csv_path).astype(str)
        # refs = res[refcol].apply(format_text_for_wer)
        # hyps = res[hypcol].apply(format_text_for_wer)
        refs = res[refcol]
        hyps = res[hypcol]
        refs = [normalizer(label) for label in refs]
        hyps = [normalizer(pred) for pred in hyps]
        if do_num2words:
            hyps = [process_num2words(pred) for pred in hyps]
            refs = [process_num2words(label) for label in refs]
        wer_meas = jiwer.compute_measures(list(refs), list(hyps))
        if not return_alignments:
            # remove alignments 
            del wer_meas['ops']
            del wer_meas['truth']
            del wer_meas['hypothesis']
        wer_meas['word_counts'] = wer_meas['substitutions']+wer_meas['deletions']+wer_meas['hits']
        wer_meas['sub_rate'] = wer_meas['substitutions']/wer_meas['word_counts'] 
        wer_meas['del_rate'] = wer_meas['deletions']/wer_meas['word_counts'] 
        wer_meas['ins_rate'] = wer_meas['insertions']/wer_meas['word_counts'] 
        return wer_meas

    if eval_from_csv:
        debug_prt("eval from csv", "[STATUS]")

        wer_meas = wer_from_csv(ASR_file_path)
        for key in ['wer','sub_rate','del_rate','ins_rate']:
            print((f"{key} = {wer_meas[key]:.2f}" ))

        table_string = f"{wer_meas['wer']:.4f} ({wer_meas['sub_rate']:.4f}, {wer_meas['del_rate']:.4f}, {wer_meas['ins_rate']:.4f})"
        print(table_string)



if __name__ == "__main__":
    do_evaluation()