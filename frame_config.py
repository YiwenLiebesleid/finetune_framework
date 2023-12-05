"""
    framework for whisper tasks
    1. data preparation
    2. model parameters setup
    3. model training
    4. model evaluation
"""
from debug_util import debug_prt
from transformers import Seq2SeqTrainingArguments
import transformers
import pandas as pd
import os
import datasets
from datasets import load_dataset, DatasetDict , Dataset, Audio


# debug settings: get [DEBUG] information
debug_prt.debug = True

# stage setup: which stages to go through
pipeline_stage_setup = {
    "data_preparation" : False,
    "model_training" : False,
    "evaluation" : True
}

# basic settings for the pipeline
config_dic = {
    # problem names setup, display in result
    "problem_name" : "ft_ami_tiny_RS_10h_on_raw_file",
    # "problem_name" : "ft_cv_tiny_PL_RS_10h",

    # wandb setups, to which project and name it'll display
    "wandb_project" : "active_learning",
    "wandb_name" : "AMI tiny RS 10h on raw file",
    # "wandb_name" : "CV tiny PL RS 10h",

    # basic model initialise settings
    # usage: WhisperTokenizer.from_pretrained(init_from_hub_path, language=lang, task=task)
    # "init_from_hub_path" : "openai/whisper-large-v2",
    "init_from_hub_path" : "openai/whisper-tiny",
    "task" : "transcribe",
    "lang" : None,

    # paths
    # data_root is used for DataSet_from_manifest, to replace $DATAROOT inside csv
    # OUT_DIR_PATH = os.path.join(out_dir_root, problem_suffix)
    "data_root" : "/home/yguan2/.cache/huggingface/datasets/downloads/extracted/",
    "out_dir_root" : "/scratch/shared/whitehill/yguan2/model/",
    "problem_suffix" : "al_ft_ami_tiny/RS_10h_on_raw_file/",
    # "problem_suffix" : "al_ft_CV_tiny/PL_RS_10h/",
    
    # data info:
    # the desired sampling rate for model input
    # the name of transcript column
    "sampling_rate" : 16000,
    "wav_col" : "audio",
    "transcript_col" : "text",
}

OUT_DIR_PATH = os.path.join(config_dic["out_dir_root"], config_dic["problem_suffix"])

# model related settings
model_config_dic = {
    # pretrained model path
    # usage: WhisperForConditionalGeneration.from_pretrained(pretrained_model_path)
    # if not using other model, will directly use init_from_hub_path as model
    "use_other_pretrained_model" : False,
    "pretrained_model_path" : "/scratch/shared/whitehill/yguan2/model/ft_ISAT_add_padding_small/rosy_whisat/merged_model",
    
    # if use lora-tuned model to continue fine-tuning
    # this will make future load "merged_lora_model_save_path" as pretrained model
    "merge_lora_model" : False,
    "merge_lora_adapter_path" : "",
    "merged_lora_model_save_path" : "",

    # training dataset
    # set use_exist_dataset to True when using HF datasets (e.g. AMI, LibriSpeech)
    # use_exist_dataset: True/False
    # exist_dataset_path: [dataset_name, subset]
    # local_dataset_train/val_path: if set to "default", means to use "exist_dataset_path"
    # when use_exist_dataset is False, then will use local_dataset_path: a csv file
    "use_exist_dataset" : False,
    "exist_dataset_path" : ["edinburghcstr/ami", "ihm"],
    # "exist_dataset_path" : ["mozilla-foundation/common_voice_13_0", "en"],
    "local_dataset_train_path" : '/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/GT_based/AMI_RS_10h.csv',
    # "local_dataset_train_path" : '/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/PL_CommonVoice/CV_PL_RS_10hours.csv',
    "local_dataset_val_path" : 'default',
    "val_dataset_name" : "validation",

    # lora ralated training settings
    # lora_target default as ["q_proj", "v_proj"], set to "all" for all linear layers
    "use_int8" : False,
    "use_lora" : False,
    "lora_target" : "",
    "lora_r" : 8,
    "lora_alpha" : 8,
    "lora_dropout" :0.05,

    # checkpoint settings
    # usage: if START_FROM_CHECKPOINT: trainer.train(resume_from_checkpoint = checkpoint_path)
    # if directly from the latest checkpoint, set checkpoint_path to True
    "start_from_checkpoint" : False,
    "checkpoint_path" : False,

    # validation settings
    # usage: to just use a part of the validation set to validate
    # also set the validate name to your desired set
    "validate_all" : False,
    "validate_range" : 8000,

    # evaluation setting
    # usage: set temperature to 0, also automatically set predict_with_generate to False
    "set_temperature_0" : False,

    # earlystopping setting
    # usage: trainer add an earlystopping strategy to callbacks
    # set to 0 if don't want to use early stopping
    "early_stopping_patience" : 20,
}

# training arguments
# directly using Seq2SeqTrainingArguments here for better extendability
training_args = Seq2SeqTrainingArguments(
    # modify this part
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,  # increase by 2x for every 2x decrease in batch size
    per_device_eval_batch_size = 4,
    eval_accumulation_steps = 2,
    learning_rate = 5e-5,
    # lr_scheduler_type = "linear",       # polynomial ? linear ? cosine_with_restarts?
    lr_scheduler_type = "cosine_with_restarts",
    warmup_steps = 200,
    logging_steps = 25,
    # max_steps=1500,
    num_train_epochs = 10,              # total number of training epochs
    evaluation_strategy = "steps",
    gradient_checkpointing = True,
    fp16 = True,
    generation_max_length = 112,
    save_steps = 300,
    eval_steps = 300,
    metric_for_best_model = "wer",
    # metric_for_best_model = "eval_loss",

    # do *NOT* modify this part
    output_dir = OUT_DIR_PATH,
    predict_with_generate = True if not model_config_dic["set_temperature_0"] else False,
    report_to = ["wandb"],
    load_best_model_at_end = True,
    greater_is_better = False,
    push_to_hub = False,
    remove_unused_columns = False,
    label_names = ["labels"],
)

# evaluation related settings
evaluate_config_dic = {
    "eval_batch_size" : 16,

    # evaluate dataset
    # set use_exist_dataset to True when using HF datasets (e.g. AMI, LibriSpeech)
    # exist_dataset_path: [dataset_name, subset, split]
    # when use_exist_dataset is False, then will use local_dataset_path: a csv file
    "use_exist_dataset" : True,
    "exist_dataset_path" : ["edinburghcstr/ami", "ihm", "test"],
    # "local_dataset_path" : '/scratch/shared/whitehill/yguan2/isat_si_seg_data/raw/TEST/test.csv',
    "local_dataset_path" : '/scratch/shared/whitehill/yguan2/isat_si_seg_data/seg_padding/TEST/test.csv',

    # evaluate model
    # if directly using a pretrained model, set model_dir to that name and model_base_name to ""
    # model_path = os.path.join(model_dir, model_base_name)
    "model_dir" : OUT_DIR_PATH,
    "model_base_name" : "checkpoint-15300",
    # "model_dir" : "openai/whisper-tiny",
    # "model_base_name" : "",

    # lora related settings
    # lora_config_path is the adapter_model dir
    "use_int8" : False,
    "use_lora" : False,
    "lora_config_path" : "/scratch/shared/whitehill/yguan2/model/al_ft_ami_tiny/baseline_all_2/checkpoint-200/adapter_model",

    # evaluation methods
    "do_num2words": False,          # whether or not to spell out numbers
    "do_english_normalize" : True,     # do english normalize, if true, should assign normalize to false
    "do_normalize_eval" : False,
    "do_isat_normalize" : False,
    "max_new_tokens" : 112,
    "set_temperature_0" : True,

    # evaluation procedures and results
    # if only want to eval from an exist csv, only set eval_from_csv to True
    "do_eval" : True,
    "eval_from_csv" : True,
}


# # this method will be used both for training data and evaluate data
# def DataSet_from_manifest(csv_path, data_root, wav_col="wav", trans_col="transcript", split='train'):
#     """
#         Get manifest data from csv

#         csv_path: input data csv

#         data_root: $DATAROOT

#         wav_col: name of wav col, default as "wav"

#         trans_col: name of transcript col, default as "transcript"

#         split: the split of the data, default as "train"
#     """
#     metadf = pd.read_csv(csv_path)
#     ds = Dataset.from_dict(
#     {
#         "ID": metadf['ID'].astype(str),
#         "audio": metadf[wav_col].str.replace('$DATAROOT', data_root, regex=False).astype(str),
#         "transcript": metadf[trans_col].astype(str),
#         # "duration": metadf["duration"].astype(float)
#     },
#     split = split
#     ).cast_column("audio", Audio())
#     return ds

def DataSet_from_manifest(csv_path, data_root, wav_col="wav", trans_col="transcript", split='train'):
    """
        Get manifest data from csv

        csv_path: input data csv

        data_root: $DATAROOT

        wav_col: name of wav col, default as "wav"

        trans_col: name of transcript col, default as "transcript"

        split: the split of the data, default as "train"
    """
    metadf = pd.read_csv(csv_path)
    ds = Dataset.from_dict(
    {
        "ID": metadf['audio_id'].astype(str),
        "audio": metadf[wav_col].str.replace('$DATAROOT', data_root, regex=False).astype(str),
        config_dic["transcript_col"]: metadf[trans_col].astype(str),
        # "duration": metadf["duration"].astype(float)
    },
    split = split
    ).cast_column("audio", Audio())
    return ds
