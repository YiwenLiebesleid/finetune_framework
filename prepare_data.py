"""
    prepare datasets
    write your own prepare methods here
    use inside the pipeline will automatically assign the data path to config

    result: saving your data to a target csv file

    return: nothing, but will change the frame_config data paths
"""
from debug_util import debug_prt
from frame_config import model_config_dic, evaluate_config_dic
from transformers import WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict , Dataset, Audio
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import datasets
from tqdm import tqdm
import gc
import os
import jiwer


### define your methods here

"""
def udf_prep_data():

    ### write codes start

    BATCH_SIZE = 32
    OUT_DIR = "/scratch/shared/whitehill/yguan2/active_learning/al_score"
    out_file_dir = "whisper-tiny_score"
    out_file_name = "ami_dur_score_bak.csv"

    init_from_hub_path = "openai/whisper-tiny"
    lang = None
    model = WhisperForConditionalGeneration.from_pretrained(init_from_hub_path).cuda()

    DS = load_dataset("edinburghcstr/ami", "ihm", split="train")
    data_remove_cols = ["meeting_id", "audio_id", "microphone_id", "speaker_id"]
    DS = DS.remove_columns(data_remove_cols)
    dur_col = [DS[i]["end_time"] - DS[i]["begin_time"] for i in tqdm(range(len(DS)))]
    DS = DS.add_column(name="duration", column=dur_col)
    

    from transformers import WhisperFeatureExtractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(init_from_hub_path)
    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained(init_from_hub_path, language=lang, task="transcribe")
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(init_from_hub_path, language=lang, task="transcribe")
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    def prepare_dataset(datum):
        audio = datum["audio"]
        datum["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        datum["labels"] = tokenizer(datum["text"]).input_ids
        return datum

    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

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
    dataloader = DataLoader(DS, batch_size=BATCH_SIZE, collate_fn=data_collator)
    

    processed_data = []
    bcount = 0
    DATA_ROOT = "/home/yguan2/.cache/huggingface/datasets/downloads/extracted"
    debug_prt(DATA_ROOT, "[DATA] DATA_ROOT")
    for step, batch in enumerate(tqdm(dataloader)):
        audio_paths = [
            str(dataloader.dataset.data["audio"][bcount+i]["path"]).replace(DATA_ROOT, "$DATAROOT") 
            for i in range(len(batch['input_features']))]
        durations = [str(dataloader.dataset.data["duration"][bcount+i]) for i in range(len(batch['input_features']))]
        bcount += len(batch['input_features'])

        input_features = batch["input_features"]
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                out = model.generate(input_features.to("cuda"),
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            temperature=0,
                                            no_repeat_ngram_size=50,
                                    )
                generated_tokens = out.sequences.cpu().numpy()[:,1:]
                transition_score = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                
                # remove padding
                # ts_mean = np.ma.array(transition_score.cpu().numpy(), 
                #             mask=generated_tokens==processor.tokenizer.bos_token_id).mean(axis=-1).data
                
                ts_mean = []
                ts_sum = []
                for ix, gt in enumerate(generated_tokens):
                    if gt.tolist()[-1] == processor.tokenizer.bos_token_id:
                        eot_id = gt.tolist().index(processor.tokenizer.bos_token_id)
                    else:
                        eot_id = len(gt) - 1
                    ts_mean.append(transition_score[ix][0:eot_id+1].mean().item())
                    ts_sum.append(transition_score[ix][0:eot_id+1].sum().item())

                # COMPUTE WER
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                # do normalize
                decoded_preds_norm = [normalizer(pred) for pred in decoded_preds]
                decoded_labels_norm = [normalizer(label) for label in decoded_labels]
                
                wer_ls = [
                    jiwer.wer(decoded_labels_norm[ix], decoded_preds_norm[ix]) if len(decoded_labels_norm[ix]) > 0 else -1.0
                    for ix in range(len(decoded_labels_norm))
                ]

                

                processed_data.extend([
                    {
                        'audio': path,
                        'text': ref,
                        'hyp': hyp,
                        'duration': dur,
                        'ref_norm': r_n,
                        'hyp_norm': h_n,
                        'ave_score': ts_m,
                        'sum_score': ts_s,
                        'wer': wer,
                    }
                    for ix, (path, ref, hyp, dur, r_n, h_n, ts_m, ts_s, wer) in enumerate(
                        zip(audio_paths, decoded_labels, decoded_preds, durations, 
                            decoded_labels_norm, decoded_preds_norm, 
                            ts_mean, ts_sum, wer_ls)
                    )
                ])

        del out, labels, batch, ts_mean
        gc.collect()
    
    ASR_dir = os.path.join(OUT_DIR, out_file_dir)
    os.makedirs(ASR_dir, exist_ok = True)
    ASR_file_path = os.path.join(ASR_dir, out_file_name)
    pd.DataFrame.from_records(processed_data).to_csv(ASR_file_path)
    debug_prt(f'saved processed_data to {ASR_file_path}', "[DATA][STATUS]")


    # ### write codes end

    # # your correct data path
    # train_path = ""
    # val_path = ""
    # test_path = ""

    # ### do not modify this part
    # debug_prt(train_path, "[DATA] train_path")
    # debug_prt(val_path, "[DATA] val_path")
    # debug_prt(test_path, "[DATA] test_path")

    # model_config_dic["local_dataset_path"] = [train_path, val_path, test_path]
    # evaluate_config_dic["local_dataset_path"] = test_path
"""



def get_time_range_data(input_df, desired_total):
    dur_list = input_df["duration"].tolist()
    n = len(dur_list)
    debug_prt(n, "[DATA] length n")
    curr_total = 0
    i = 0
    while i < n:
        if curr_total >= desired_total:  break
        curr_total += dur_list[i]
        i += 1
    return curr_total, i

# get  desire_total hours data
# method: random select
def RS(df, desired_total):
    df_shuffled = df.sample(frac=1)
    curr_total, i = get_time_range_data(df_shuffled, desired_total)
    return curr_total, i, df_shuffled[0:i]

# method: top N
def topN(df, desired_total, col_name, ascending=True):
    df_sorted = df.sort_values(by=[col_name], ascending=ascending)
    curr_total, i = get_time_range_data(df_sorted, desired_total)
    return curr_total, i, df_sorted[0:i]

# method: score range [lower, upper]
def scoreRange(df, desired_total, col_name, lower, upper):
    df_sliced = df[(df[col_name] >= lower) & (df[col_name] <= upper)]
    curr_total, i = get_time_range_data(df_sliced, desired_total)
    return curr_total, i, df_sliced[0:i]

# do regularization on score, reallocate into [0,1]
def do_regularize(df, col_name):
    max_val = df[col_name].max()
    min_val = df[col_name].min()
    df[col_name] = (df[col_name] - min_val) / (max_val - min_val)
    return df


def new_udf_prep_data(prep_option="topN", bounds=[0.5,1.0], col_name="ave_score", desired_total=3600, regularize=True, ascending=True):
    """
        prep_option:
        1. topN: get the largest N samples that make up desired_total time
        2. range: get samples from score range [lower, upper] that make up desired total time
        3. RS: random select

        can be done in combination: range+topN, range+RS
    """

    csvdata_path = "/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/ami_ihm_with_duration_and_score.csv"
    debug_prt(csvdata_path, "[DATA][STATUS]: read from csv path")

    df = pd.read_csv(csvdata_path)
    debug_prt(desired_total, "[DATA] desired total")

    if regularize is True:
        df = do_regularize(df, col_name)
        debug_prt("do regularization", "[DATA][STATUS]")

    if prep_option == "range":
        lower, upper = bounds
        curr_total, i, df_processed = scoreRange(df, desired_total, col_name, lower, upper)
        df_processed = df_processed.sample(frac=1)      # shuffle
        debug_prt(f"do range [{lower}, {upper}]", "[DATA][STATUS]")
    elif prep_option == "RS":
        curr_total, i, df_processed = RS(df, desired_total)
        debug_prt("do RS", "[DATA][STATUS]")
    elif prep_option == "topN":
        curr_total, i, df_processed = topN(df, desired_total, col_name, ascending)
        df_processed = df_processed.sample(frac=1)      # shuffle
        debug_prt(f"do topN {col_name}", "[DATA][STATUS]")
    elif prep_option == "range+RS":
        lower, upper = bounds
        df_sliced = df[(df[col_name] >= lower) & (df[col_name] <= upper)]
        curr_total, i, df_processed = RS(df_sliced, desired_total)
        debug_prt(f"do range + RS {col_name}", "[DATA][STATUS]")
    elif prep_option == "range+topN":
        lower, upper = bounds
        df_sliced = df[(df[col_name] >= lower) & (df[col_name] <= upper)]
        curr_total, i, df_processed = topN(df_sliced, desired_total, col_name, ascending)
        df_processed = df_processed.sample(frac=1)      # shuffle
        debug_prt(f"do range + topN {col_name}", "[DATA][STATUS]")
    
    debug_prt(curr_total, "[DATA] current total")
    debug_prt(i, "[DATA] current number")

    save_filename = prep_option
    if "range" in prep_option:
        lower, upper = bounds
        save_filename = save_filename + f"_{lower}_{upper}"
    save_filename = save_filename + f"_{col_name}"
    savedata_path = f"/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/AMI_{save_filename}.csv"
    df_processed.to_csv(savedata_path, index=False)
    debug_prt(savedata_path, "[DATA][STATUS]: save to csv path")


def new_udf_prep_data_2(col_name="ave_score", bounds=[0,0], RS_time=1800, topN_time=1800, ascending=True):
    """
        select topN samples along with RS samples
        mix 2 sources of data
        to make up a same dataset
    """
    csvdata_path = "/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/ami_ihm_with_duration_and_score.csv"
    debug_prt(csvdata_path, "[DATA][STATUS]: read from csv path")

    df = pd.read_csv(csvdata_path)
    debug_prt(col_name, "[DATA] col_name")
    
    # RS
    _, _, df_random = RS(df, RS_time)
    debug_prt("do RS", "[DATA][STATUS]")

    # top N
    lower, upper = bounds
    df_sliced = df[(df[col_name] >= lower) & (df[col_name] <= upper)]
    _, _, df_top = topN(df_sliced, topN_time, col_name, ascending)
    
    # concate
    df_processed = pd.concat([df_top, df_random], axis=0).drop_duplicates()

    df_processed = df_processed.sample(frac=1)      # shuffle
    debug_prt(f"do range + topN {col_name}", "[DATA][STATUS]")
    
    curr_total = df_processed["duration"].sum()
    i = df_processed.count()[0]

    debug_prt(curr_total, "[DATA] current total")
    debug_prt(i, "[DATA] current number")

    save_filename = "mix_topN_RS"
    lower, upper = bounds
    save_filename = save_filename + f"_{lower}_{upper}"
    save_filename = save_filename + f"_{col_name}"
    savedata_path = f"/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/AMI_{save_filename}.csv"
    df_processed.to_csv(savedata_path, index=False)
    debug_prt(savedata_path, "[DATA][STATUS]: save to csv path")


def pseudo_labeling_prep_data(col_name, desired_total):
    csvdata_path = "/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/ami_with_hyp.csv"
    debug_prt(csvdata_path, "[DATA][STATUS]: read from csv path")
    df = pd.read_csv(csvdata_path)
    debug_prt(col_name, "[DATA] col_name")

    if col_name == "wer":
        # should not taking wer=0 samples (it's cheating)
        df_sliced = df[(df[col_name] > 0)]
        # taking topN samples with smallest WER > 0
        curr_total, i, df_processed = topN(df_sliced, desired_total, col_name, ascending=True)
    elif col_name == "ave_score" or col_name == "sum_score":
        # select those with highest probabilities
        curr_total, i, df_processed = topN(df, desired_total, col_name, ascending=False)
    
    # rename to make hyp the label
    df_processed = df_processed.rename(columns={
        "text": "text_real",
        "hyp": "text"
    })

    # shuffle
    df_processed = df_processed.sample(frac=1)
    debug_prt(curr_total, "[DATA] current total")
    debug_prt(i, "[DATA] current number")

    save_filename = "PL"
    save_filename = save_filename + f"_{col_name}"
    savedata_path = f"/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/AMI_{save_filename}.csv"
    df_processed.to_csv(savedata_path, index=False)
    debug_prt(savedata_path, "[DATA][STATUS]: save to csv path")


def udf_prep_data():
    new_udf_prep_data(
            prep_option="RS",
            # col_name="ave_score",
            col_name="wer",
            # bounds=[0,0],
            bounds=[0,1],
            desired_total=60*60*10,
            regularize=False,
            ascending = False
    )
    
    # new_udf_prep_data_2(
    #     col_name="wer", 
    #     bounds=[0,0.5], 
    #     RS_time=60*60*3+600, 
    #     topN_time=60*60*7+600,
    #     ascending=False
    # )

    # pseudo_labeling_prep_data(col_name="wer", 
    #                           desired_total=60*60*10)



if __name__ == "__main__":
    new_udf_prep_data(
                        prep_option="range",
                        col_name="ave_score",
                        bounds=[0.3,0.7],
                        desired_total=3600,
                        regularize=True
                      )