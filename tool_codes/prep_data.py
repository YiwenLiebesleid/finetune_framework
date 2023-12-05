import numpy as np
import os
import csv
import pandas as pd
import re
from pydub import AudioSegment
import contextlib
from tqdm import tqdm
from debug_util import debug_prt
import sys

# read inputs
_, data_split, pd_len = sys.argv

"""
    Basic setups
    data split: TRAIN/TEST/DEV
    root dir: raw/seg_padding
    debug: True/False
    padding length: positive float value indicating seconds (not ms)
"""
pd_len = float(pd_len)
if pd_len == 0.0:   root_dir = "raw"
else: root_dir = "seg_padding" + "_" + str(pd_len)
debug = True
if debug:
    debug_prt(debug, "debug")
    debug_prt(data_split, "data split")
    debug_prt(root_dir, "root dir")

configure = {
    "channels": 1,
    "sample_width": 2,
    "sample_rate": 16000,
    "bit_width": 16,
    # for padding problem
    "padding": True,  # default should be False
    "padding_length": pd_len,  # sec, default should be 0
}
if root_dir == "raw":
    configure["padding"] = False
    configure["padding_length"] = 0
if debug:
    if configure["padding"]:
        debug_prt(f"{configure['padding_length']} sec", "padding length")
    else:
        debug_prt("padding: off", "padding")

# options for filtering audio
filters = {
    "remove_nontarget": True,
    "remove_etc": True,
    "remove_blank": True,
    "normalization": True
}


# Define data path and generate directories
# input path and target path
paths = {
    "input_data": f"/scratch/shared/whitehill/datasets/ISAT-SI/{data_split}/full/WAVs",
    "input_label": f"/scratch/shared/whitehill/datasets/ISAT-SI/{data_split}/full/utt_labels",
    "target": f"/scratch/shared/whitehill/yguan2/isat_si_seg_data/{root_dir}/{data_split}"
}

paths["target_wav"] = os.path.join(paths["target"], "WAVs")
debug_prt(f"{paths['target_wav']}", "target_wav")
if not os.path.exists(paths["target_wav"]):
    os.makedirs(paths["target_wav"])
    debug_prt(f"mkdir {paths['target_wav']}", "STATUS")

paths["target_trans"] = os.path.join(paths["target"], "transcripts")
debug_prt(f"{paths['target_trans']}", "target_trans")
if not os.path.exists(paths["target_trans"]):
    os.makedirs(paths["target_trans"])
    debug_prt(f"mkdir {paths['target_trans']}", "STATUS")


# Define filtering and normalizing methods
def remove_in_brackets(text):
    # removes any clause in brackets or parens, and the brackets themselves
    return re.sub("[\(\[\<].*?[\)\]\>]+", " ", text)

def normalize_utterance(utt):
    utt = re.sub("[\(\[\<].*?[\)\]\>]+", "", utt)
    patt = "([^a-zA-Z0-9 '])*"
    utt = re.sub(patt, "", utt)
    return utt.strip().upper()


# process data
# required rows: ['ID','duration','wav','transcript']
summary_df = pd.DataFrame(columns=["ID", "duration", "wav", "transcript"])
row = 0

filelist = os.listdir(paths["input_data"])
for filename in tqdm(filelist):
    sessname = filename.split(".")[0]
    
    # load audio file
    wav_file_path = os.path.join(paths["input_data"], f"{sessname}.wav")
    sess_audio = AudioSegment.from_wav(wav_file_path)
    total_duration = sess_audio.duration_seconds
    
    # load label file
    label_file_path = os.path.join(paths["input_label"], f"{sessname}.csv")
    with open(label_file_path) as in_file:
        reader = csv.reader(in_file, delimiter=",")
        next(reader)  # skip header
        for s, utt in enumerate(reader):
            speaker, utterance, start_sec, end_sec = utt
            start_sec, end_sec = float(start_sec), float(end_sec)
            
            # add padding
            if configure["padding"]:
                pad_len = configure["padding_length"]
                start_sec = max(0, start_sec - pad_len)
                end_sec = min(total_duration, end_sec + pad_len)
            
            dur = end_sec - start_sec
            
            # apply filters
            # remove nontarget speech
            if filters["remove_nontarget"]:
                if not "student" in speaker.lower() or "student-other" in speaker.lower():
                    # debug_prt(f"sess {sessname} segment {s}: removing utt from speaker {speaker}")
                    continue
            # filter out inaudible etc
            if filters["remove_etc"]:
                if not remove_in_brackets(utterance).strip():
                    # debug_prt(f"sess {sessname} segment {s}: removing utt containing only inaudible/whispering/singing etc")
                    continue
            # filter out empty utterances
            if filters['remove_blank']:
                if not utterance.strip():
                    # debug_prt(f"sess {sessname} segment {s}: removing utterance with blank transcript")
                    continue
            
            # normalize utterance => delete special characters, to upper
            # TODO: also need to remove empty utterances
            if filters['normalization']:
                utterance = normalize_utterance(utterance)
            
            # export to wav file
            seg_audio = sess_audio[start_sec * 1000: end_sec * 1000]
            seg_wav_path = os.path.join(paths["target_wav"], f"{sessname}_{s}.wav")
            seg_audio.export(seg_wav_path, format="wav")
            
            # export transcript to text file
            seg_transcript_path = os.path.join(paths["target_trans"], f"{sessname}_{s}.txt")
            with open(seg_transcript_path, 'w') as outfile:
                outfile.write(utterance)
            
            # save the summary information to one csv xxxxx.csv
            ID = row
            duration = dur
            # $DATAROOT = "/scratch/shared/whitehill/yguan2/isat_si_seg_data/"
            wav = f"$DATAROOT/{root_dir}/{data_split}/WAVs/{sessname}_{s}.wav"
            transcript = utterance
            summary_df.loc[row] = [ID, duration, wav, transcript]
            row += 1

summary_path = os.path.join(paths["target"], f"{data_split.lower()}.csv")
debug_prt(summary_path, "summary path")
summary_df.to_csv(summary_path, index=False)
