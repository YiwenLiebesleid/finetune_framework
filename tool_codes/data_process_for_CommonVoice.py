from debug_util import debug_prt
from datasets import load_dataset, DatasetDict , Dataset, Audio
import pandas as pd
import datasets
from tqdm import tqdm
from pydub import AudioSegment
import re

debug_prt("load dataset", "[STATUS]")
DS = load_dataset("mozilla-foundation/common_voice_13_0", "en")

DS = DS["train"]
data_remove_cols = ["up_votes", "client_id", "down_votes", "age", "gender", "accent", "locale", "segment", "variant"]
DS = DS.remove_columns(data_remove_cols)
DS = DS.rename_column("sentence", "text")

bad_id = 733462
debug_prt(bad_id, "bad_id")

debug_prt("process data...", "[STATUS]")


debug_prt("do normalization", "[STATUS]")
def normalize_utterance(utt):
    utt = re.sub("[\(\[\<].*?[\)\]\>]+", "", utt)
    patt = "([^a-zA-Z0-9 '])*"
    utt = re.sub(patt, "", utt)
    return utt.strip().upper()

summary_df = pd.DataFrame(columns = ["audio_id", "audio", "text", "duration"])
DATAROOT = "/home/yguan2/.cache/huggingface/datasets/downloads/extracted"
row = 0
for ind in tqdm(range(len(DS))):
    if ind == bad_id:
        continue
    audio = DS[ind]["path"]
    duration = DS[ind]["audio"]["array"].size / 48000

    text = DS[ind]["text"]
    text = normalize_utterance(text)
    if not text.strip():
        # remove empty utts
        continue
    
    wav = audio.replace(DATAROOT, "$DATAROOT")
    summary_df.loc[row] = [row, wav, text, duration]
    row += 1
summary_path = "/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/commom_voice_dur_score.csv"
summary_df.to_csv(summary_path, index=False)

debug_prt("Finished!", "[STATUS]")
