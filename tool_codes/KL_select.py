import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm
import math
from debug_util import debug_prt

df_word_cnt = pd.read_csv("/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/ami_with_hyp.csv")
df_word_cnt = df_word_cnt.drop(columns=["Unnamed: 0","hyp","hyp_norm","ave_score","sum_score"])

texts = list(df_word_cnt["ref_norm"])
texts = [texts[i].split() for i in range(len(texts))]
word_ind_dict = dict()
word_freq_train = []
ind = 0
for text in texts:
    words = text
    for word in words:
        if word in word_ind_dict:
            word_freq_train[word_ind_dict[word]] += 1
            continue
        else:
            word_ind_dict[word] = ind
            word_freq_train.append(1)
            ind += 1
print(ind)
print(len(word_freq_train))

def KLD(x, y, axis=0):
    return scipy.stats.entropy(x, y, axis=axis)

def saveFile(index, df_word_cnt, select, aud_ids):
    id_select = np.array(aud_ids)[list(select)]
    df_sampled = df_word_cnt[df_word_cnt["audio_id"].isin(id_select)]
    path = f"/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/GT_based/KL_WER_0_10h_{index}.csv"
    df_sampled.to_csv(path, index=False)
    print(f"save to path: {path}")



### for checkpoint
# arr = np.zeros((len(df_word_cnt), len(word_freq_train)))
# for i in range(len(texts)):
#     for word in texts[i]:
#         arr[i][word_ind_dict[word]] += 1

### TODO
### 
df_word_cnt = df_word_cnt[df_word_cnt["wer"]>=0]
### 

aud_ids = list(df_word_cnt["audio_id"])
durations = list(df_word_cnt["duration"])
texts = list(df_word_cnt["ref_norm"])
texts = [texts[i].split() for i in range(len(texts))]


### load from checkpoint
# df_check = pd.read_csv("/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/GT_based/KL_WER_gt_0_10h_8.0_new.csv")
# check_aud_ids = list(df_check["audio_id"])
# check_select = set(check_aud_ids)
# select = set()
# for id, ind in enumerate(aud_ids):
#     if ind in check_select:
#         select.add(id)
# curr_dur = df_check["duration"].sum()
# curr_freq = np.zeros((1,len(word_freq_train)))
# fT = np.sum(arr, axis=1)        # will not change
# curr_fT = 0
# prev_KL = 0
# for ind in check_aud_ids:
#     curr_freq = curr_freq + arr[ind]
#     curr_fT += fT[ind]
# prev_KL = KLD(curr_freq, word_freq_train, axis=-1)
# print(curr_dur, curr_fT, prev_KL)


arr = np.zeros((len(df_word_cnt), len(word_freq_train)))
for i in range(len(texts)):
    for word in texts[i]:
        arr[i][word_ind_dict[word]] += 1
fT = np.sum(arr, axis=1)        # will not change

# ## start from 0
select = set()
curr_dur = 0
curr_freq = np.zeros((1,len(word_freq_train)))
fT = np.sum(arr, axis=1)        # will not change
curr_fT = 0
prev_KL = 0

save_file_dur = 0

with tqdm(total=3600 * 10) as pbar:
    pbar.update(curr_dur)
    
    while curr_dur < 3600 * 10:
        
        KL = KLD(curr_freq + arr, word_freq_train, axis=-1)
        if curr_dur == 0:
            log_fT_diff = np.log(fT)
        else:
            log_fT_diff = np.log(fT + curr_fT) - np.log(curr_fT)
        res = log_fT_diff - (KL - prev_KL)
        if len(select) > 0:
            res[list(select)] = -float('inf')
        
        ind = np.argmax(res)
        select.add(ind)
        curr_dur += durations[ind]
        curr_freq = curr_freq + arr[ind]
        curr_fT += fT[ind]
        prev_KL = KL[ind]
        debug_prt(ind, f"Add {durations[ind]}")

        save_file_dur += durations[ind]
        if save_file_dur >= 3600:   # checkpoints
            save_file_dur = 0
            saveFile(curr_dur // 3600 - 1, df_word_cnt, select, aud_ids)

        pbar.update(durations[ind])

id_select = np.array(aud_ids)[list(select)]
df_sampled = df_word_cnt[df_word_cnt["audio_id"].isin(id_select)]
path = "/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/GT_based/KL_WER_0_10h.csv"
df_sampled.to_csv(path, index=False)
print(f"save to path: {path}")
