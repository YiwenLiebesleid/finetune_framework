import pandas as pd
import numpy as np

# load data
df = pd.read_csv("/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/ami_with_hyp.csv")

S = set()       # sentence set
select = []
curr_duration = 0

# shuffle
df_shuffled = df.sample(frac=1)
ids = np.array(df_shuffled["audio_id"])
texts = np.array(df_shuffled["text"])
durations = np.array(df_shuffled["duration"])

n = len(df)
i = 0
while curr_duration < 36000 and i < n:
    if texts[i] in S:
        i += 1
        continue
    S.add(texts[i])
    select.append(ids[i])
    curr_duration += durations[i]
    i += 1
print(f"cnt: {len(select)}, duration: {curr_duration}")

df_sampled = df_shuffled[df_shuffled["audio_id"].isin(select)]
df_sampled.to_csv("/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/GT_based/RS_unique_sentence_10h.csv", index=False)
