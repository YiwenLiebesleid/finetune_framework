import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm

from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# df = pd.read_csv("/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/ami_with_hyp.csv")

# n_clusters = 500
# # do kmeans on text (references)
# sents_ref = list(df["text"])
# embs_ref = model.encode(sents_ref)
# cluster_ref = KMeans(n_clusters=n_clusters, n_init=5, verbose=0).fit(embs_ref)
# centroid_ref = cluster_ref.cluster_centers_

# y_pred_ref = cluster_ref.labels_
# print(y_pred_ref, centroid_ref)

# def Euclidean_distance(A, B):
#     return norm(A - B)
# df["ref_kmeans_id"] = y_pred_ref
# dists = [0] * len(df)
# for i in range(len(df)):
#     k_id = df["ref_kmeans_id"][i]
#     dist = Euclidean_distance(embs_ref[i], centroid_ref[k_id])
#     dists[i] = dist
# df["ref_euc_dist"] = dists
# # save cluster info to file
# df.to_csv("/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/ami_with_hyp+text_KMEANS.csv", index=False)



# load data
df = pd.read_csv("/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/ami_with_hyp+text_KMEANS.csv")
df = df.drop(columns=["Unnamed: 0", "sum_score", "hyp_kmeans_id", "hyp_euc_dist"])

### Random Selection
df = df.sample(frac=1)
dur = np.array(df["duration"])
aud_id = np.array(df["audio_id"])
desired_total = 10*60*60
select = []
curr_total = 0
ind = 0
while curr_total < desired_total:
    curr_total += dur[ind]
    ind += 1
select = aud_id[0:ind]
print(curr_total, len(select))
df_sampled = df[df["audio_id"].isin(select)]

# ### method 1: duration %
# n_clusters = 500
# # rate = 17.1 / 78
# rate = 10 / 78
# select = []
# curr_total = 0
# for i in tqdm(range(n_clusters)):
#     df_sliced = df[df["ref_kmeans_id"] == i]
#     df_sliced = df_sliced.sort_values(by=["ref_euc_dist"], ascending=True)
#     dur = np.array(df_sliced["duration"])
#     aud_id = np.array(df_sliced["audio_id"])
#     cluster_total_len = dur.sum()
#     cluster_max_pick_len = cluster_total_len * rate
#     curr_cluster_total = 0
#     for j in range(len(dur)):
#         curr_cluster_total += dur[j]
#         select.append(aud_id[j])
#         if curr_cluster_total >= cluster_max_pick_len:
#             break
#     curr_total += curr_cluster_total

# ### method 2: RS from each cluster %
# n_clusters = 500
# rate = 10 / 78
# select = []
# curr_total = 0
# for i in tqdm(range(n_clusters)):
#     df_sliced = df[df["ref_kmeans_id"] == i]
#     df_sliced = df_sliced.sample(frac=1)        # Random Selection
#     dur = np.array(df_sliced["duration"])
#     aud_id = np.array(df_sliced["audio_id"])
#     cluster_total_len = dur.sum()
#     cluster_max_pick_len = cluster_total_len * rate
#     curr_cluster_total = 0
#     for j in range(len(dur)):
#         curr_cluster_total += dur[j]
#         select.append(aud_id[j])
#         if curr_cluster_total >= cluster_max_pick_len:
#             break
#     curr_total += curr_cluster_total

print(curr_total, len(select))
df_sampled = df[df["audio_id"].isin(select)]
df_sampled.to_csv("/scratch/shared/whitehill/yguan2/active_learning/al_score/whisper-tiny_score/GT_based/AMI_RS_10h.csv", index=False)
