"""
PHASE 4: Root-Cause Clusters
--------------------------------
Unsupervised clustering of complaints WITHIN each category, so that
"fan noise" and "fan not working" split into separate root-cause
clusters even though both are HVAC/Fan complaints.

Uses TF-IDF vectors here for portability (no extra installs needed).
Swap in embeddings.npy for better results if you have sentence-transformers.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

df = pd.read_pickle("df_processed.pkl")
with open("X_tfidf.pkl", "rb") as f:
    X_tfidf = pickle.load(f)

cluster_ids = pd.Series([""] * len(df), index=df.index, dtype=object)
kmeans_models = {}   # category -> fitted KMeans, so NEW complaints can be assigned a cluster later

for category in df["category"].unique():
    idx = df.index[df["category"] == category]
    X_cat = X_tfidf[idx]

    # number of clusters = number of true issue_subtypes in this category
    # (in a real deployment you wouldn't know this -- use elbow method
    #  or silhouette score to pick k instead)
    n_clusters = df.loc[idx, "issue_subtype"].nunique()

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_cat)

    cluster_ids.loc[idx] = [f"{category}_cluster_{l}" for l in labels]
    kmeans_models[category] = km

df["cluster_id"] = cluster_ids

# save the fitted per-category KMeans models so the integration pipeline
# can assign a cluster to a brand-new, unseen complaint
with open("kmeans_models.pkl", "wb") as f:
    pickle.dump(kmeans_models, f)
    print("kmeans created successfully")

# human-readable label for each cluster = its most common true issue_subtype
cluster_labels = (
    df.groupby("cluster_id")["issue_subtype"]
    .agg(lambda s: s.value_counts().idxmax())
    .to_dict()
)
with open("cluster_labels.pkl", "wb") as f:
    pickle.dump(cluster_labels, f)

# --- Evaluate against ground truth issue_subtype using Adjusted Rand Index ---
# ARI = 1.0 means perfect agreement with true root-cause labels, 0 = random
ari = adjusted_rand_score(df["issue_subtype"], df["cluster_id"])
print(f"Adjusted Rand Index vs true issue_subtype: {ari:.3f}")

print("\nSample cluster assignments:")
print(df[["complaint_text", "category", "issue_subtype", "cluster_id"]].sample(8, random_state=1))

df.to_pickle("df_clustered.pkl")
print("\nSaved df_clustered.pkl")