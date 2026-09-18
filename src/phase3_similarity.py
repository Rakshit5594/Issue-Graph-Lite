"""
PHASE 3: Similar Historical Incidents + Duplicate Detection
--------------------------------------------------------------
Given a new complaint, find the most semantically similar past
complaints and flag likely duplicates.

TWO VERSIONS below:
  A) TF-IDF + cosine similarity  -> works right now, no extra installs
  B) Sentence embeddings + cosine similarity -> better at catching
     paraphrases with ZERO shared words (e.g. "helicopter" vs "vibrating")
     Requires: pip install sentence-transformers --break-system-packages
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle("df_processed.pkl")

with open("X_tfidf.pkl", "rb") as f:
    X_tfidf = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

DUPLICATE_THRESHOLD = 0.85


# ---------------------------------------------------------------
# VERSION A: TF-IDF based (works out of the box)
# ---------------------------------------------------------------
def find_similar_tfidf(new_text, top_n=5):
    new_vec = tfidf_vectorizer.transform([new_text])
    sims = cosine_similarity(new_vec, X_tfidf).flatten()
    top_idx = sims.argsort()[::-1][:top_n]

    results = []
    for idx in top_idx:
        results.append({
            "complaint_id": df.iloc[idx]["complaint_id"],
            "text": df.iloc[idx]["complaint_text"],
            "category": df.iloc[idx]["category"],
            "similarity": round(float(sims[idx]), 3),
            "is_duplicate": bool(sims[idx] >= DUPLICATE_THRESHOLD),
        })
    return results


# ---------------------------------------------------------------
# VERSION B: Embedding based (better semantic matching)
# ---------------------------------------------------------------
def find_similar_embeddings(new_text, top_n=5):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.load("embeddings.npy")  # from Phase 1

    new_vec = model.encode([new_text])
    sims = cosine_similarity(new_vec, embeddings).flatten()
    top_idx = sims.argsort()[::-1][:top_n]

    results = []
    for idx in top_idx:
        results.append({
            "complaint_id": df.iloc[idx]["complaint_id"],
            "text": df.iloc[idx]["complaint_text"],
            "category": df.iloc[idx]["category"],
            "similarity": round(float(sims[idx]), 3),
            "is_duplicate": bool(sims[idx] >= DUPLICATE_THRESHOLD),
        })
    return results


def evaluate_duplicate_detection(sample_size=500):
    """
    Uses the ground-truth duplicate_group_id column to check how well
    similarity search recovers TRUE duplicates. Good evidence to put
    in your report/demo.
    """
    sample = df.sample(sample_size, random_state=1)
    correct = 0
    for _, row in sample.iterrows():
        results = find_similar_tfidf(row["complaint_text"], top_n=5)
        # skip the exact same row if it appears
        matches = [r for r in results if r["complaint_id"] != row["complaint_id"]]
        if not matches:
            continue
        top_match_id = matches[0]["complaint_id"]
        top_match_group = df.loc[df["complaint_id"] == top_match_id, "duplicate_group_id"].values[0]
        if top_match_group == row["duplicate_group_id"]:
            correct += 1
    print(f"Top-1 same-issue match accuracy: {correct}/{sample_size} = {correct/sample_size:.2%}")


if __name__ == "__main__":
    example = "Ceiling fan sounds like a helicopter and vibrates a lot"
    print(f"New complaint: {example!r}\n")

    print("--- TF-IDF similarity results ---")
    for r in find_similar_tfidf(example):
        print(r)

    print("\n--- Evaluating duplicate detection against ground truth ---")
    evaluate_duplicate_detection(sample_size=300)