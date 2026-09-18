"""
PHASE 1: Data Loading + Text Representation
--------------------------------------------
Converts raw complaint text into two numeric forms:
  1. TF-IDF vectors      -> used for classification (Phase 2, 5)
  2. Sentence embeddings -> used for similarity/clustering (Phase 3, 4, 6)

Run this first. Everyone else in the group should load the saved
outputs (.pkl / .npy files) instead of re-processing text themselves.
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "issuegraph_lite_complaints_dataset.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} complaints across {df['category'].nunique()} categories")
    return df


def build_tfidf(df, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams, e.g. "not working"
    )
    X_tfidf = vectorizer.fit_transform(df["complaint_text"])
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    return X_tfidf, vectorizer


def build_embeddings(df):
    """
    Sentence embeddings using a pretrained SBERT model.
    Captures MEANING, not just word overlap -- this is what lets
    "ceiling fan sounds like helicopter" match "fan vibrating badly".

    NOTE: requires `pip install sentence-transformers --break-system-packages`
    This downloads a small pretrained model (~80MB) the first time it runs.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["complaint_text"].tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    print(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    df = load_data()

    # --- TF-IDF (fast, always run this) ---
    X_tfidf, tfidf_vectorizer = build_tfidf(df)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open("X_tfidf.pkl", "wb") as f:
        pickle.dump(X_tfidf, f)

    # --- Embeddings (slower, needs sentence-transformers installed) ---
    try:
        embeddings = build_embeddings(df)
        np.save("embeddings.npy", embeddings)
        print("Saved embeddings.npy")
    except ImportError:
        print("sentence-transformers not installed -- skipping embeddings for now.")
        print("Install with: pip install sentence-transformers --break-system-packages")

    df.to_pickle("df_processed.pkl")
    print("\nPhase 1 complete. Saved: tfidf_vectorizer.pkl, X_tfidf.pkl, ""embeddings.npy (if available), df_processed.pkl")