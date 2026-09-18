"""
INTEGRATION PIPELINE — IssueGraph-Lite
==========================================
Ties all 6 phases together. Takes ONE raw complaint as input and runs
it through the full system:

    category  -> similar/duplicate complaints -> root-cause cluster
    -> escalation priority -> knowledge graph update

Run this AFTER phase1, phase2, phase4, phase5, and phase6 have each
been run once (they save the model/vector files this script loads).

Usage:
    python3 integration_pipeline.py
    (edit the `new_complaint` variable at the bottom, or import
    IssueGraphPipeline into your Streamlit app / demo script)
"""

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

DUPLICATE_THRESHOLD = 0.85
SIMILAR_THRESHOLD = 0.35
TOP_N_SIMILAR = 5


class IssueGraphPipeline:
    def __init__(self):
        print("Loading models and data...")

        with open("tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open("X_tfidf.pkl", "rb") as f:
            self.X_tfidf = pickle.load(f)
        with open("category_classifier.pkl", "rb") as f:
            self.category_clf = pickle.load(f)
        with open("priority_classifier.pkl", "rb") as f:
            self.priority_clf = pickle.load(f)
        with open("kmeans_models.pkl", "rb") as f:
            self.kmeans_models = pickle.load(f)   # category -> fitted KMeans
        with open("cluster_labels.pkl", "rb") as f:
            self.cluster_labels = pickle.load(f)  # cluster_id -> readable name

        self.df = pd.read_pickle("df_clustered.pkl")

        # try to load real sentence embeddings if available; fall back to TF-IDF
        try:
            self.embeddings = np.load("embeddings.npy")
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.use_embeddings = True
            print("Using sentence embeddings for similarity search.")
        except (FileNotFoundError, ImportError):
            self.embeddings = None
            self.embed_model = None
            self.use_embeddings = False
            print("Embeddings not available -- falling back to TF-IDF for similarity search.")
            print("(Run: pip install sentence-transformers, then re-run phase1 to enable this.)")

        # running knowledge graph, built up as new complaints come in.
        # seeded with a small random sample so it's not empty on first run.
        self.graph = nx.Graph()
        self._seed_graph(sample_size=40)

        self._next_new_id = int(self.df["complaint_id"].max()) + 1
        print("Pipeline ready.\n")

    # -----------------------------------------------------------------
    def _seed_graph(self, sample_size=40):
        sample = self.df.sample(sample_size, random_state=1)
        for _, row in sample.iterrows():
            self._add_complaint_node(row["complaint_id"], row["complaint_text"],
                                      row["category"], row["cluster_id"])

    def _add_complaint_node(self, complaint_id, text, category, cluster_id):
        c_node = f"C{complaint_id}"
        cat_node = f"CAT::{category}"
        clu_node = f"CLU::{cluster_id}"
        self.graph.add_node(c_node, kind="complaint", label=text[:40])
        self.graph.add_node(cat_node, kind="category", label=category)
        self.graph.add_node(clu_node, kind="cluster",
                             label=self.cluster_labels.get(cluster_id, cluster_id))
        self.graph.add_edge(c_node, cat_node, kind="belongs_to")
        self.graph.add_edge(c_node, clu_node, kind="belongs_to")
        return c_node

    # -----------------------------------------------------------------
    def _predict_category(self, text):
        vec = self.vectorizer.transform([text])
        return self.category_clf.predict(vec)[0], vec

    def _predict_priority(self, text):
        vec = self.vectorizer.transform([text])
        return self.priority_clf.predict(vec)[0]

    def _assign_cluster(self, category, tfidf_vec):
        km = self.kmeans_models.get(category)
        if km is None:
            return None
        cluster_num = km.predict(tfidf_vec)[0]
        cluster_id = f"{category}_cluster_{cluster_num}"
        return cluster_id

    def _find_similar(self, text, tfidf_vec, top_n=TOP_N_SIMILAR):
        if self.use_embeddings:
            new_vec = self.embed_model.encode([text])
            sims = cosine_similarity(new_vec, self.embeddings).flatten()
        else:
            sims = cosine_similarity(tfidf_vec, self.X_tfidf).flatten()

        top_idx = sims.argsort()[::-1][:top_n]
        results = []
        for idx in top_idx:
            row = self.df.iloc[idx]
            results.append({
                "complaint_id": int(row["complaint_id"]),
                "text": row["complaint_text"],
                "category": row["category"],
                "similarity": round(float(sims[idx]), 3),
                "is_duplicate": bool(sims[idx] >= DUPLICATE_THRESHOLD),
            })
        return results

    # -----------------------------------------------------------------
    def process(self, text):
        """Run one new complaint through the full pipeline and return a report."""
        category, tfidf_vec = self._predict_category(text)
        priority = self._predict_priority(text)
        cluster_id = self._assign_cluster(category, tfidf_vec)
        cluster_label = self.cluster_labels.get(cluster_id, "unknown")
        similar = self._find_similar(text, tfidf_vec)

        duplicates = [s for s in similar if s["is_duplicate"]]

        # add this new complaint into the running knowledge graph
        new_id = self._next_new_id
        self._next_new_id += 1
        c_node = self._add_complaint_node(new_id, text, category, cluster_id)
        for s in similar:
            if s["similarity"] >= SIMILAR_THRESHOLD:
                other_node = f"C{s['complaint_id']}"
                if other_node in self.graph:
                    self.graph.add_edge(c_node, other_node, kind="similar_to",
                                         weight=s["similarity"])

        report = {
            "complaint_id": new_id,
            "complaint_text": text,
            "predicted_category": category,
            "predicted_priority": priority,
            "root_cause_cluster": cluster_label,
            "similar_complaints": similar,
            "duplicate_of": duplicates,
            "is_likely_duplicate": len(duplicates) > 0,
        }
        return report

    def graph_stats(self):
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }


def print_report(report):
    print("=" * 70)
    print(f"NEW COMPLAINT #{report['complaint_id']}: {report['complaint_text']!r}")
    print("-" * 70)
    print(f"Predicted category : {report['predicted_category']}")
    print(f"Predicted priority : {report['predicted_priority']}")
    print(f"Root-cause cluster : {report['root_cause_cluster']}")
    print(f"Likely duplicate?  : {report['is_likely_duplicate']}")
    print("\nTop similar historical complaints:")
    for s in report["similar_complaints"]:
        dup_flag = " [DUPLICATE]" if s["is_duplicate"] else ""
        print(f"  ({s['similarity']:.3f}){dup_flag} #{s['complaint_id']} [{s['category']}] {s['text']}")
    print("=" * 70)


if __name__ == "__main__":
    pipeline = IssueGraphPipeline()

    test_complaints = [
        "Ceiling fan sounds like a helicopter and vibrates a lot",
        "There is sparking near the switchboard, looks very unsafe",
        "WiFi keeps disconnecting every few minutes in my room",
    ]

    for complaint in test_complaints:
        report = pipeline.process(complaint)
        print_report(report)
        print()

    print("Graph stats after processing:", pipeline.graph_stats())