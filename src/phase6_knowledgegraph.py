"""
PHASE 6: The Knowledge Graph
-------------------------------
Ties everything together into a graph:
  Complaint --belongs_to--> Category
  Complaint --belongs_to--> Cluster (root cause)
  Complaint --similar_to--> Complaint  (weighted by cosine similarity)

This is the "IssueGraph" visual centerpiece for your demo.
For a big dataset, only a SAMPLE is visualized (a graph of 100k nodes
is unreadable) -- but the same code works on the full similarity matrix
if you want to power a backend search feature instead of a picture.
"""

import pickle
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle("df_clustered.pkl")
with open("X_tfidf.pkl", "rb") as f:
    X_tfidf = pickle.load(f)

SIMILARITY_THRESHOLD = 0.4   # only draw an edge if complaints are fairly similar
SAMPLE_SIZE = 60             # keep the visual readable

random.seed(7)
sample_idx = random.sample(range(len(df)), SAMPLE_SIZE)
sample_df = df.iloc[sample_idx].reset_index(drop=True)
X_sample = X_tfidf[sample_idx]

sims = cosine_similarity(X_sample)

G = nx.Graph()

# add complaint, category, and cluster nodes
for i, row in sample_df.iterrows():
    complaint_node = f"C{row['complaint_id']}"
    category_node = f"CAT::{row['category']}"
    cluster_node = f"CLU::{row['cluster_id']}"

    G.add_node(complaint_node, kind="complaint", label=row["complaint_text"][:30])
    G.add_node(category_node, kind="category", label=row["category"])
    G.add_node(cluster_node, kind="cluster", label=row["issue_subtype"])

    G.add_edge(complaint_node, category_node, kind="belongs_to")
    G.add_edge(complaint_node, cluster_node, kind="belongs_to")

# add similarity edges between complaints above the threshold
for i in range(len(sample_df)):
    for j in range(i + 1, len(sample_df)):
        if sims[i, j] >= SIMILARITY_THRESHOLD:
            c1 = f"C{sample_df.iloc[i]['complaint_id']}"
            c2 = f"C{sample_df.iloc[j]['complaint_id']}"
            G.add_edge(c1, c2, kind="similar_to", weight=round(float(sims[i, j]), 2))

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- Visualize ---
color_map = {"complaint": "#6FA8DC", "category": "#F6B26B", "cluster": "#93C47D"}
node_colors = [color_map[G.nodes[n]["kind"]] for n in G.nodes]
node_sizes = [80 if G.nodes[n]["kind"] == "complaint" else 400 for n in G.nodes]

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.4, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)

belongs_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "belongs_to"]
similar_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "similar_to"]

nx.draw_networkx_edges(G, pos, edgelist=belongs_edges, edge_color="#cccccc", alpha=0.5)
nx.draw_networkx_edges(G, pos, edgelist=similar_edges, edge_color="#cc0000", width=2, alpha=0.7)

# label only category/cluster nodes to keep it readable
labels = {n: G.nodes[n]["label"] for n in G.nodes if G.nodes[n]["kind"] != "complaint"}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

plt.title("IssueGraph: Complaints -> Category / Root Cause + Similarity Links", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("issuegraph_visualization.png", dpi=150)
print("Saved issuegraph_visualization.png")