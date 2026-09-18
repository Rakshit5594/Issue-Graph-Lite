"""
PHASE 2: Standard Issue Category Classifier
---------------------------------------------
Supervised multi-class text classification.
Input:  TF-IDF vectors from Phase 1
Target: the 'category' column (Electrical, Plumbing, HVAC/Fan, etc.)
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- Load Phase 1 outputs ---
with open("X_tfidf.pkl", "rb") as f:
    X = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
df = pd.read_pickle("df_processed.pkl")

y = df["priority"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train classifier ---
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred))

# --- Save the trained model ---
with open("category_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)


def predict_category(text):
    """Predict category for a brand-new complaint."""
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]


if __name__ == "__main__":
    # quick sanity check with new, unseen phrasing
    test_complaints = [
        "The ceiling fan in my room sounds like a helicopter",
        "There is no water coming out of the tap",
        "WiFi keeps disconnecting every few minutes",
    ]
    print("\n--- Sample predictions on new text ---")
    for c in test_complaints:
        print(f"{c!r} -> {predict_category(c)}")