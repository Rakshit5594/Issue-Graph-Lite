"""
PHASE 5: Escalation Priority Classifier
-------------------------------------------
Same pattern as Phase 2, different target column: predicts
priority (Low / Medium / High) instead of category.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

with open("X_tfidf.pkl", "rb") as f:
    X = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
df = pd.read_pickle("df_processed.pkl")

y = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred))

with open("priority_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
    print("priority_classifier created successfully")


def predict_priority(text):
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]


if __name__ == "__main__":
    test_complaints = [
        "There is sparking near the switchboard, looks very unsafe",
        "The chair in my room has a small wobble",
    ]
    print("\n--- Sample predictions on new text ---")
    for c in test_complaints:
        print(f"{c!r} -> {predict_priority(c)}")