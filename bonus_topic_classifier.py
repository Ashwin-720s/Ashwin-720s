"""
BONUS (optional) — A second, genuinely-trained-from-scratch model.

This isn't required for the Unsloth pipeline in GUIDE.md, but it's a nice
extra: a small classifier you train yourself (in seconds, on CPU, no
Colab needed) that predicts a question's subject/chapter. It's useful if
you want two "I trained this" talking points instead of one, or as a
backup demo if your fine-tuned model or Colab session gives you trouble
close to the deadline.

WHAT IT DOES:
Takes the text chunks from data/chunks.csv and trains a classifier that,
given a student's question, predicts which SUBJECT + CHAPTER it belongs to
(e.g. "Physics - Thermodynamics"). This is a genuine supervised ML model:
- Input (X): text, converted to numbers using TF-IDF
- Output (y): subject+chapter label
- Model: Logistic Regression (fast, interpretable, trains in seconds on CPU)

WHY THIS MODEL:
- TF-IDF + Logistic Regression trains on thousands of text chunks in under
  a minute on a plain i5 CPU. No GPU needed at all.
- It gives you real evaluation metrics (accuracy, confusion matrix) you can
  put straight on your poster — genuine "science" content for judges.

Run with:  python 2_train_classifier.py
Outputs:
  models/classifier.joblib       <- the trained model + vectorizer
  models/label_encoder.joblib    <- maps label numbers back to names
  data/confusion_matrix.png      <- plot for your poster
  data/training_report.txt       <- accuracy + per-class metrics
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "data/chunks.csv"
MODEL_DIR = "models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"Can't find {DATA_FILE}. Run 1_prepare_data.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    df["label"] = df["subject"] + " - " + df["chapter"]

    # Drop labels that have too few examples to learn from (need at least
    # a handful of chunks per chapter — if a chapter has <5 chunks, the
    # model can't generalize on it).
    counts = df["label"].value_counts()
    valid_labels = counts[counts >= 5].index
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    print(f"Training on {len(df)} chunks across {df['label'].nunique()} classes:")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        print("\nNeed at least 2 different chapters/subjects to train a classifier.")
        print("Add more PDFs (different chapters) and re-run 1_prepare_data.py.")
        return

    # --- Encode labels ---
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    X_text = df["text"]

    # --- Train/test split ---
    # stratify keeps the class balance similar in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Vectorize text with TF-IDF ---
    # This turns each chunk of text into a vector of word-importance scores.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams capture short phrases too
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # --- Train the classifier ---
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    )

    print(f"\nTest accuracy: {acc:.2%}")
    print(report)

    with open("data/training_report.txt", "w") as f:
        f.write(f"Test accuracy: {acc:.2%}\n\n")
        f.write(report)

    # --- Confusion matrix plot (great for your poster) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=encoder.classes_, yticklabels=encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Test Accuracy: {acc:.1%})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("data/confusion_matrix.png", dpi=150)
    print("Saved confusion matrix plot to data/confusion_matrix.png")

    # --- Save everything the app needs later ---
    joblib.dump({"model": model, "vectorizer": vectorizer}, os.path.join(MODEL_DIR, "classifier.joblib"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    print(f"Saved trained model to {MODEL_DIR}/classifier.joblib")


if __name__ == "__main__":
    main()
