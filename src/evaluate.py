import argparse
import joblib
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os

def evaluate(model_path, vectorizer_path, fake_path, true_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df])
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    X = vectorizer.transform(df["text"])
    y = df["label"]

    preds = model.predict(X)
    print("[INFO] Accuracy:", accuracy_score(y, preds))
    print("\n[INFO] Classification Report:\n")
    print(classification_report(y, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--vectorizer", default="models/tfidf_vectorizer.joblib", help="Path to vectorizer")
    parser.add_argument("--fake", required=True, help="Path to Fake.csv")
    parser.add_argument("--true", required=True, help="Path to True.csv")
    args = parser.parse_args()

    evaluate(args.model, args.vectorizer, args.fake, args.true)
