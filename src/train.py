import argparse
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os
import json

def load_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1   # fake = 1
    true_df["label"] = 0   # true = 0

    df = pd.concat([fake_df, true_df])
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df[["text", "label"]]

def main(args):
    print("[INFO] Loading data...")
    df = load_data(args.fake, args.true)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=args.max_features)

    # Transform text
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression
    print("[INFO] Training Logistic Regression...")
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_tfidf, y_train)
    log_preds = log_model.predict(X_test_tfidf)

    # Random Forest
    print("[INFO] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=args.rf_estimators, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)
    rf_preds = rf_model.predict(X_test_tfidf)

    # Naive Bayes
    print("[INFO] Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_preds = nb_model.predict(X_test_tfidf)

    os.makedirs(args.out, exist_ok=True)

    # Save vectorizer and models
    joblib.dump(vectorizer, os.path.join(args.out, "tfidf_vectorizer.joblib"))
    joblib.dump(log_model, os.path.join(args.out, "logistic_regression.joblib"))
    joblib.dump(rf_model, os.path.join(args.out, "random_forest.joblib"))
    joblib.dump(nb_model, os.path.join(args.out, "naive_bayes.joblib"))

    # Save metrics
    metrics = {
        "logistic_regression": accuracy_score(y_test, log_preds),
        "random_forest": accuracy_score(y_test, rf_preds),
        "naive_bayes": accuracy_score(y_test, nb_preds),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("[INFO] Training complete. Metrics saved in metrics.json")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", required=True, help="Path to Fake.csv")
    parser.add_argument("--true", required=True, help="Path to True.csv")
    parser.add_argument("--out", default="models", help="Output folder")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF max features")
    parser.add_argument("--rf-estimators", type=int, default=100, help="Number of trees in Random Forest")
    args = parser.parse_args()
    main(args)
