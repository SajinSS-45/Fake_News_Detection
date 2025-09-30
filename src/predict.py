import argparse
import joblib
import os

def predict(text, model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "FAKE" if pred == 1 else "REAL"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="News text to classify")
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--vectorizer", default="models/tfidf_vectorizer.joblib", help="Path to saved TF-IDF vectorizer")
    args = parser.parse_args()

    result = predict(args.text, args.model, args.vectorizer)
    print(f"\n[RESULT] The news is predicted as: {result}")
