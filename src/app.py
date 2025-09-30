from flask import Flask, render_template_string, request
import joblib

app = Flask(__name__)

MODEL_PATH = "models/logistic_regression.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
</head>
<body style="font-family: Arial; margin: 40px;">
    <h2>Fake News Detection</h2>
    <form method="POST">
        <textarea name="text" rows="10" cols="80" placeholder="Enter news text here..."></textarea><br><br>
        <input type="submit" value="Check News">
    </form>
    {% if result %}
        <h3>Prediction: {{ result }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        result = "FAKE" if pred == 1 else "REAL"
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)
