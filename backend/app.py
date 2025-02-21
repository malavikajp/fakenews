from flask import Flask, request, jsonify, render_template
import joblib
import string
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # ✅ Allow all origins for "/predict"

# Load Model and Vectorizer
try:
    model = joblib.load("backend/model.pkl")
    vectorizer = joblib.load("backend/vectorizer.pkl")
except FileNotFoundError:
    print("Error: Model or Vectorizer file not found! Make sure 'model.pkl' and 'vectorizer.pkl' exist.")

# Preprocessing Function (WITHOUT STOPWORDS)
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text  # ✅ Removed stopwords filtering

@app.route("/")
def home():
    return render_template("index.html")

THRESHOLD = 0.55  # Adjust threshold

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    text = clean_text(text)
    vectorized_text = vectorizer.transform([text])
    
    probabilities = model.predict_proba(vectorized_text)[0]  # Get probabilities
    
    if probabilities[1] > THRESHOLD:  # If real news probability > 0.55
        prediction = "Real"
    else:
        prediction = "Fake"

    return jsonify({
        "prediction": prediction,
        "probabilities": {
            "fake": round(probabilities[0], 3),
            "real": round(probabilities[1], 3)
        }
    })
'''
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = clean_text(data["text"])
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]

    return jsonify({"prediction": "Real" if prediction == 1 else "Fake"})
'''
if __name__ == "__main__":
    app.run(debug=True)
