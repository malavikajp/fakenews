from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import string
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

# Load Model and Vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("Error: Model or Vectorizer file not found!")

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text  

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    text = clean_text(text)
    vectorized_text = vectorizer.transform([text])
    
    probabilities = model.predict_proba(vectorized_text)[0]
    prediction = "Real" if probabilities[1] > 0.55 else "Fake"

    return jsonify({
        "prediction": prediction,
        "probabilities": {
            "fake": round(probabilities[0], 3),
            "real": round(probabilities[1], 3)
        }
    })

@app.route("/fetch-news", methods=["POST"])
def fetch_news():
    data = request.get_json()
    url = data.get("url")
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch the news article"}), 500

        soup = BeautifulSoup(response.text, "html.parser")

        # Try extracting text content (modify based on the site structure)
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])

        if not content:
            return jsonify({"error": "Could not extract article text"}), 500

        return jsonify({"content": content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
