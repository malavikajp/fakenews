from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import string
import requests
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory

# Set seed for consistent language detection results
DetectorFactory.seed = 0  

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load Model and Vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("Error: Model or Vectorizer file not found!")
    model, vectorizer = None, None  # Ensure app doesn't crash

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text  

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model or vectorizer not found on server!"}), 500
    
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Clean and transform text
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
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch the news article"}), 500

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract meaningful text from <p> tags
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs if p.get_text()])

        # Ensure extracted text is valid
        if not content or len(content) < 100:
            return jsonify({"error": "Could not extract meaningful article text"}), 500

        # ✅ Language Detection (Only allow English)
        detected_lang = detect(content)
        if detected_lang != "en":
            return jsonify({"error": "Only English news articles are supported."}), 400

        return jsonify({"content": content})

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "No internet connection or website is unreachable."}), 500

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
