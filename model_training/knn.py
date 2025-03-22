import pandas as pd
import joblib
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Assign Labels: 1 for Real News, 0 for Fake News
true_df["label"] = 1
fake_df["label"] = 0

# Merge both datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

df["text"] = df["text"].apply(clean_text)

# Prepare Data
X = df["text"]
y = df["label"]

# Convert Text to Numerical Features
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)  
X = vectorizer.fit_transform(X)

# Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train K-Nearest Neighbors (KNN) Model
knn_model = KNeighborsClassifier(n_neighbors=5, metric='cosine')  # Cosine similarity is better for text
knn_model.fit(X_train, y_train)

# Make Predictions
y_pred = knn_model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Metrics
print(f"KNN Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Visualizing the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - KNN")
plt.show()

# Save Model and Vectorizer
joblib.dump(knn_model, "knn_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("KNN Model and Vectorizer Saved Successfully!")

# Sample Predictions
sample_texts = [
    "NASA successfully landed a rover on Mars.",
    "Breaking: A celebrity was found involved in a secret scandal.",
    "Government announces new economic policies for 2025."
]

sample_vectors = vectorizer.transform(sample_texts)
sample_preds = knn_model.predict(sample_vectors)

for text, pred in zip(sample_texts, sample_preds):
    print(f"Text: {text}\nPredicted: {'Real' if pred == 1 else 'Fake'}\n")
