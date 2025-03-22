import pandas as pd
import joblib
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Load Dataset
try:
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")
    print("✅ Datasets loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset files not found!")
    exit()

# ✅ Check if 'text' column exists
if "text" not in true_df.columns or "text" not in fake_df.columns:
    print("❌ Error: 'text' column not found in dataset!")
    exit()

# ✅ Assign Labels: 1 for Real News, 0 for Fake News
true_df["label"] = 1
fake_df["label"] = 0

# ✅ Merge both datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
print(f"Dataset size before preprocessing: {df.shape}")

# ✅ Check for missing values before processing
print("Checking for missing values in dataset:")
print(df.isnull().sum())

# ✅ Drop NaN and empty rows before preprocessing
df = df.dropna(subset=["text"])  # Remove NaN rows
df = df[df["text"].str.strip() != ""]  # Remove empty rows
print(f"Dataset size after dropping NaN/empty values: {df.shape}")

# ✅ Debugging: Print before cleaning
print("Before Cleaning (First 5 Texts):")
print(df["text"].head())

# ✅ Preprocessing Function
def clean_text(text):
    text = str(text).lower()  # Convert to string (handles NaN issues)
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    return text

df["text"] = df["text"].apply(clean_text)

# ✅ Debugging: Print after cleaning
print("After Cleaning (First 5 Texts):")
print(df["text"].head())

# ✅ Remove empty rows again after cleaning
df = df[df["text"].str.strip() != ""]

# ✅ Final check
if df.empty:
    print("❌ Error: Dataset is empty after preprocessing!")
    exit()

# ✅ Prepare Data
X = df["text"]
y = df["label"]

# ✅ Convert Text to Numerical Features
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vectorizer.fit_transform(X)
print("✅ Text vectorization completed!")

# ✅ Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ✅ Train Logistic Regression Model
model = LogisticRegression(solver='liblinear', max_iter=500)
model.fit(X_train, y_train)
print("✅ Model training completed!")

# ✅ Make Predictions
y_pred = model.predict(X_test)

# ✅ Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# ✅ Print Metrics
print(f"📊 Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n", report)
print("📊 Confusion Matrix:\n", conf_matrix)

# ✅ Visualizing the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ✅ Save Model and Vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and Vectorizer Saved Successfully!")

# ✅ Sample Predictions
print("🔍 Sample Predictions on Test Data:")
sample_texts = [
    "NASA successfully landed a rover on Mars.",
    "Breaking: A celebrity was found involved in a secret scandal.",
    "Government announces new economic policies for 2025."
]

sample_vectors = vectorizer.transform(sample_texts)
sample_preds = model.predict(sample_vectors)

for text, pred in zip(sample_texts, sample_preds):
    print(f"📰 Text: {text}\n📢 Predicted: {'Real' if pred == 1 else 'Fake'}\n")
