import pandas as pd
import joblib
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
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

# Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF Vectorization with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the model with parameters to prevent overfitting
model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Cross-validation to check for overfitting
cv_score = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
cv_mean = cv_score.mean()

# Print results
print(f"Model: Decision Tree")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Cross-Validation Accuracy: {cv_mean:.4f}")
print(f"Difference (Overfit Check): {abs(accuracy - cv_mean):.4f}")
print("Classification Report:\n", report)

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Visualize tree features (optional)
feature_importances = pd.DataFrame({
    'feature': vectorizer.get_feature_names_out()[:20],  # Top 20 features
    'importance': model.feature_importances_[:20]
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()

# Save the model and vectorizer
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(vectorizer, "vectorizer_decision_tree.pkl")

print("Model and Vectorizer Saved Successfully!")

# Sample predictions
print("\nSample Predictions:")
sample_texts = [
    "NASA successfully landed a rover on Mars.",
    "Breaking: A celebrity was found involved in a secret scandal.",
    "Government announces new economic policies for 2025."
]

sample_vectors = vectorizer.transform(sample_texts)
sample_preds = model.predict(sample_vectors)

for text, pred in zip(sample_texts, sample_preds):
    print(f"Text: {text}\nPredicted: {'Real' if pred == 1 else 'Fake'}\n")