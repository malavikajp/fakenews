import pandas as pd
import joblib
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load Dataset
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Assign Labels: 1 for Real News, 0 for Fake News
true_df["label"] = 1
fake_df["label"] = 0

# Merge both datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Preprocessing Function (With Stopword Removal)
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

df["text"] = df["text"].apply(clean_text)

# Prepare Data
X = df["text"]
y = df["label"]

# Convert Text to Numerical Features
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")
X = vectorizer.fit_transform(X)

# Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Dictionary to store model performances
model_performance = {}

# Train Naïve Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
model_performance["Naïve Bayes"] = accuracy_score(y_test, y_pred_nb)

# Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
model_performance["Logistic Regression"] = accuracy_score(y_test, y_pred_lr)

# Train Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
model_performance["Decision Tree"] = accuracy_score(y_test, y_pred_dt)

# BERT Model (Deep Learning Approach)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize Text for BERT
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512, return_tensors="pt")

train_labels = torch.tensor(y_train.values)
test_labels = torch.tensor(y_test.values)

# Train BERT
optimizer = torch.optim.AdamW(bert_model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

bert_model.train()
for epoch in range(2):  # Small number of epochs for demonstration
    optimizer.zero_grad()
    outputs = bert_model(**train_encodings)
    loss = loss_fn(outputs.logits, train_labels)
    loss.backward()
    optimizer.step()

# Evaluate BERT
bert_model.eval()
with torch.no_grad():
    outputs = bert_model(**test_encodings)
    preds = torch.argmax(outputs.logits, dim=1)

accuracy_bert = accuracy_score(test_labels.numpy(), preds.numpy())
model_performance["BERT"] = accuracy_bert

# Print Model Performance
print("\n### Model Accuracies ###")
for model, acc in model_performance.items():
    print(f"{model}: {acc:.4f}")

# Save Best Performing Model
best_model = max(model_performance, key=model_performance.get)
if best_model == "Naïve Bayes":
    joblib.dump(nb_model, "model.pkl")
elif best_model == "Logistic Regression":
    joblib.dump(lr_model, "model.pkl")
elif best_model == "Decision Tree":
    joblib.dump(dt_model, "model.pkl")
else:
    torch.save(bert_model.state_dict(), "bert_model.pth")

joblib.dump(vectorizer, "vectorizer.pkl")

print("\nBest model saved:", best_model)
