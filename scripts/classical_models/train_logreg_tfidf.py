# TF-IDF + Logistic Regression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/processed/news_with_impact.csv")
df = df.dropna(subset=['headline', 'impact'])
df['impact'] = df['impact'].astype(int)

# Features and labels
X = df['headline']
y = df['impact']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Neu", "Pos"], yticklabels=["Neg", "Neu", "Pos"])
plt.title("Confusion Matrix - TF-IDF + Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix_logreg.png")
print("✅ Confusion matrix saved.")

# Save top tokens
idf_scores = vectorizer.idf_
tokens = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame({'token': tokens, 'tfidf_score': idf_scores})
tfidf_df = tfidf_df.sort_values(by='tfidf_score', ascending=False).head(20)
os.makedirs("data/processed", exist_ok=True)
tfidf_df.to_csv("data/processed/tfidf_top_tokens.csv", index=False)
print("✅ TF-IDF top tokens saved.")

