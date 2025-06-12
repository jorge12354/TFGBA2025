# LSTM + Embedding
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
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
X = df['headline'].astype(str)
y = df['impact'].map({-1: 0, 0: 1, 1: 2})

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=30)
y_cat = tf.keras.utils.to_categorical(y, num_classes=3)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.2, stratify=y, random_state=42)

# Build model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=30),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Neu", "Pos"], yticklabels=["Neg", "Neu", "Pos"])
plt.title("Confusion Matrix - LSTM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix_lstm.png")
print("✅ Confusion matrix saved.")
