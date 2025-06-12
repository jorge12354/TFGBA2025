# FinBERT + MLP
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name).to(device).eval()

# Load data
df = pd.read_csv("data/processed/news_with_impact.csv")
df = df.dropna(subset=['headline', 'impact'])
df['impact'] = df['impact'].astype(int)
df['label'] = df['impact'].map({-1: 0, 0: 1, 1: 2})

# Generate embeddings
@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = bert_model(**inputs).last_hidden_state[:, 0, :]
    return output.squeeze().cpu().numpy()

print("🔄 Generating FinBERT embeddings...")
X = np.vstack([get_embedding(t) for t in df['headline']])
y = df['label'].values

# Save embeddings
df_emb = pd.DataFrame(X)
df_emb['label'] = y
os.makedirs("data/processed", exist_ok=True)
df_emb.to_csv("data/processed/finbert_embeddings.csv", index=False)
print("✅ FinBERT embeddings saved.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# Define MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    def forward(self, x):
        return self.model(x)

model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
print("🚀 Training MLP...")
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Evaluation
model.eval()
preds, labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb)
        pred = torch.argmax(out, dim=1).cpu().numpy()
        preds.extend(pred)
        labels.extend(yb.numpy())

print("\n📊 Classification Report:")
print(classification_report(labels, preds, digits=4))

cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Neu", "Pos"], yticklabels=["Neg", "Neu", "Pos"])
plt.title("Confusion Matrix - FinBERT + MLP")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix_finbert.png")
print("✅ Confusion matrix saved.")
