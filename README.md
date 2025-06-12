# TFGBA2025
Code used to implement the models of my Final Degree Project

# Tech Stock Sentiment Impact Prediction

This project is part of my Bachelor's Final Project in Data Analytics. It explores how the sentiment of news headlines influences the stock prices of five major US tech companies: **Amazon, Apple, Google, Meta, and Nvidia**.

Using NLP, sentiment analysis, and classification models, the project predicts whether a headline has a **positive, negative, or neutral** impact on the corresponding stock price.

---

## Folder Structure

TFGBA2025/
│
├── data/raw
│   ├── StockPAAPL.xlsx
│   ├── StockPAMZN.xlsx
│   ├── StockPGOOGL.xlsx
│   ├── StockPMETA.xlsx
│   ├── StockPNVDA.xlsx
│   ├── newskeydefprueba_con_sentimiento.xlsx
│   
├── data/processed
│   ├── finbert_embeddings.csv
│   ├── news_with_impact.csv
│   ├── tfidf_top_tokens.csv
│   ├── preprocess_data.py

├── notebooks/
│   ├── final_model_comparison.ipynb

├── results/
│   ├── confusion_matrix_finbert.png
│   ├── confusion_matrix_logreg.png
│   ├── confusion_matrix_lstm.png

├── scripts/classical_models
│   ├── train_logreg_tfidf.py

├── scripts/deep_learning
│   ├── train_lstm.py
│   └── train_finbert.py

├── scripts/preprocessing
│   ├── preprocess_data.py

└── README.md

## Models Trained

- **TF-IDF + Logistic Regression**
- **LSTM Neural Network**
- **FinBERT (pre-trained transformer for finance)**

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

## HOW TO RUN:

Train models with:

python models/train_logreg_tfidf.py
python models/train_lstm.py
python models/train_finbert.py

## Data Sources
All headline and price data collected manually from Bloomberg and Yahoo Finance (2023).
