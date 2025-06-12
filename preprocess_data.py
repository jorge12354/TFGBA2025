import pandas as pd
import os

# --- Configuración inicial ---
DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
PRICE_FILES = {
    "META": "StockPMETA.xlsx",
    "GOOGL": "StockPGOOGL.xlsx",
    "AAPL": "StockPAAPL.xlsx",
    "AMZN": "StockPAMZN.xlsx",
    "NVDA": "StockPNVDA.xlsx",
}

# --- Cargar titulares ---
def load_headlines(file_path):
    df = pd.read_excel(file_path)
    df = df.rename(columns={
        'Headline': 'headline',
        'Date': 'date',
        'sentiment_label': 'sentiment_label',
        'sentiment_score': 'sentiment_score',
    })
    df['date'] = pd.to_datetime(df['date'])
    df['company'] = df['headline'].str.extract(r'\b(META|GOOGL|AAPL|AMZN|NVDA)\b')
    df = df.dropna(subset=['company'])
    return df

# --- Cargar precios y calcular cambio mensual ---
def load_price_data(ticker, filename):
    df = pd.read_excel(os.path.join(DATA_DIR, filename))
    df['date'] = pd.to_datetime(df['DATE'])
    df = df[['date', 'LAST PRICE']].sort_values('date')
    df['next_month_return'] = df['LAST PRICE'].pct_change().shift(-1) * 100
    df = df[['date', 'next_month_return']]
    df['company'] = ticker
    return df

# --- Etiquetar impacto: subida (>1%), neutro (-1% a 1%), bajada (<-1%) ---
def label_impact(change):
    if pd.isna(change):
        return None
    elif change > 1:
        return 1
    elif change < -1:
        return -1
    else:
        return 0

# --- Procesamiento completo ---
def process_data(news_file):
    headlines = load_headlines(os.path.join(DATA_DIR, news_file))
    price_dfs = [load_price_data(ticker, fname) for ticker, fname in PRICE_FILES.items()]
    prices = pd.concat(price_dfs)

    df = pd.merge(headlines, prices, on=['company', 'date'], how='left')
    df['impact'] = df['next_month_return'].apply(label_impact)

    output_file = os.path.join(OUTPUT_DIR, "news_with_impact.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Datos procesados guardados en: {output_file}")

if __name__ == "__main__":
    process_data("newskeydefprueba_con_sentimiento.xlsx")
