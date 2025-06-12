import pandas as pd

df = pd.read_excel("data/raw/StockPMETA.xlsx")
print("📊 Columnas en StockPMETA.xlsx:")
print(df.columns)
