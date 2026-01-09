import yfinance as yf
import pandas as pd

def download_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Coleta dados do Yahoo Finance[cite: 17, 24]."""
    print(f"Baixando dados para {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date)
    # Ajuste para garantir que pegamos apenas o 'Close'
    if 'Close' in df.columns:
        df = df[['Close']]
    return df