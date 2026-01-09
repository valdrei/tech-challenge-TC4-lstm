import requests

payload = {
    "last_60_days": [
        0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
        0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
        0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
        0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
        0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
        0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69
    ]
}

# import yfinance as yf

# # Obtenha os últimos 60 preços de fechamento
# df = yf.download('PETR4.SA', period='6mo')['Close'].tail(60).tolist()  # Ajuste period para capturar dados até hoje

# payload = {"last_60_days": df}

resp = requests.post("http://localhost:8000/predict", json=payload)
print(resp.json())