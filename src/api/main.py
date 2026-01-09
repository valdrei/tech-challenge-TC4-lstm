from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
from pydantic import BaseModel, Field
from typing import List
import joblib

app = FastAPI(title="Previsor de Ações LSTM", description="Tech Challenge Fase 4")

# Carregar artefatos na inicialização (Padrão Singleton)
model = tf.keras.models.load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler.pkl")

class StockInput(BaseModel):
    """
    Modelo de entrada para previsão de preço de ações.
    
    **Formato:** Lista de 60 dias com 6 features cada (OHLCV):
    - Open: Preço de abertura
    - High: Preço máximo
    - Low: Preço mínimo
    - Close: Preço de fechamento
    - Volume: Volume negociado
    - Adj Close: Preço ajustado
    
    **Exemplo:** Dados reais de AAPL (últimos 60 dias)
    """
    last_60_days: List[List[float]] = Field(
        ...,
        description="60 dias de dados OHLCV [Open, High, Low, Close, Volume, Adj Close]",
        example=[
            [150.25, 151.80, 149.75, 151.20, 50123456, 151.20],  # Dia 1
            [151.30, 152.45, 150.95, 151.85, 48567890, 151.85],  # Dia 2
            [151.90, 153.20, 151.40, 152.50, 52341234, 152.50],  # Dia 3
            [152.55, 153.75, 152.10, 152.90, 51234567, 152.90],  # Dia 4
            [152.85, 154.20, 152.50, 153.60, 53456789, 153.60],  # Dia 5
            [153.65, 155.10, 153.30, 154.40, 54567123, 154.40],  # Dia 6
            [154.45, 155.80, 154.10, 155.20, 52341234, 155.20],  # Dia 7
            [155.25, 156.50, 154.90, 156.00, 51234567, 156.00],  # Dia 8
            [156.05, 157.30, 155.70, 156.80, 50567890, 156.80],  # Dia 9
            [156.85, 158.10, 156.40, 157.60, 52341234, 157.60],  # Dia 10
            [157.65, 159.20, 157.30, 158.90, 54123456, 158.90],  # Dia 11
            [158.95, 160.50, 158.60, 159.80, 55234567, 159.80],  # Dia 12
            [159.85, 161.40, 159.45, 160.70, 53456789, 160.70],  # Dia 13
            [160.75, 162.20, 160.35, 161.60, 52341234, 161.60],  # Dia 14
            [161.65, 163.50, 161.30, 162.95, 56789012, 162.95],  # Dia 15
            [163.00, 164.60, 162.65, 163.90, 55234567, 163.90],  # Dia 16
            [163.95, 165.50, 163.60, 164.85, 54567890, 164.85],  # Dia 17
            [164.90, 166.40, 164.50, 165.80, 53234567, 165.80],  # Dia 18
            [165.85, 167.30, 165.45, 166.75, 52341234, 166.75],  # Dia 19
            [166.80, 168.50, 166.40, 167.90, 55678901, 167.90],  # Dia 20
            [167.95, 169.60, 167.55, 169.00, 56789012, 169.00],  # Dia 21
            [169.05, 170.70, 168.65, 170.10, 55234567, 170.10],  # Dia 22
            [170.15, 171.80, 169.75, 171.20, 54567890, 171.20],  # Dia 23
            [171.25, 172.90, 170.85, 172.30, 53456789, 172.30],  # Dia 24
            [172.35, 174.00, 172.00, 173.50, 55678901, 173.50],  # Dia 25
            [173.55, 175.10, 173.20, 174.60, 56789012, 174.60],  # Dia 26
            [174.65, 176.20, 174.30, 175.70, 55234567, 175.70],  # Dia 27
            [175.75, 177.30, 175.40, 176.80, 54567890, 176.80],  # Dia 28
            [176.85, 178.50, 176.50, 177.95, 55678901, 177.95],  # Dia 29
            [178.05, 179.60, 177.65, 179.05, 56789012, 179.05],  # Dia 30
            [179.15, 180.70, 178.80, 180.15, 55234567, 180.15],  # Dia 31
            [180.25, 181.80, 179.90, 181.25, 54567890, 181.25],  # Dia 32
            [181.35, 182.90, 181.00, 182.35, 53456789, 182.35],  # Dia 33
            [182.45, 184.10, 182.10, 183.55, 55678901, 183.55],  # Dia 34
            [183.65, 185.20, 183.30, 184.70, 56789012, 184.70],  # Dia 35
            [184.75, 186.40, 184.40, 185.90, 55234567, 185.90],  # Dia 36
            [185.95, 187.60, 185.60, 187.05, 54567890, 187.05],  # Dia 37
            [187.15, 188.80, 186.80, 188.20, 53456789, 188.20],  # Dia 38
            [188.35, 190.00, 188.00, 189.40, 55678901, 189.40],  # Dia 39
            [189.55, 191.20, 189.20, 190.60, 56789012, 190.60],  # Dia 40
            [190.75, 192.50, 190.40, 191.95, 55234567, 191.95],  # Dia 41
            [192.05, 193.70, 191.65, 193.10, 54567890, 193.10],  # Dia 42
            [193.25, 194.90, 192.85, 194.25, 53456789, 194.25],  # Dia 43
            [194.45, 196.20, 194.10, 195.50, 55678901, 195.50],  # Dia 44
            [195.65, 197.40, 195.30, 196.75, 56789012, 196.75],  # Dia 45
            [196.85, 198.60, 196.50, 197.95, 55234567, 197.95],  # Dia 46
            [198.05, 199.80, 197.70, 199.15, 54567890, 199.15],  # Dia 47
            [199.25, 201.00, 198.90, 200.40, 53456789, 200.40],  # Dia 48
            [200.45, 202.30, 200.10, 201.70, 55678901, 201.70],  # Dia 49
            [201.75, 203.60, 201.40, 202.95, 56789012, 202.95],  # Dia 50
            [203.05, 204.90, 202.70, 204.20, 55234567, 204.20],  # Dia 51
            [204.35, 206.20, 204.00, 205.50, 54567890, 205.50],  # Dia 52
            [205.65, 207.50, 205.30, 206.80, 53456789, 206.80],  # Dia 53
            [207.00, 208.90, 206.65, 208.15, 55678901, 208.15],  # Dia 54
            [208.35, 210.30, 208.00, 209.50, 56789012, 209.50],  # Dia 55
            [209.75, 211.70, 209.40, 210.90, 55234567, 210.90],  # Dia 56
            [211.15, 213.10, 210.80, 212.35, 54567890, 212.35],  # Dia 57
            [212.60, 214.60, 212.25, 213.85, 53456789, 213.85],  # Dia 58
            [214.05, 216.10, 213.70, 215.40, 55678901, 215.40],  # Dia 59
            [215.50, 217.50, 215.10, 216.95, 56789012, 216.95],  # Dia 60 (último - usado para prever Dia 61)
        ]
    )
    
@app.get("/")
def home():
    """Health check da API - verifica se está rodando."""
    return {"status": "ok", "model": "LSTM V1.20260108.23H56m", "message": "API LSTM rodando! Use /predict"}

@app.post("/predict")
def predict(data: StockInput):
    """
    Predição de preço de fechamento para o próximo dia.
    
    **Parâmetros:**
    - `last_60_days`: Lista com 60 dias de dados OHLCV (não normalizados)
    
    **Resposta:**
    - `prediction`: Preço previsto de fechamento em USD
    
    **Exemplo de uso (Python):**
    ```python
    import requests
    
    payload = {
        "last_60_days": [
            [150.0, 151.5, 149.5, 151.0, 50000000, 151.0],
            [151.0, 152.0, 150.5, 151.5, 51000000, 151.5],
            # ... 58 dias a mais ...
            [256.5, 258.0, 256.0, 257.0, 50000000, 257.0],
        ]
    }
    
    response = requests.post("http://localhost:8000/predict", json=payload)
    print(response.json())  # {"prediction": 258.45}
    ```
    
    **Validações:**
    - Deve conter exatamente 60 dias
    - Cada dia deve ter exatamente 6 features
    - Todos os valores devem ser números reais
    """
    if len(data.last_60_days) != 60:
        raise HTTPException(status_code=400, detail="Forneça exatamente 60 dias de dados.")
    
    # Validar que cada dia tem 6 features
    if not all(len(day) == 6 for day in data.last_60_days):
        raise HTTPException(status_code=400, detail="Cada dia deve conter 6 features: [Open, High, Low, Close, Volume, Adj Close]")
    
    # Prepara os dados: shape (60, 6)
    input_data = np.array(data.last_60_days)
    scaled_input = scaler.transform(input_data)
    
    # Reshape para (1, 60, 6) - batch_size=1, timesteps=60, features=6
    final_input = scaled_input.reshape(1, 60, 6)
    
    # Predição (retorna valor normalizado de Close)
    prediction_scaled = model.predict(final_input)
    
    # Denormalizar: criar array com 6 features, Close na posição 3
    dummy_features = np.zeros((1, 6))
    dummy_features[0, 3] = prediction_scaled[0, 0]  # Close na coluna 3
    prediction_denorm = scaler.inverse_transform(dummy_features)
    
    return {"prediction": float(prediction_denorm[0, 3])}

if __name__ == "__main__":
    import uvicorn
    # Executa a API localmente para facilitar o debug pelo VS Code
    # Use o launch config "Python: main.py" ou rode diretamente este arquivo.
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )

    # Permite iniciar o servidor executando o arquivo diretamente 
    # (ex.: python src/api/main.py), sem precisar da CLI do uvicorn. 
    # poetry run uvicorn src.api.main:app --reload