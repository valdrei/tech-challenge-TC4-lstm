import os

# Force CPU to avoid GPU driver/ptx issues in serving environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI(title="Previsor de Ações LSTM", description="Tech Challenge Fase 4")

# Carregar artefatos na inicialização (Padrão Singleton)
model = tf.keras.models.load_model("modelos/lstm_mvp.keras")
scaler = joblib.load("modelos/scaler.pkl")

class StockInput(BaseModel):
    # Espera uma lista de 60 valores float (preços dos últimos 60 dias)
    last_60_days: List[float]
    
@app.get("/")
def home():
    return {"status": "ok", "model": "LSTM V1", "message": "API LSTM rodando! Use /predict"}

@app.post("/predict")
def predict(data: StockInput):
    if len(data.last_60_days) != 60:
        raise HTTPException(status_code=400, detail="Forneça exatamente 60 dias de dados.")
    
    # Prepara os dados
    input_data = np.array(data.last_60_days).reshape(-1, 1)
    scaled_input = scaler.transform(input_data)
    
    # Reshape para (1, 60, 1)
    final_input = scaled_input.reshape(1, 60, 1)
    
    # Predição
    prediction_scaled = model.predict(final_input)
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return {"prediction": float(prediction[0][0])}

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