import os

# Force CPU to avoid CUDA/PTX issues on this GPU/TF build.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
from src.ml.data_loader import download_data
from src.ml.process import DataProcessor

# Configurações
SYMBOL = 'PETR4.SA' # Exemplo B3 ou 'DIS' [cite: 21]
START = '2018-01-01'
END = '2024-07-20'
TIME_STEP = 60

def train_model():
    mlflow.set_experiment("LSTM_Stock_Prediction")
    
    with mlflow.start_run():
        # 1. Coleta
        df = download_data(SYMBOL, START, END)
        dataset = df.values
        
        # 2. Processamento
        processor = DataProcessor()
        scaled_data = processor.fit_transform(dataset)
        os.makedirs("models", exist_ok=True)
        processor.save_scaler()
        mlflow.log_artifact("models/scaler.save")

        # Split Treino/Teste (Sequencial para séries temporais)
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:1]
        
        X_train, y_train = processor.create_dataset(train_data, TIME_STEP)
        X_test, y_test = processor.create_dataset(test_data, TIME_STEP)
        
        # Reshape para [samples, time steps, features] necessário para LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # 3. Modelagem LSTM [cite: 27]
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)))
        model.add(Dropout(0.2)) # Evitar Overfitting
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Treinamento
        history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
        
        # 4. Avaliação
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Inverter escala para calcular métricas reais
        y_train_inv = processor.inverse_transform([y_train])
        y_test_inv = processor.inverse_transform([y_test])
        train_predict_inv = processor.inverse_transform(train_predict)
        test_predict_inv = processor.inverse_transform(test_predict)
        
        # Métricas [cite: 29]
        rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict_inv[:,0]))
        mae = mean_absolute_error(y_test_inv[0], test_predict_inv[:,0])
        
        print(f"RMSE: {rmse}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # 5. Salvar Modelo [cite: 31]
        model.save("models/model.keras")
        mlflow.log_artifact("models/model.keras")

        # Plot de Validação
        plt.figure(figsize=(10,6))
        plt.plot(y_test_inv[0], label='Real')
        plt.plot(test_predict_inv[:,0], label='Previsto')
        plt.legend()
        plt.savefig("models/validation_plot.png")
        mlflow.log_artifact("models/validation_plot.png")

if __name__ == "__main__":
    train_model()