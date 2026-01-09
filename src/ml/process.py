import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def fit_transform(self, data):
        return self.scaler.fit_transform(data)
    
    def transform(self, data):
        return self.scaler.transform(data)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def create_dataset(self, dataset, time_step=60):
        """Cria estrutura para LSTM (Janela de tempo)."""
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)
    
    def save_scaler(self, path='models/scaler.save'):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path='models/scaler.save'):
        self.scaler = joblib.load(path)