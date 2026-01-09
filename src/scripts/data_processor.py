"""
Processamento de dados integrado com Feature Registry
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .cache_manager import get_stock_data

logger = logging.getLogger(__name__)


class StockDataProcessor:
    """Processador de dados com suporte a features modulares"""
    
    def __init__(self, symbol: str = "TSLA", start_date: str = "2020-01-01", 
                 end_date: str = "2024-12-31"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        # self.raw_data = None
        self.processed_data = None
        
        logger.info(f"StockDataProcessor inicializado para {symbol}")
    
    def load_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Carrega dados usando cache manager"""
        logger.info(f"Carregando dados para {self.symbol}")
        self.raw_data = get_stock_data(
            self.symbol, 
            self.start_date, 
            self.end_date, 
            force_refresh
        )
        return self.raw_data
            
    def prepare_for_lstm(self, target_column: str = 'Close',
                        sequence_length: int = 30,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.2) -> Tuple:
        """
        Prepara dados para treinamento LSTM
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names)
        """
        if self.processed_data is None:
            raise ValueError("Dados não processados. Chame process_with_features() primeiro.")
                
        # Filtrar colunas disponíveis
        available_features = self.processed_data.columns
        
        if target_column not in available_features:
            raise ValueError(f"Target column '{target_column}' não encontrada nos dados")
        
        logger.info(f"Preparando LSTM com {len(available_features)} features")
        logger.info(f"Features: {available_features}")
        
        data = self.processed_data[available_features].copy()
        data = data.dropna()
        
        # 1. Definir pontos de corte nos dados BRUTOS
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # 2. Fit do Scaler APENAS no Treino
        scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler = StandardScaler()
        
        # Treino: usa fit_transform
        train_data = data.iloc[:train_end]
        scaler.fit(train_data)
        
        # 3. Aplicar transformação em TUDO usando o scaler do treino
        # Isso garante que se o teste tiver valores maiores que o treino, 
        # eles ficarão > 1.0 (o que é correto e esperado em time series)
        scaled_data = scaler.transform(data)
        
        # 4. Criar sequências (agora seguro)
        # Encontrar o índice da coluna 'Close'
        close_idx = list(available_features).index(target_column)
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])  # Todas as features como input
            y.append(scaled_data[i+sequence_length, close_idx])  # Apenas Close como target
        
        X = np.array(X)
        y = np.array(y)
        
        # 5. Split das Sequências (mantendo a proporção temporal)
        # Recalcula índices baseados nas sequências geradas
        n_seq = len(X)
        train_size = int(n_seq * train_ratio)
        val_size = int(n_seq * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Log com informações detalhadas
        total_sequences = len(X)
        logger.info(f"\n✅ Dataset split:")
        logger.info(f"  Treino: {len(X_train)} sequências ({len(X_train)/total_sequences*100:.1f}%)")
        logger.info(f"  Validação: {len(X_val)} sequências ({len(X_val)/total_sequences*100:.1f}%)")
        logger.info(f"  Teste: {len(X_test)} sequências ({len(X_test)/total_sequences*100:.1f}%)")
        logger.info(f"  Shape X_train: {X_train.shape} (samples, timesteps, features)")
        logger.info(f"  Shape y_train: {y_train.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler, available_features
    
    def process_pipeline(self, force_refresh: bool = False, 
                        sequence_length: int = 30,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.2) -> Tuple:
        """
        Pipeline completo de processamento
        
        Returns:
            (processed_df, train_val_test_data)
        """
        # 1. Carregar dados
        raw_df = self.load_data(force_refresh)
        
        # 2. Armazenar dados processados
        self.processed_data = raw_df
        
        # 3. Preparar para LSTM com parâmetros dinâmicos
        lstm_data = self.prepare_for_lstm(
            sequence_length=sequence_length,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
        
        return raw_df, lstm_data