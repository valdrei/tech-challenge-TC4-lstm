"""
Script para retreinar modelo LSTM e garantir que scaler e modelo estejam sincronizados.
Salva em models/ para uso pela API.
"""
import sys
from pathlib import Path

# Setup de caminhos
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print(f"📂 Project root: {project_root}")
print(f"📂 Salvando em: {project_root / 'models'}")

import numpy as np
import joblib
from src.scripts.data_processor import StockDataProcessor
from src.scripts.utils_train import build_model

# Configurações de dados
SYMBOL = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

# Configuração do Modelo - V1 BASELINE
model_config = {
    'sequence_length': 60,
    'lstm_units': [100, 50],
    'dropout_rate': 0.25,
    'regularization_l2': 0.003,
    'dense_units': 16,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'train_ratio': 0.7,
    'val_ratio': 0.2
}

print("\n" + "="*80)
print("PROCESSANDO DADOS")
print("="*80)

# Processar dados
processor = StockDataProcessor(SYMBOL, START_DATE, END_DATE)
processed_df, lstm_data = processor.process_pipeline(
    sequence_length=model_config['sequence_length'],
    train_ratio=model_config['train_ratio'],
    val_ratio=model_config['val_ratio']
)

X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names = lstm_data

print(f"✅ X_train: {X_train.shape}")
print(f"✅ Features: {len(feature_names)}")
print(f"✅ Scaler features: {scaler.n_features_in_}")

print("\n" + "="*80)
print("TREINANDO MODELO")
print("="*80)

# Construir modelo
model = build_model(
    config=model_config,
    input_shape=(model_config['sequence_length'], X_train.shape[2])
)

# Treinar
history = model.fit(
    X_train, y_train,
    batch_size=model_config['batch_size'],
    epochs=model_config['epochs'],
    validation_data=(X_val, y_val),
    verbose=1
)

# Avaliar
y_test_pred = model.predict(X_test)
mae = float(np.mean(np.abs(y_test - y_test_pred)))
rmse = float(np.sqrt(np.mean((y_test - y_test_pred)**2)))
print(f"\n📊 MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

print("\n" + "="*80)
print("SALVANDO ARTEFATOS")
print("="*80)

# Criar diretório models/
models_dir = project_root / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

# Salvar modelo
model_path = models_dir / 'lstm_model.keras'
model.save(str(model_path))
print(f"✅ Modelo salvo: {model_path}")
print(f"   Tamanho: {model_path.stat().st_size / 1024:.1f} KB")

# Salvar scaler
scaler_path = models_dir / 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler salvo: {scaler_path}")
print(f"   Features: {scaler.n_features_in_}")
print(f"   Tamanho: {scaler_path.stat().st_size / 1024:.1f} KB")

# Verificar
scaler_test = joblib.load(scaler_path)
print(f"\n🧪 Verificação: Scaler carregado com {scaler_test.n_features_in_} features")

print("\n✅ Treino completo! Pronto para testar a API.")
