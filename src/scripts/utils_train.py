from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def build_model(config, input_shape):
    model = Sequential()
    
    # Primeira Camada (Obrigatória)
    # Se houver uma segunda camada configurada, return_sequences=True
    use_second_layer = len(config['lstm_units']) > 1
    
    model.add(LSTM(units=config['lstm_units'][0],
                   return_sequences=use_second_layer, # True se tiver prox camada
                   kernel_regularizer=l2(config['regularization_l2']),
                   recurrent_regularizer=l2(config['regularization_l2']),
                   input_shape=input_shape))
    model.add(Dropout(config['dropout_rate']))

    # Segunda Camada (Opcional)
    if use_second_layer:
        model.add(LSTM(units=config['lstm_units'][1],
                       return_sequences=False,
                       kernel_regularizer=l2(config['regularization_l2']),
                       recurrent_regularizer=l2(config['regularization_l2'])))
        model.add(Dropout(config['dropout_rate']))
    
    model.add(Dense(config['dense_units'], activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) # Use MAE ou MSE
    return model

def build_model_v2(config,input_shape):
    model = Sequential()
    
    # Camada LSTM com retorno de sequência (para empilhar outra LSTM se quiser)
    model.add(LSTM(units=config['lstm_units'][0],
                   return_sequences=True,
                   kernel_regularizer=l2(config['regularization_l2']),
                   recurrent_regularizer=l2(config['regularization_l2']),
                   recurrent_dropout=config.get('recurrent_dropout', 0.0),
                   input_shape=input_shape))
    model.add(Dropout(config['dropout_rate'])) # Evita Overfitting
    model.add(LSTM(units=config['lstm_units'][1],
                   return_sequences=False,
                   kernel_regularizer=l2(config['regularization_l2']),
                   recurrent_regularizer=l2(config['regularization_l2']),
                   recurrent_dropout=config.get('recurrent_dropout', 0.0)))
    model.add(Dropout(config['dropout_rate']))
    model.add(Dense(
        units=config['dense_units'],
        activation='relu',
        kernel_regularizer=l2(config['regularization_l2'])
    ))
    model.add(Dropout(config['dropout_rate'] * 0.5))
    
    # Output
    model.add(Dense(1, activation='linear'))
    
    optimizer = Adam(learning_rate=config['learning_rate'])
    
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mape'])
    return model

def inverse_transform_predictions(y_true_scaled, y_pred_scaled, scaler_obj, feature_names_list, target_col='Close'):
    """Inverte a normalização das previsões"""
    feature_names_list = list(feature_names_list)  # garante que suporte Index/array
    
    # Extrai vetor 1D do alvo, mesmo que venha 2D
    if hasattr(y_true_scaled, 'ndim') and y_true_scaled.ndim > 1:
        y_true_vec = np.asarray(y_true_scaled)[:, 0]
    else:
        y_true_vec = np.asarray(y_true_scaled).flatten()
    
    if hasattr(y_pred_scaled, 'ndim') and y_pred_scaled.ndim > 1:
        y_pred_vec = np.asarray(y_pred_scaled)[:, 0]
    else:
        y_pred_vec = np.asarray(y_pred_scaled).flatten()
    
    n_samples = len(y_true_vec)
    n_features = len(feature_names_list)
    target_idx = feature_names_list.index(target_col)
    
    dummy_true = np.zeros((n_samples, n_features))
    dummy_pred = np.zeros((n_samples, n_features))
    dummy_true[:, target_idx] = y_true_vec
    dummy_pred[:, target_idx] = y_pred_vec
    
    y_true_original = scaler_obj.inverse_transform(dummy_true)[:, target_idx]
    y_pred_original = scaler_obj.inverse_transform(dummy_pred)[:, target_idx]
    
    return y_true_original, y_pred_original

def calculate_metrics(y_true, y_pred, name=''):
    """Calcula métricas de avaliação"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        min_len = min(len(true_dir), len(pred_dir))
        dir_acc = np.mean(true_dir[:min_len] == pred_dir[:min_len]) * 100
    else:
        dir_acc = 0
    
    return {
        'Dataset': name,
        'MAE ($)': mae,
        'RMSE ($)': rmse,
        'MAPE (%)': mape,
        'R² Score': r2,
        'Dir. Accuracy (%)': dir_acc
    }
