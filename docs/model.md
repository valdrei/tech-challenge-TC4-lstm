# 🔬 Especificação Técnica Detalhada - Modelo LSTM para Previsão de Ações

> 📚 **Navegação:** [← Voltar para README](../README.md) | [📖 Ver Índice de Docs](README.md) | [🔌 Ver API REST](api.md)

**Símbolo Padrão:** AAPL (Apple Inc.)  
**Período de Treinamento:** 2018-01-01 a 2025-12-31  
**Versão do Modelo:** V1.20260108 (Baseline)  
**Framework:** TensorFlow/Keras

---

## 📋 Sumário Executivo

Modelo de rede neural LSTM (Long Short-Term Memory) treinado para prever preços de fechamento de ações com base em dados históricos OHLCV (Open, High, Low, Close, Volume, Adjusted Close). A arquitetura utiliza duas camadas LSTM com dropout e regularização L2 para evitar overfitting.

**Performance Final (AAPL 2018-2025):**
- **R² (Teste):** ~85-93% (variação explicada)
- **MAE (Teste):** ~$9-12 (erro médio absoluto)
- **RMSE (Teste):** ~$12-15
- **MAPE (Teste):** ~3-4%

Este documento explica o pipeline de dados, arquitetura do modelo e treinamento em detalhes técnicos.

**📖 Documentos Relacionados:**
- [API REST - Guia Completo](api.md) - Como usar o modelo em produção
- [README Principal](../README.md) - Visão geral do projeto
- [Índice de Documentação](README.md) - Todos os documentos

---

## 📐 Arquitetura Detalhada

### Fluxo de Dados Completo

```
ENTRADA (Batch Size B)
    ↓
    └─ Shape: (B, 60, 6)
       • B = número de sequências no batch (padrão: 32)
       • 60 = timesteps (dias históricos)
       • 6 = features (Open, High, Low, Close, Volume, Adj Close)
       • Valores: Normalizados [0, 1]

    ↓ LSTM Layer 1 (100 unidades)
    ├─ Input: (B, 60, 6)
    ├─ Processamento:
    │  ├─ 100 células LSTM independentes
    │  ├─ Cada célula processa todos os 60 timesteps
    │  ├─ Cada célula mantém estado interno (memory cell)
    │  ├─ Return sequences = True → saída inclui todos os timesteps
    ├─ Output: (B, 60, 100)
    ├─ L2 Regularization (0.003):
    │  └─ Penalty = 0.003 * sum(weights²)
    └─ Parâmetros:
       └─ ~47K (100 * (6 + 100) * 4 gates)

    ↓ Dropout Layer 1 (25%)
    ├─ Durante treinamento: Remove 25% das ativações aleatoriamente
    ├─ Durante inferência: Usa todas (escaladas automaticamente)
    └─ Efeito: Regularização, evita co-adaptação

    ↓ LSTM Layer 2 (50 unidades)
    ├─ Input: (B, 60, 100)
    ├─ Processamento:
    │  ├─ 50 células LSTM independentes
    │  ├─ Processa 100 features de entrada
    │  └─ Return sequences = False → saída é last timestep
    ├─ Output: (B, 50)
    ├─ L2 Regularization (0.003)
    └─ Parâmetros:
       └─ ~30K (50 * (100 + 50) * 4 gates)

    ↓ Dropout Layer 2 (25%)
    ├─ Remove 25% das 50 ativações
    └─ Output: (B, 50)

    ↓ Dense Layer (16 unidades, ReLU)
    ├─ Transformação não-linear:
    │  └─ output[i] = max(0, sum(input[j] * weight[i,j]) + bias[i])
    ├─ ReLU: max(0, x) → introduz não-linearidade
    ├─ Output: (B, 16)
    ├─ L2 Regularization (0.003)
    └─ Parâmetros:
       └─ ~816 (16 * (50 + 1))

    ↓ Output Layer (1 unidade, Linear)
    ├─ Transformação linear:
    │  └─ output = sum(input[i] * weight[i]) + bias
    ├─ Linear: Sem função de ativação (regressão contínua)
    ├─ Output: (B, 1)
    ├─ Valor: [0, 1] normalizado
    └─ Parâmetros:
       └─ ~17 (1 * (16 + 1))

SAÍDA (Preço Previsto)
    ├─ Shape: (B, 1)
    ├─ Valores: [0, 1] normalizado
    └─ Pós-processamento: Inverter com scaler
       └─ Preço real = scaler.inverse_transform()
       
Total de Parâmetros: ~78,000
```

---

## 🧠 Célula LSTM: Como Funciona

### Mecanismo Interno

```
LSTM Cell em Timestep t:
┌─────────────────────────────────────────────────┐
│                                                 │
│  Inputs:                                        │
│  • x_t: vetor de entrada (6 features)           │
│  • h_{t-1}: hidden state anterior (100)         │
│  • c_{t-1}: cell state anterior (100)           │
│                                                 │
│  Operações (4 Gates):                           │
│                                                 │
│  1. Forget Gate: f_t = σ(W_f[h_{t-1}, x_t] + b_f)
│     → Controla quanto do passado é esquecido    │
│     → σ (sigmoid) dá valores 0-1               │
│     → 0 = esquecer tudo, 1 = lembrar tudo      │
│                                                 │
│  2. Input Gate: i_t = σ(W_i[h_{t-1}, x_t] + b_i)
│     → Controla quanto da entrada entra          │
│                                                 │
│  3. Candidate: C̃_t = tanh(W_c[h_{t-1}, x_t] + b_c)
│     → Nova informação candidata                 │
│     → tanh dá valores -1 a 1                   │
│                                                 │
│  4. Cell State Update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t
│     → Memória de longo prazo                    │
│     → ⊙ = multiplicação elemento-wise          │
│     → Esquece parte antiga + adiciona nova info │
│                                                 │
│  5. Output Gate: o_t = σ(W_o[h_{t-1}, x_t] + b_o)
│     → Controla saída                           │
│                                                 │
│  6. Hidden State: h_t = o_t ⊙ tanh(c_t)
│     → Saída para próximo timestep              │
│     → É o "output" da célula                   │
│                                                 │
└─────────────────────────────────────────────────┘

Onde:
  σ = sigmoid (0-1, controla fluxo de informação)
  tanh = (-1 a 1, pode "esquecer" valores antigos)
  ⊙ = multiplicação elemento-wise (Hadamard product)
  W, b = pesos e vieses (aprendidos durante treinamento)
```

### Por Que LSTM e Não RNN Simples?

| Aspecto | RNN Simples | LSTM |
|--------|-----------|------|
| **Janela efetiva** | ~5-10 timesteps | ~60+ timesteps |
| **Vanishing gradient** | Severo (∂h/∂h < 1)^60 ≈ 0 | Mitigado (cell state) |
| **Memória longa** | Difícil (esquece rápido) | Fácil (memory cell) |
| **Dependências** | Apenas curtas | Longas e curtas |
| **Backprop** | Gradientes "morrem" | Gradientes fluem |

**Exemplo Prático:**
- RNN: "Se preço subiu ontem, sobe hoje" (dependência curta)
- LSTM: "Se há 50 dias houve earnings positivos, há tendência de alta" (dependência longa)

---

## 📊 Pipeline de Dados

### 1. Coleta de Dados
**Classe:** `StockDataProcessor` ([src/scripts/data_processor.py](../src/scripts/data_processor.py))

```python
processor = StockDataProcessor('AAPL', '2018-01-01', '2025-12-31')
processed_df, lstm_data = processor.process_pipeline(
    sequence_length=60,
    train_ratio=0.7,
    val_ratio=0.2
)
```

**Features utilizadas (6 colunas):**

| # | Feature | Descrição | Tipo |
|---|---------|-----------|------|
| 1 | `Open` | Preço de abertura do dia | Float (USD) |
| 2 | `High` | Preço máximo do dia | Float (USD) |
| 3 | `Low` | Preço mínimo do dia | Float (USD) |
| 4 | `Close` | Preço de fechamento **(TARGET)** | Float (USD) |
| 5 | `Volume` | Volume negociado | Int64 |
| 6 | `Adj Close` | Preço ajustado por splits/dividendos | Float (USD) |

**Por que essas 6 features?**
- ✅ **OHLCV** é o padrão da indústria financeira
- ✅ Captura toda a informação básica de preço
- ✅ `Adj Close` corrige distorções históricas
- ✅ `Volume` indica força/interesse no movimento
- ✅ Dados sempre disponíveis (yfinance, APIs)

### 2. Normalização com MinMaxScaler

#### Transformação Forward (Treinamento)

Para cada feature f:
```
X_normalized[f] = (X_raw[f] - X_min[f]) / (X_max[f] - X_min[f])
```

**Exemplo com Close (AAPL):**
```python
# Dados brutos históricos (2018-2025)
X_raw['Close'] = [150, 180, 200, 250, 300, ...]
X_min['Close'] = 150  # Mínimo histórico
X_max['Close'] = 300  # Máximo histórico

# Normalização
X_normalized['Close'] = [
    (150-150)/(300-150) = 0.000,  # Mínimo vira 0
    (180-150)/(300-150) = 0.200,
    (200-150)/(300-150) = 0.333,
    (250-150)/(300-150) = 0.667,
    (300-150)/(300-150) = 1.000   # Máximo vira 1
]
```

#### Transformação Inverse (Inferência)

```
X_raw[f] = X_normalized[f] * (X_max[f] - X_min[f]) + X_min[f]

Exemplo:
y_pred_norm = 0.75  # Previsão normalizada do modelo
y_pred_real = 0.75 * (300 - 150) + 150 
            = 0.75 * 150 + 150
            = 262.5  # Preço real em USD
```

**Arquivo Scaler Salvo:**
```python
# models/scaler.pkl
MinMaxScaler(
  feature_range=(0, 1),
  n_features_in_=6,
  feature_names_in_=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'],
  data_min_=[150.0, 148.0, 145.0, 150.0, 50000000.0, 150.0],
  data_max_=[300.0, 305.0, 298.0, 300.0, 500000000.0, 300.0],
  data_range_=[150.0, 157.0, 153.0, 150.0, 450000000.0, 150.0]
)
```

### 3. Criação de Sequências

#### Janelas Deslizantes (Sliding Windows)

```
Dados brutos (1500 dias, exemplo):
┌──────────────────────────────────────────────────────────────┐
│ t1  t2  t3  t4  t5  t6  ... t1499  t1500                     │
└──────────────────────────────────────────────────────────────┘

Com Sequence Length = 60:

Sequência 1:  [t1:t60]    → predict t61
Sequência 2:  [t2:t61]    → predict t62
Sequência 3:  [t3:t62]    → predict t63
...
Sequência 1441: [t1441:t1500] → predict t1501 (não existe!)

Resultado:
  • 1440 sequências de 1500 dados (1500 - 60)
  • Sobreposição: 59/60 dias compartilhados entre sequências
  • Data Augmentation implícito (diferentes "visões")
  • Cada dia participa de múltiplas sequências
```

**Código Simplificado:**
```python
sequence_length = 60
sequences = []

for i in range(len(data) - sequence_length):
    X_seq = data[i:i+sequence_length, :]  # 60 dias, 6 features
    y_target = data[i+sequence_length, 3]  # Dia 61, coluna Close
    sequences.append((X_seq, y_target))
```

### 4. Split de Dados (Temporal)

```
Total: 1440 sequências (exemplo)

┌─────────────────┬──────────────┬─────────────┐
│   TREINO (70%)  │  VAL (20%)   │ TESTE (10%) │
│   1008 seqs     │  288 seqs    │  144 seqs   │
│   2018-2023     │  2023-2024   │  2024-2025  │
└─────────────────┴──────────────┴─────────────┘

⚠️ CRÍTICO: Split temporal (sem shuffle!)
   • Treino: Dados mais antigos
   • Teste: Dados mais recentes
   • Simula realidade (não vemos futuro)
```

**Por que não shuffle?**
- ❌ Shuffle misturaria passado/futuro → data leakage
- ❌ Modelo veria "dados do futuro" durante treino
- ✅ Split temporal = único válido para séries temporais

### 5. Saídas do Pipeline

`process_pipeline()` retorna:
```python
X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names
```

**Shapes:**
- **X_train:** (1008, 60, 6) - Sequências de treino
- **y_train:** (1008,) - Targets (Close normalizado)
- **X_val:** (288, 60, 6) - Sequências de validação
- **y_val:** (288,) - Targets de validação
- **X_test:** (144, 60, 6) - Sequências de teste
- **y_test:** (144,) - Targets de teste
- **scaler:** MinMaxScaler fitted (necessário para API)
- **feature_names:** ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

## 🏗️ Arquitetura do Modelo

### Estrutura (V1 Baseline)
**Função:** `build_model()` em [src/scripts/utils_train.py](../src/scripts/utils_train.py)

```python
Input: (60, 6)  # 60 timesteps, 6 features
    ↓
LSTM(100, return_sequences=True, recurrent_dropout=0.0)
    ↓
Dropout(0.25)
    ↓
LSTM(50, return_sequences=False)
    ↓
Dropout(0.25)
    ↓
Dense(16, activation='relu', kernel_regularizer=l2(0.003))
    ↓
Dense(1)  # Saída: Close normalizado
```

### Hiperparâmetros (Baseline)
```python
model_config = {
    'sequence_length': 60,
    'lstm_units': [100, 50],
    'dropout_rate': 0.25,
    'recurrent_dropout': 0.0,
    'dense_units': 16,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'regularization_l2': 0.003,
    'early_stop_patience': 15,
    'reduce_lr_patience': 8,
    'train_ratio': 0.7,
    'val_ratio': 0.2
}
```

### Compilação
- **Otimizador:** `Adam(learning_rate=0.001)`
- **Loss:** `mean_squared_error` (MSE)
- **Métricas:** `['mape']` (Mean Absolute Percentage Error)

### Callbacks de Treinamento
```python
EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    verbose=1
)
```

## 🎯 Treinamento (via Notebook)

### Processo Completo
**Notebook:** [notebooks/02_treino.ipynb](../notebooks/02_treino.ipynb)

#### 1. Configuração de Hiperparâmetros
```python
SYMBOL = 'AAPL'
START_DATE = '2018-01-01'
END_DATE = '2025-12-31'

model_config = {
    'sequence_length': 60,
    'lstm_units': [100, 50],
    'dropout_rate': 0.25,
    # ... (ver seção Arquitetura)
}
```

#### 2. Processamento de Dados
```python
processor = StockDataProcessor(SYMBOL, START_DATE, END_DATE)
processed_df, lstm_data = processor.process_pipeline(
    sequence_length=model_config['sequence_length'],
    train_ratio=model_config['train_ratio'],
    val_ratio=model_config['val_ratio']
)

X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names = lstm_data
```

#### 3. Construção e Treinamento
```python
input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 6)
model = build_model(model_config, input_shape)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

#### 4. Avaliação e Métricas
- **Inversão de normalização:** `inverse_transform_predictions()`
- **Cálculo de métricas:** MAE, RMSE, MAPE, R²
- **Comparação:** Treino vs Validação vs Teste

#### 5. MLflow Tracking
- **Experiment:** `LSTM_Stock_Prediction`
- **Parâmetros:** Todos os valores de `model_config`
- **Métricas:** MAE, RMSE, MAPE, val_loss, R²
- **Artefatos:**
  - Modelo: `mlflow.keras.log_model()`
  - Scaler: `scaler.pkl`
  - Plots: validação, loss, resíduos

#### 6. Salvamento de Artefatos
**Essenciais para API:**
- `models/lstm_model.keras` - Modelo treinado
- `models/scaler.pkl` - MinMaxScaler fitted

**Diagnóstico (opcional):**
- `data/temp_plots/validation_plot.png`
- `data/temp_plots/loss_plot.png`
- `data/temp_plots/residuals_*.png`

### Como Treinar um Novo Modelo

1. **Abrir notebook:**
   ```bash
   # No VS Code: Open notebooks/02_treino.ipynb
   # Ou Jupyter: jupyter lab notebooks/02_treino.ipynb
   ```

2. **Ajustar hiperparâmetros:** Editar célula 3 (`model_config`)

3. **Executar todas as células:** Run All (Ctrl+Shift+Enter)

4. **Verificar artefatos:**
   ```bash
   ls -lh models/
   # Deve mostrar lstm_model.keras e scaler.pkl
   ```

5. **Testar API:**
   ```bash
   make run-local  # Terminal 1
   make test-api   # Terminal 2
   ```

## 📈 Métricas de Avaliação

### Definições
- **MAE (Mean Absolute Error):** Erro absoluto médio em $ (quanto o modelo erra em média)
- **RMSE (Root Mean Squared Error):** Raiz do erro quadrático médio (penaliza erros grandes)
- **MAPE (Mean Absolute Percentage Error):** Erro percentual médio (ex: 3.84% = ~$10 de erro em $260)
- **R² Score:** Proporção da variância explicada (0-100%, quanto mais próximo de 100% melhor)

### Resultados do Modelo Atual (V1 Baseline)
**Ação:** AAPL (2018-2025) | **Data:** Jan 2026

| Conjunto   | MAE ($) | RMSE ($) | MAPE (%) | R² Score (%) |
|------------|---------|----------|----------|-------------|
| **Treino** | $6.31   | $9.86    | 2.63%    | 99.28%      |
| **Val**    | $8.72   | $11.30   | 3.68%    | 93.31%      |
| **Teste**  | $9.15   | $12.46   | 3.84%    | 85.19%      |

### Interpretação
- **R² gap (Treino → Teste):** 99.28% → 85.19% = **14.09%**
  - ✅ **Aceitável:** Gap < 20% indica bom equilíbrio (não overfit severo)
  - Modelo generaliza bem para dados não vistos

- **MAPE Teste: 3.84%**
  - Preço médio AAPL ~$250 → Erro médio de ~$9.60
  - ✅ **Excelente:** < 5% é considerado muito bom para previsão de ações

- **MAE crescente:** $6.31 → $8.72 → $9.15
  - Esperado: dados de teste são os mais recentes (maior volatilidade)
  - Ainda dentro de limites aceitáveis

### Como Calcular
**Função:** `calculate_metrics()` em [src/scripts/utils_train.py](../src/scripts/utils_train.py)

```python
def calculate_metrics(y_true, y_pred, split_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Split': split_name,
        'MAE ($)': mae,
        'RMSE ($)': rmse,
        'MAPE (%)': mape,
        'R² Score': r2
    }
```

## 🔧 Dicas de Hyperparameter Tuning

### Parâmetros Principais

#### 1. Sequence Length (Janela Temporal)
```python
sequence_length: 40–100
```
- **Recomendado:** 60-80 dias (2-3 meses)
- **Menor (30-40):** Captura padrões curto prazo, treina mais rápido
- **Maior (80-100):** Captura tendências longas, requer mais dados
- **Trade-off:** Mais longo = menos samples de treino

#### 2. Arquitetura LSTM
```python
lstm_units: [[100, 50], [128, 64], [192, 128], [256, 128]]
```
- **Baseline:** [100, 50] - Bom equilíbrio
- **Mais complexo:** [256, 128] - Se tem muito dados (> 2000 dias)
- **Mais simples:** [64, 32] - Se tem pouco dados ou quer evitar overfit

#### 3. Regularização
```python
dropout_rate: 0.15–0.30  # Recomendado: 0.20-0.25
recurrent_dropout: 0.0–0.15  # Cuidado: pode desacelerar treino
regularization_l2: 0.001–0.005  # Recomendado: 0.003
```
- **Dropout alto (> 0.3):** Use se R² treino >> R² teste (overfit)
- **L2 alto (> 0.01):** Pode sub-ajustar, comece com 0.003

#### 4. Camada Dense Final
```python
dense_units: 16–64
```
- **16-32:** Geralmente suficiente
- **64+:** Se arquitetura LSTM é grande (256+ units)

#### 5. Otimização
```python
learning_rate: [0.001, 0.0007, 0.0005, 0.0003]
batch_size: [16, 32, 64]
```
- **LR 0.001:** Baseline (Adam)
- **LR 0.0003-0.0005:** Use se loss oscilar muito
- **Batch 32:** Baseline
- **Batch 16:** Se pouco dados ou quer mais atualizações
- **Batch 64:** Se muito dados e quer treino mais rápido

#### 6. Callbacks
```python
early_stop_patience: 10–20  # Recomendado: 15
reduce_lr_patience: 5–10   # Recomendado: 8
reduce_lr_factor: 0.5       # Reduz LR pela metade
```

### Estratégia de Tuning

#### Passo 1: Baseline (Começar aqui)
```python
model_config = {
    'sequence_length': 60,
    'lstm_units': [100, 50],
    'dropout_rate': 0.25,
    'learning_rate': 0.001,
    'batch_size': 32,
}
```

#### Passo 2: Se Underfitting (R² teste < 70%)
- ✅ Aumentar complexidade: `lstm_units = [128, 64]` ou `[192, 128]`
- ✅ Reduzir dropout: `dropout_rate = 0.15`
- ✅ Aumentar `sequence_length = 80`
- ✅ Mais épocas (se parou cedo)

#### Passo 3: Se Overfitting (R² treino >> R² teste, gap > 20%)
- ✅ Aumentar dropout: `dropout_rate = 0.30`
- ✅ Adicionar recurrent_dropout: `0.1`
- ✅ Aumentar L2: `regularization_l2 = 0.005`
- ✅ Reduzir complexidade: `lstm_units = [64, 32]`
- ✅ Early stopping mais agressivo: `patience = 10`

#### Passo 4: Ajuste Fino
- Testar learning rates menores: `0.0005`, `0.0003`
- Ajustar `dense_units`
- Experimentar `batch_size`

### Exemplo de Experimentos

| Experimento | LSTM Units | Dropout | LR    | Seq Len | R² Teste | MAE ($) | Status |
|-------------|------------|---------|-------|---------|----------|---------|--------|
| Baseline    | [100, 50]  | 0.25    | 0.001 | 60      | 85.19%   | $9.15   | ✅ Bom  |
| Exp 1       | [128, 64]  | 0.25    | 0.001 | 60      | 87.32%   | $8.76   | ✅ Melhor|
| Exp 2       | [256, 128] | 0.25    | 0.001 | 60      | 84.12%   | $9.89   | ⚠️ Overfit|
| Exp 3       | [128, 64]  | 0.30    | 0.0007| 80      | 88.15%   | $8.21   | ✅ Melhor!|
| Exp 4       | [64, 32]   | 0.20    | 0.001 | 60      | 79.45%   | $11.32  | ❌ Underfit|

### Monitoramento no MLflow

Todos os experimentos são registrados automaticamente:
```bash
cd notebooks
mlflow ui --port 5000
# Acesse: http://localhost:5000
```

**Comparar experimentos:**
- Ordene por `mae` ou `r2_test` (menor/maior)
- Verifique gráficos de loss e resíduos
- Compare hiperparâmetros dos top 3 modelos

---

## 📞 Próximos Passos

### Implementar em Produção
👉 Ver [API REST - Guia Completo](api.md) para:
- Como carregar o modelo treinado
- Endpoints disponíveis
- Exemplos de integração
- Troubleshooting de API

### Melhorar o Modelo
📊 Experimente:
1. Ajustar hiperparâmetros (ver [seção de Tuning](#🔧-dicas-de-hyperparameter-tuning))
2. Adicionar features técnicas (RSI, MACD, etc.)
3. Testar com outras ações (TSLA, MSFT, GOOGL)
4. Aumentar sequence_length (80, 100 dias)

### Monitorar Resultados
📈 Use MLflow:
```bash
cd notebooks
poetry run mlflow ui --port 5000
# Acesse: http://localhost:5000
```

---

## 📚 Documentação Relacionada

- **[← README Principal](../README.md)** - Visão geral e quick start
- **[🔌 API REST](api.md)** - Como usar o modelo via API
- **[📖 Índice de Docs](README.md)** - Todas as documentações

---

**Última Atualização:** 8 de Janeiro de 2026  
**Versão:** V1.20260108 (Baseline)  
**Status:** ✅ Documentação Completa