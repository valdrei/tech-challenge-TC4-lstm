# 🚀 LSTM Stock Price Prediction Model + API

## Tech Challenge Fase 4 - PosTech FIAP

**Status:** ✅ PRONTO PARA PRODUÇÃO  
**Performance:** R² ~85-93% | MAE ~$9-12 | MAPE ~3-4%  
**API:** ✅ IMPLEMENTADA E TESTADA  
**Última Atualização:** 8 de Janeiro de 2026

---

## 📌 Visão Geral

Modelo de Deep Learning (LSTM) para previsão de preços de ações com API REST completa. Treinado em dados históricos OHLCV com normalização MinMaxScaler e arquitetura de 2 camadas LSTM.

### 🎯 Objetivos Atendidos
- ✅ Coleta de Dados (Yahoo Finance via yfinance)
- ✅ Pré-processamento e Normalização (MinMaxScaler)
- ✅ Modelo LSTM (2 layers, ~78K parâmetros)
- ✅ Treinamento com Early Stopping
- ✅ Avaliação (MAE, RMSE, MAPE, R²)
- ✅ API REST (FastAPI)
- ✅ Containerização (Docker)

### 🧠 Tecnologias
- **Framework ML:** TensorFlow/Keras 2.12+
- **API:** FastAPI 0.104+
- **Linguagem:** Python 3.11.5
- **Containerização:** Docker
- **Cloud:** AWS ECR (opcional)

---

## 🚀 Quick Start - API

### Opção 1: Rodar Localmente (Recomendado)
```bash
# 1. Instalar dependências
poetry install

# 2. Iniciar API
make run-local
# ou: poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Opção 2: Com Docker
```bash
make docker-build
make docker-run
```

**Acesse:**
- 🌐 API: http://localhost:8000
- 📖 Swagger UI: http://localhost:8000/docs
- 📚 ReDoc: http://localhost:8000/redoc

### Testar a API

**Teste Local:**
```bash
# Terminal 1: Rodar API
make run-local

# Terminal 2: Testar com dados reais da AAPL
make test-api
```

**Teste AWS (Load Balancer):**
```bash
# Opção 1: Modo interativo (solicita URL)
make test-aws

# Opção 2: URL direto
make test-aws-url URL=http://lstm-alb-xxxx.sa-east-1.elb.amazonaws.com
```

👉 **Guia completo:** [docs/api.md](docs/api.md)

---

## 📚 Documentação

### 🎯 Para Começar:

| Perfil | Documento | Descrição |
|--------|-----------|-----------|
| **👨‍💻 Desenvolvedores** | [docs/api.md](docs/api.md) | Guia completo da API REST |
| **🧠 Data Scientists** | [docs/model.md](docs/model.md) | Arquitetura, treinamento, tuning |
| **👨‍💼 Gestores** | Este README | Visão geral e métricas |

### 📖 Documentação Completa:

1. **[docs/api.md](docs/api.md)** - API REST (endpoints, exemplos, testes AWS, segurança)
2. **[docs/model.md](docs/model.md)** - Modelo LSTM (arquitetura detalhada, matemática, tuning)
3. **[docs/README.md](docs/README.md)** - Índice da documentação
4. **[.env.example](.env.example)** - Template de variáveis de ambiente

---

## 📊 Desempenho do Modelo

### Métricas (Exemplo: AAPL 2018-2025)

| Conjunto | R² Score | MAE ($) | RMSE ($) | MAPE (%) |
|----------|----------|---------|----------|----------|
| **Treino** | 99.28% | $6.31 | $9.86 | 2.63% |
| **Validação** | 93.31% | $8.72 | $11.30 | 3.68% |
| **Teste** | 85.19% | $9.15 | $12.46 | 3.84% |

### Interpretação
- ✅ **R² Teste: 85.19%** - Excelente (explica 85% da variação)
- ✅ **MAE: $9.15** - Erro médio pequeno (~3.7% do preço médio)
- ✅ **Gap Treino→Teste: 14%** - Baixo overfitting
- ✅ **MAPE: 3.84%** - Muito bom para séries financeiras

### Arquitetura
```
Input (60 dias, 6 features)
    ↓
LSTM(100) + Dropout(0.25)
    ↓
LSTM(50) + Dropout(0.25)
    ↓
Dense(16, ReLU) + L2(0.003)
    ↓
Output(1) - Preço previsto
```

**Features utilizadas (6):** Open, High, Low, Close, Volume, Adj Close  
**Total de parâmetros:** ~78,000  
**Normalização:** MinMaxScaler [0, 1]

---

## 🛠️ Setup de Desenvolvimento

### Pré-requisitos
1. **pyenv** (gerenciador de versão Python):
   ```bash
   # macOS (Homebrew)
   brew install pyenv
   
   # Linux (Ubuntu/Debian)
   curl https://pyenv.run | bash
   ```
   
2. **Python 3.11.5** (via pyenv):
   ```bash
   pyenv install 3.11.5
   pyenv local 3.11.5  # Define versão para este projeto
   python --version    # Verifica (deve ser 3.11.5)
   ```
   > O arquivo `.python-version` garante que todos usem a mesma versão.

3. **Poetry** (gerenciador de dependências):
   ```bash
   pip install poetry
   poetry install      # Instala dependências do pyproject.toml
   ```

### Execução Local

```bash
# 1. Ativar ambiente Poetry
poetry shell

# 2. Opção A: Treinar novo modelo
# Abra notebooks/02_treino.ipynb no VS Code/Jupyter
# Execute todas as células sequencialmente
# Artefatos salvos em: models/lstm_model.keras, models/scaler.pkl

# 2. Opção B: Usar modelo existente + API
make run-local

# 3. Testar API (em outro terminal)
make test-api
```

### Comandos Úteis (Makefile)

```bash
# Desenvolvimento
make setup          # Instalar dependências com Poetry
make run-local      # Iniciar API localmente (porta 8000)
make test-api       # Testar API com dados reais da AAPL

# Docker
make docker-build   # Build da imagem Docker
make docker-run     # Rodar container localmente (porta 8000)

# AWS
make aws-login      # Autenticar no ECR
make aws-push       # Build + Tag + Push para ECR
make test-aws       # Testar API na AWS (modo interativo)
make test-aws-url   # Testar API na AWS com URL específico
make git-push       # Git add + commit + push
```

> **Nota:** O treinamento é feito via notebook interativo, não via CLI

---

## 📁 Estrutura do Projeto

```
lstm-predict/
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI app (endpoints /predict)
│   ├── scripts/
│   │   ├── cache_manager.py  # Cache de dados históricos
│   │   ├── data_processor.py # StockDataProcessor class
│   │   ├── utils_train.py    # build_model, métricas, etc.
│   │   ├── teste_local.py    # Script de teste da API
│   │   └── teste_aws.py      # Script de teste AWS
│   └── __init__.py
├── notebooks/
│   ├── 01_exploracao.ipynb   # Análise exploratória
│   ├── 02_treino.ipynb       # ⭐ Pipeline de treinamento
│   ├── mlruns/               # MLflow tracking
│   └── data/cache/           # Cache de dados yfinance
├── models/                    # 🧠 Modelo treinado
│   ├── lstm_model.keras      # Modelo LSTM (~2 MB)
│   └── scaler.pkl            # MinMaxScaler fitted
├── docs/                      # 📚 Documentação completa
│   ├── README.md             # Índice de documentação
│   ├── api.md                # ⭐ Guia completo da API
│   └── model.md              # ⭐ Guia do modelo LSTM
├── data/
│   ├── raw/                  # Dados brutos (CSV)
│   └── temp_plots/           # Gráficos temporários
├── Dockerfile                # Container image
├── Makefile                  # Comandos de desenvolvimento
├── pyproject.toml            # Poetry dependencies
├── .python-version           # Python 3.11.5
└── README.md                 # Este arquivo
```

---

## 🐳 Docker

### Build e Run
```bash
# Build da imagem
make docker-build
# ou: docker build -t lstm-api .

# Run localmente
make docker-run
# ou: docker run -p 8000:8000 lstm-api
```

### Push para AWS ECR
```bash
# Autenticar
make aws-login

# Build + Tag + Push
make aws-push
```

---

## 📊 API - Exemplo de Uso

### Endpoint Principal: `POST /predict`

**Entrada:** 60 dias de dados OHLCV (não normalizados)

```json
{
  "last_60_days": [
    [open, high, low, close, volume, adj_close],  // Dia 1
    [open, high, low, close, volume, adj_close],  // Dia 2
    ...
    [open, high, low, close, volume, adj_close]   // Dia 60
  ]
}
```

**Saída:** Previsão do próximo preço de fechamento

```json
{
  "prediction": 245.67
}
```

### Exemplo Completo (Python)

```python
import yfinance as yf
import requests

# 1. Baixar dados históricos (últimos 3 meses)
df = yf.download('AAPL', period='3mo', auto_adjust=False, progress=False)

# 2. Preparar últimos 60 dias (6 features)
last_60 = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].tail(60)

# 3. Fazer requisição
response = requests.post(
    "http://localhost:8000/predict",
    json={"last_60_days": last_60.values.tolist()}
)

# 4. Ver resultado
print(f"Preço atual: ${last_60['Close'].iloc[-1]:.2f}")
print(f"Previsão:    ${response.json()['prediction']:.2f}")
```

### Teste Rápido

```bash
# Terminal 1: Iniciar API
make run-local

# Terminal 2: Testar com dados reais da AAPL
make test-api
```

**Documentação completa:** [docs/api.md](docs/api.md)

---

## 🎯 Pipeline de Treinamento

### 1. Notebook Interativo (Recomendado)

Abra `notebooks/02_treino.ipynb` no VS Code/Jupyter e execute:

```python
# Célula 1: Configuração
SYMBOL = 'AAPL'
START_DATE = '2018-01-01'
END_DATE = '2025-12-31'

model_config = {
    'sequence_length': 60,
    'lstm_units': [100, 50],
    'dropout_rate': 0.25,
    'learning_rate': 0.001,
    # ... ver notebook para config completa
}

# Célula 2: Processar dados
processor = StockDataProcessor(SYMBOL, START_DATE, END_DATE)
processed_df, lstm_data = processor.process_pipeline(...)

# Célula 3: Treinar modelo
model = build_model(model_config, input_shape)
history = model.fit(X_train, y_train, ...)

# Célula 4: Avaliar
y_test_pred = model.predict(X_test)
metrics = calculate_metrics(y_test, y_test_pred)

# Célula 5: Salvar artefatos
model.save('models/lstm_model.keras')
joblib.dump(scaler, 'models/scaler.pkl')
```

### 2. Artefatos Gerados

- ✅ `models/lstm_model.keras` - Modelo treinado (~2 MB)
- ✅ `models/scaler.pkl` - MinMaxScaler fitted
- ✅ `data/temp_plots/` - Gráficos de diagnóstico
- ✅ `notebooks/mlruns/` - Tracking MLflow

### 3. Visualizar Experimentos (MLflow)

```bash
cd notebooks
poetry run mlflow ui --port 5000
# Acesse: http://localhost:5000
```

**Guia completo:** [docs/model.md](docs/model.md)

---

## ✅ Requisitos Tech Challenge - Checklist

- ✅ **Coleta de Dados** - Yahoo Finance (yfinance)
- ✅ **Pré-processamento** - MinMaxScaler, janelas de 60 dias
- ✅ **Modelo LSTM** - 2 camadas, ~78K parâmetros
- ✅ **Treinamento** - Adam optimizer, Early Stopping, ReduceLROnPlateau
- ✅ **Avaliação** - MAE, RMSE, MAPE, R² em 3 conjuntos (treino/val/teste)
- ✅ **Salvamento** - Keras format (.keras) + Scaler (.pkl)
- ✅ **API REST** - FastAPI com endpoint /predict
- ✅ **Documentação** - README.md + docs/ completos
- ✅ **Containerização** - Dockerfile + docker-compose
- ✅ **Testes** - Script de teste automatizado

---

## 🎓 Tecnologias Utilizadas

| Categoria | Tecnologia | Versão |
|-----------|------------|--------|
| **Linguagem** | Python | 3.11.5 |
| **ML Framework** | TensorFlow/Keras | 2.12+ |
| **API** | FastAPI | 0.104+ |
| **Dados** | yfinance | 0.2+ |
| **Normalização** | scikit-learn | 1.3+ |
| **Tracking** | MLflow | 2.9+ |
| **Containerização** | Docker | 24+ |
| **Orquestração** | Poetry | 1.7+ |

---

## 📝 Informações do Projeto

- **Criado:** Janeiro 2026
- **Status:** ✅ Completo e Validado
- **Versão:** V1.20260108 (Baseline)
- **Próximos Passos:** Deploy em produção (AWS ECS)

---

## � Segurança e Dados Sensíveis

### ⚠️ Arquivos que NÃO devem subir para o Git

O projeto já está configurado com `.gitignore` para proteger dados sensíveis:

#### 1. Modelos Treinados (Arquivos Grandes)
```
models/                    # Modelos treinados (~2-50 MB)
├── lstm_model.keras       # ❌ NÃO COMMITAR (arquivo grande)
└── scaler.pkl             # ❌ NÃO COMMITAR
```

#### 2. Dados de Treinamento
```
data/                      # Dados brutos e processados
notebooks/data/            # Cache do yfinance
notebooks/mlruns/          # Experimentos MLflow
```

#### 3. Credenciais AWS
```
.env                       # ❌ NÃO COMMITAR (credenciais)
.env.local                 # ❌ NÃO COMMITAR
infra/terraform.tfvars     # ❌ NÃO COMMITAR (variáveis sensíveis)
infra/.terraform/          # ❌ NÃO COMMITAR (estado local)
infra/*.tfstate*           # ❌ NÃO COMMITAR (estado Terraform)
```

#### 4. Configurações IDE/Locais
```
.vscode/                   # Configurações pessoais do editor
.idea/                     # Configurações PyCharm
__pycache__/               # Cache Python
*.log                      # Logs
```

### ✅ Como Configurar Dados Sensíveis Localmente

#### Opção 1: Variáveis de Ambiente (Recomendado)

**Crie um arquivo `.env` na raiz do projeto:**
```bash
# .env (NÃO COMMITAR)
AWS_PROFILE=default
AWS_REGION=sa-east-1
AWS_ACCOUNT_ID=123456789012
LOAD_BALANCER_URL=http://lstm-alb-xxxx.sa-east-1.elb.amazonaws.com
```

**Carregue no código:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
aws_account = os.getenv('AWS_ACCOUNT_ID')
```

#### Opção 2: AWS CLI Profile

**Configure suas credenciais AWS:**
```bash
# Configurar credenciais (interativo)
aws configure --profile lstm-api

# Ou editar manualmente
vim ~/.aws/credentials
```

**Conteúdo de `~/.aws/credentials`:**
```ini
[lstm-api]
aws_access_key_id = SEU_ACCESS_KEY
aws_secret_access_key = SUA_SECRET_KEY
region = sa-east-1
```

**Usar no Makefile:**
```bash
make aws-login AWS_PROFILE=lstm-api
```

#### Opção 3: Terraform Variables

**Crie `infra/terraform.tfvars` (NÃO COMMITAR):**
```hcl
aws_region     = "sa-east-1"
project_name   = "lstm-api"
environment    = "production"
ecr_repository = "lstm-api"
```

### 🔐 Boas Práticas de Segurança

1. **✅ SEMPRE** verifique antes de commitar:
   ```bash
   git status  # Ver arquivos staged
   git diff    # Ver mudanças
   ```

2. **✅ NUNCA** commite:
   - Credenciais AWS (access keys, secret keys)
   - URLs de produção com tokens
   - Arquivos `.env` ou `.tfvars`
   - Modelos treinados (use Git LFS ou S3)

3. **✅ USE** secrets managers para produção:
   - AWS Secrets Manager
   - AWS Systems Manager Parameter Store
   - HashiCorp Vault

4. **✅ ROTACIONE** credenciais regularmente:
   ```bash
   aws iam create-access-key --user-name lstm-api
   aws iam delete-access-key --access-key-id OLD_KEY
   ```

### 📦 Como Compartilhar Modelos Treinados

**Opção 1: AWS S3 (Recomendado para produção)**
```bash
# Upload
aws s3 cp models/lstm_model.keras s3://seu-bucket/models/
aws s3 cp models/scaler.pkl s3://seu-bucket/models/

# Download (outro dev)
aws s3 cp s3://seu-bucket/models/lstm_model.keras models/
aws s3 cp s3://seu-bucket/models/scaler.pkl models/
```

**Opção 2: Git LFS (para arquivos grandes)**
```bash
# Instalar Git LFS
git lfs install

# Rastrear modelos
git lfs track "*.keras"
git lfs track "*.pkl"

# Commitar normalmente
git add .gitattributes models/
git commit -m "Add trained models"
```

**Opção 3: Google Drive/Dropbox (desenvolvimento)**
- Compartilhe link do arquivo
- Equipe baixa manualmente para `models/`

### 🚨 Vazou Credenciais no Git?

**Ação Imediata:**
```bash
# 1. Rotacionar credenciais IMEDIATAMENTE
aws iam create-access-key --user-name seu-usuario
aws iam delete-access-key --access-key-id CHAVE_VAZADA

# 2. Remover do histórico Git (use git-filter-repo)
pip install git-filter-repo
git filter-repo --path .env --invert-paths

# 3. Force push (CUIDADO: coordene com equipe)
git push origin --force --all
```

---

## �📞 Suporte e Contato

### Para Diferentes Perfis

| Você é... | Leia... | Próximos Passos |
|-----------|---------|-----------------|
| 👨‍💼 **Gestor/PM** | Este README | Ver métricas de desempenho |
| 👨‍💻 **Desenvolvedor** | [docs/api.md](docs/api.md) | Implementar integração |
| 🧠 **Data Scientist** | [docs/model.md](docs/model.md) | Experimentar tuning |
| 🚀 **DevOps** | Makefile + Dockerfile | Configurar deploy |

### Recursos Úteis

- **Swagger UI:** http://localhost:8000/docs (API interativa)
- **ReDoc:** http://localhost:8000/redoc (documentação alternativa)
- **MLflow UI:** http://localhost:5000 (tracking de experimentos)

### Problemas Comuns

1. **API não inicia:** Verifique se porta 8000 está livre
2. **Erro de modelo:** Certifique-se que `models/lstm_model.keras` existe
3. **MLflow não abre:** Verifique se está no diretório `notebooks/`
4. **Credenciais AWS:** Configure via `aws configure` ou arquivo `.env`
5. **Load Balancer URL:** Obtenha no AWS Console (EC2 → Load Balancers)

---

**Última Atualização:** 8 de Janeiro de 2026  
**Status:** ✅ PRONTO PARA PRODUÇÃO  
**Tech Challenge Fase 4 - PosTech FIAP**
