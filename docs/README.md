# 📚 Documentação - LSTM Stock Prediction

Índice centralizado de toda a documentação do projeto.

---

## 🎯 Documentos Principais

### 1. [📖 API REST - Guia Completo](api.md)
**Para:** Desenvolvedores, DevOps, Integradores

**Conteúdo:**
- Endpoints disponíveis (`/` e `/predict`)
- Formato de entrada/saída
- Exemplos práticos (Python, cURL, JavaScript)
- Testes AWS com Load Balancer
- Troubleshooting
- Métricas de latência
- Validação de dados

**Quando usar:** Quando precisar integrar ou consumir a API

---

### 2. [🧠 Modelo LSTM - Especificação Técnica](model.md)
**Para:** Data Scientists, ML Engineers, Pesquisadores

**Conteúdo:**
- Arquitetura detalhada (fluxo de dados, células LSTM)
- Pipeline de dados (coleta, normalização, sequências)
- Matemática (MinMaxScaler, backpropagation, loss functions)
- Hiperparâmetros e tuning
- Métricas de avaliação (R², MAE, RMSE, MAPE)
- Troubleshooting de treinamento

**Quando usar:** Quando precisar entender, retreinar ou melhorar o modelo

---



## 🚀 Fluxo de Leitura Recomendado

### Para Começar Rápido:
1. Leia o [README.md](../README.md) principal (visão geral)
2. Siga o Quick Start para rodar a API
3. Teste com `make test-api`

### Para Integrar a API:
1. [API REST - Guia Completo](api.md) (leitura completa)
2. Seção "Exemplos de Uso"
3. Seção "Troubleshooting"

### Para Entender o Modelo:
1. [Modelo LSTM - Especificação Técnica](model.md)
2. Seção "Pipeline de Dados"
3. Seção "Arquitetura Detalhada"

### Para Retreinar o Modelo:
1. [Modelo LSTM - Seção Treinamento](model.md#🎯-treinamento-via-notebook)
2. Abrir `notebooks/02_treino.ipynb`
3. Seguir seção "Como Treinar um Novo Modelo"
4. Ver seção "Hyperparameter Tuning"

### Para Deploy em Produção:
1. [README.md - Seção Docker](../README.md#🐳-docker)
2. [API - Seção Produção](api.md#🔐-considerações-de-produção)
3. Configurar monitoramento (MLflow, logs)

---

## 📊 Estrutura da Documentação

```
docs/
├── README.md           ← Você está aqui (índice)
├── api.md              ← Guia da API REST
└── model.md            ← Especificação do modelo LSTM
```

---

## 🔗 Links Rápidos

| Necessidade | Link Direto |
|-------------|-------------|
| Rodar API localmente | [README - Quick Start](../README.md#🚀-quick-start---api) |
| Testar API na AWS | [API - Testes AWS](api.md#🌐-testando-na-aws-load-balancer) |
| Entender arquitetura | [Model - Arquitetura](model.md#📐-arquitetura-detalhada) |
| Treinar novo modelo | [Model - Treinamento](model.md#🎯-treinamento-via-notebook) |
| Fazer tuning | [Model - Tuning](model.md#🔧-dicas-de-hyperparameter-tuning) |
| Ver métricas | [README - Desempenho](../README.md#📊-desempenho-do-modelo) |
| Troubleshooting API | [API - Troubleshooting](api.md#🐛-troubleshooting) |
| Troubleshooting Modelo | [Model - Troubleshooting](model.md#🔍-troubleshooting) |

---

## 📖 Glossário Rápido

| Termo | Significado |
|-------|-------------|
| **LSTM** | Long Short-Term Memory (tipo de rede neural recorrente) |
| **OHLCV** | Open, High, Low, Close, Volume (dados de ações) |
| **MinMaxScaler** | Normalização de dados para [0, 1] |
| **R² Score** | Coeficiente de determinação (0-100%, quanto maior melhor) |
| **MAE** | Mean Absolute Error (erro médio em $) |
| **RMSE** | Root Mean Squared Error (penaliza erros grandes) |
| **MAPE** | Mean Absolute Percentage Error (erro em %) |
| **Sequence Length** | Número de dias históricos usados (60) |
| **Dropout** | Técnica de regularização (desativa neurônios) |
| **Early Stopping** | Para treinamento quando validação não melhora |
| **MLflow** | Ferramenta de tracking de experimentos |

---

## 🆘 Precisa de Ajuda?

### Erro ao rodar a API?
→ Ver [API - Troubleshooting](api.md#🐛-troubleshooting)

### Modelo não converge?
→ Ver [Model - Troubleshooting](model.md#🔍-troubleshooting)

### Dúvidas sobre features?
→ Ver [Model - Pipeline de Dados](model.md#📊-pipeline-de-dados)

### Performance ruim?
→ Ver [Model - Hyperparameter Tuning](model.md#🔧-dicas-de-hyperparameter-tuning)

### Como configurar .env?
→ Ver [README - Segurança](../README.md#🔒-segurança-e-dados-sensíveis)

---

## 📞 Suporte por Perfil

| Você é... | Comece por... | Documentos Essenciais |
|-----------|---------------|----------------------|
| 👨‍💼 **Gestor/PM** | [README principal](../README.md) | README.md |
| 👨‍💻 **Desenvolvedor** | [API - Guia](api.md) | api.md |
| 🧠 **Data Scientist** | [Model - Especificação](model.md) | model.md + notebooks/ |
| 🚀 **DevOps** | [README - Docker](../README.md#🐳-docker) | README.md + Dockerfile |

---

**Última Atualização:** 8 de Janeiro de 2026  
**Versão do Modelo:** V1.20260108  
**Status:** ✅ Documentação Completa
