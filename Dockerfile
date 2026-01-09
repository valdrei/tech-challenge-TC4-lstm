# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app
RUN pip install poetry && \
    poetry self add poetry-plugin-export
COPY pyproject.toml poetry.lock ./
# Exportar requirements do poetry para instalar via pip (mais leve no final)
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --only main

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app/requirements.txt .
# Instalar dependências (sem cache para economizar espaço)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código e modelos
COPY src/ ./src/
COPY models/ ./models/

# Expor porta e rodar (usa PORT fornecido pelo provedor, ex.: Render)
EXPOSE 8000
# Use shell form para permitir expansão de variável de ambiente PORT (Render define $PORT)
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]