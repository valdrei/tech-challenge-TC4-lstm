APP_NAME=lstm-api
AWS_REGION?=sa-east-1
AWS_ACCOUNT_ID?=$(shell aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "SET_YOUR_ACCOUNT_ID")
ECR_REPO=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP_NAME)
AWS_PROFILE?=default

.PHONY: help setup train run-local test-api docker-build docker-run aws-login aws-push tf-init tf-validate tf-plan tf-apply tf-destroy deploy-all git-push

help:
	@echo "=== LSTM API Makefile ==="
	@echo ""
	@echo "📦 Setup:"
	@echo "  setup          - Instalar dependências com Poetry"
	@echo ""
	@echo "🚀 Desenvolvimento:"
	@echo "  run-local      - Iniciar API localmente (porta 8000)"
	@echo "  test-api       - Testar API local com dados da AAPL"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build   - Build da imagem Docker"
	@echo "  docker-run     - Rodar container localmente"
	@echo ""
	@echo "☁️  AWS:"
	@echo "  aws-login      - Autenticar no ECR"
	@echo "  aws-push       - Build + Push para ECR"
	@echo "  test-aws       - Testar API na AWS (pede URL interativa)"
	@echo "  test-aws-url   - Testar API na AWS com URL"
	@echo "                   Uso: make test-aws-url URL=http://load-balancer.com"
	@echo ""
	@echo "📝 Git:"
	@echo "  git-push       - Add + Commit + Push"

# ===== PYTHON / POETRY =====
setup:
	poetry install

train:
	poetry run python src/ml/train.py

run-local:
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test-api:
	poetry run python tests/teste_local.py

test-aws:
	@echo "Testando API na AWS..."
	@poetry run python tests/teste_aws.py

test-aws-url:
	@echo "Testando API na AWS com URL específica..."
	@echo "Uso: make test-aws-url URL=http://seu-load-balancer.elb.amazonaws.com"
	@poetry run python tests/teste_aws.py $(URL)

# ===== DOCKER =====
docker-build:
	docker build -t $(APP_NAME):latest .

docker-run:
	docker run -p 8000:8000 $(APP_NAME):latest

# ===== AWS ECR =====
aws-login:
	aws ecr get-login-password --region $(AWS_REGION) --profile $(AWS_PROFILE) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

aws-push: aws-login docker-build
	docker tag $(APP_NAME):latest $(ECR_REPO):latest
	docker push $(ECR_REPO):latest
	@echo "✓ Image pushed to ECR: $(ECR_REPO):latest"

# ===== GIT =====
git-push:
	git add .
	git commit -m "Update project MVP"
	git push origin main