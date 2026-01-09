APP_NAME=lstm-api
AWS_REGION=sa-east-1
AWS_ACCOUNT_ID=507879919760
ECR_REPO=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP_NAME)

setup:
	poetry install

train:
	poetry run python src/ml/train.py

run-local:
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test-api:
	poetry run python scripts/teste_local.py

# Docker Local
docker-build:
	docker build -t $(APP_NAME) .

docker-run:
	docker run -p 8000:8000 $(APP_NAME)

# AWS ECR Login e Criação de Repo (se não existir)
aws-login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

# aws-create-repo:
# 	aws ecr create-repository --repository-name $(APP_NAME) --region $(AWS_REGION) || true

aws-push: aws-login aws-create-repo
	docker tag $(APP_NAME):latest $(ECR_REPO):latest
	docker push $(ECR_REPO):latest

# deploy-ecs:  # Após configurar manualmente no console inicialmente
# 	aws ecs update-service --cluster lstm-cluster --service lstm-service --force-new-deployment --region sa-east-1

# Teste
test-api:
	poetry run python scripts/teste_local.py

git-push:
	git add .
	git commit -m "Update project MVP"
	git push origin main