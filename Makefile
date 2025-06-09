# Makefile for RAG Microservice

# Variables
PROJECT_ID := $(shell gcloud config get-value project)
REGION := us-central1
SERVICE_NAME := rag-microservice
GCR_PATH := gcr.io/$(PROJECT_ID)/$(SERVICE_NAME)
VERSION := $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")
TERRAFORM_DIR := ./infra

# Default target
.PHONY: all
all: help

# Help message
.PHONY: help
help:
	@echo "RAG Microservice Deployment Tool"
	@echo ""
	@echo "Usage:"
	@echo "  make build            - Build Docker image locally"
	@echo "  make push             - Push Docker image to GCR"
	@echo "  make terraform-init   - Initialize Terraform"
	@echo "  make terraform-plan   - Plan Terraform changes"
	@echo "  make terraform-apply  - Apply Terraform changes"
	@echo "  make deploy           - Build, push, and deploy the service"
	@echo "  make test             - Run all tests"
	@echo "  make load-test        - Run load tests with Locust"
	@echo "  make clean            - Clean up resources and temporary files"
	@echo ""

# Build the Docker image
.PHONY: build
build:
	@echo "Building Docker image $(GCR_PATH):$(VERSION)..."
	docker build -t $(GCR_PATH):$(VERSION) -t $(GCR_PATH):latest .

# Push the image to Google Container Registry
.PHONY: push
push: build
	@echo "Pushing Docker image to GCR..."
	docker push $(GCR_PATH):$(VERSION)
	docker push $(GCR_PATH):latest

# Initialize Terraform
.PHONY: terraform-init
terraform-init:
	@echo "Initializing Terraform..."
	cd $(TERRAFORM_DIR) && terraform init

# Plan Terraform changes
.PHONY: terraform-plan
terraform-plan:
	@echo "Planning Terraform changes..."
	cd $(TERRAFORM_DIR) && terraform plan \
		-var="project_id=$(PROJECT_ID)" \
		-var="region=$(REGION)" \
		-var="pinecone_api_key=$(PINECONE_API_KEY)" \
		-var="pinecone_env=$(PINECONE_ENV)" \
		-var="environment=$(ENVIRONMENT)"

# Apply Terraform changes
.PHONY: terraform-apply
terraform-apply:
	@echo "Applying Terraform changes..."
	cd $(TERRAFORM_DIR) && terraform apply -auto-approve \
		-var="project_id=$(PROJECT_ID)" \
		-var="region=$(REGION)" \
		-var="pinecone_api_key=$(PINECONE_API_KEY)" \
		-var="pinecone_env=$(PINECONE_ENV)" \
		-var="environment=$(ENVIRONMENT)"

# Full deployment
.PHONY: deploy
deploy: build push terraform-apply
	@echo "Deployment completed!"
	@echo "RAG Service should be available shortly at:"
	@cd $(TERRAFORM_DIR) && terraform output service_url

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	pytest -v tests/

# Run load tests with Locust
.PHONY: load-test
load-test:
	@echo "Running load tests..."
	@echo "Visit http://localhost:8089 to view the Locust UI"
	locust -f tests/locustfile.py

# Clean up
.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "*.tfstate" -delete
	find . -type f -name "*.tfstate.backup" -delete
