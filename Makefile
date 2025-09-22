# Mental Health Crisis Detection System - Makefile
# =============================================================================

.PHONY: help install dev test lint format clean build run deploy

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo "$(BLUE)Mental Health Crisis Detection System$(NC)"
	@echo "$(BLUE)=====================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# INSTALLATION
# =============================================================================
install: ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Development dependencies installed!$(NC)"

install-models: ## Download and install ML models
	@echo "$(BLUE)Downloading ML models...$(NC)"
	$(PYTHON) -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
	$(PYTHON) -m spacy download en_core_web_sm
	@echo "$(GREEN)Models downloaded successfully!$(NC)"

# =============================================================================
# DEVELOPMENT
# =============================================================================
dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up -d
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "$(YELLOW)API: http://localhost:8001$(NC)"
	@echo "$(YELLOW)Streamlit: http://localhost:8000$(NC)"

dev-stop: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	$(DOCKER_COMPOSE) --profile dev down
	@echo "$(GREEN)Development environment stopped!$(NC)"

dev-logs: ## Show development logs
	$(DOCKER_COMPOSE) --profile dev logs -f

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Tests completed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/unit/ -v
	@echo "$(GREEN)Unit tests completed!$(NC)"

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/integration/ -v
	@echo "$(GREEN)Integration tests completed!$(NC)"

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	$(PYTHON) -m pytest tests/e2e/ -v
	@echo "$(GREEN)End-to-end tests completed!$(NC)"

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/performance/ -v --benchmark-only
	@echo "$(GREEN)Performance tests completed!$(NC)"

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run linting
	@echo "$(BLUE)Running linting...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "$(GREEN)Linting completed!$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black . --check --diff
	isort . --check-only --diff
	@echo "$(GREEN)Code formatting check completed!$(NC)"

format-fix: ## Fix code formatting
	@echo "$(BLUE)Fixing code formatting...$(NC)"
	black .
	isort .
	@echo "$(GREEN)Code formatting fixed!$(NC)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy . --ignore-missing-imports
	@echo "$(GREEN)Type checking completed!$(NC)"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r . -f json -o bandit-report.json
	safety check --json --output safety-report.json
	@echo "$(GREEN)Security checks completed!$(NC)"

# =============================================================================
# DATA AND MODELS
# =============================================================================
generate-data: ## Generate synthetic training data
	@echo "$(BLUE)Generating synthetic data...$(NC)"
	$(PYTHON) train.py --generate_data
	@echo "$(GREEN)Synthetic data generated!$(NC)"

train: ## Train models
	@echo "$(BLUE)Training models...$(NC)"
	$(PYTHON) train.py --model_type ensemble --use_bert --use_spacy
	@echo "$(GREEN)Models trained successfully!$(NC)"

train-bert: ## Train BERT model
	@echo "$(BLUE)Training BERT model...$(NC)"
	$(PYTHON) train.py --model_type bert --use_bert
	@echo "$(GREEN)BERT model trained!$(NC)"

train-ensemble: ## Train ensemble model
	@echo "$(BLUE)Training ensemble model...$(NC)"
	$(PYTHON) train.py --model_type ensemble --use_bert --use_spacy
	@echo "$(GREEN)Ensemble model trained!$(NC)"

train-multimodal: ## Train multimodal model
	@echo "$(BLUE)Training multimodal model...$(NC)"
	$(PYTHON) train.py --multimodal --use_text --use_audio --use_image
	@echo "$(GREEN)Multimodal model trained!$(NC)"

# =============================================================================
# APPLICATION
# =============================================================================
run: ## Run the application locally
	@echo "$(BLUE)Starting application...$(NC)"
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
	@echo "$(GREEN)Application started!$(NC)"

run-streamlit: ## Run Streamlit app
	@echo "$(BLUE)Starting Streamlit app...$(NC)"
	streamlit run app.py
	@echo "$(GREEN)Streamlit app started!$(NC)"

run-api: ## Run API server only
	@echo "$(BLUE)Starting API server...$(NC)"
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
	@echo "$(GREEN)API server started!$(NC)"

# =============================================================================
# DOCKER
# =============================================================================
build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER) build -t mental-health-crisis-detector:latest .
	@echo "$(GREEN)Docker images built!$(NC)"

build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	$(DOCKER) build --target development -t mental-health-crisis-detector:dev .
	@echo "$(GREEN)Development Docker image built!$(NC)"

build-prod: ## Build production Docker image
	@echo "$(BLUE)Building production Docker image...$(NC)"
	$(DOCKER) build --target production -t mental-health-crisis-detector:prod .
	@echo "$(GREEN)Production Docker image built!$(NC)"

# =============================================================================
# DOCKER COMPOSE
# =============================================================================
up: ## Start all services with Docker Compose
	@echo "$(BLUE)Starting all services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)All services started!$(NC)"

up-dev: ## Start development services
	@echo "$(BLUE)Starting development services...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up -d
	@echo "$(GREEN)Development services started!$(NC)"

up-prod: ## Start production services
	@echo "$(BLUE)Starting production services...$(NC)"
	$(DOCKER_COMPOSE) --profile production up -d
	@echo "$(GREEN)Production services started!$(NC)"

up-monitoring: ## Start services with monitoring
	@echo "$(BLUE)Starting services with monitoring...$(NC)"
	$(DOCKER_COMPOSE) --profile production --profile monitoring up -d
	@echo "$(GREEN)Services with monitoring started!$(NC)"

down: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)All services stopped!$(NC)"

logs: ## Show logs for all services
	$(DOCKER_COMPOSE) logs -f

logs-api: ## Show API logs
	$(DOCKER_COMPOSE) logs -f app

logs-db: ## Show database logs
	$(DOCKER_COMPOSE) logs -f db

logs-redis: ## Show Redis logs
	$(DOCKER_COMPOSE) logs -f redis

# =============================================================================
# KUBERNETES
# =============================================================================
k8s-apply: ## Apply Kubernetes configurations
	@echo "$(BLUE)Applying Kubernetes configurations...$(NC)"
	$(KUBECTL) apply -f k8s/
	@echo "$(GREEN)Kubernetes configurations applied!$(NC)"

k8s-delete: ## Delete Kubernetes resources
	@echo "$(BLUE)Deleting Kubernetes resources...$(NC)"
	$(KUBECTL) delete -f k8s/
	@echo "$(GREEN)Kubernetes resources deleted!$(NC)"

k8s-status: ## Check Kubernetes status
	@echo "$(BLUE)Checking Kubernetes status...$(NC)"
	$(KUBECTL) get pods -n crisis-detector
	$(KUBECTL) get services -n crisis-detector
	$(KUBECTL) get ingress -n crisis-detector

k8s-logs: ## Show Kubernetes logs
	$(KUBECTL) logs -f deployment/crisis-detector-api -n crisis-detector

# =============================================================================
# DATABASE
# =============================================================================
db-init: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON) scripts/init_db.py
	@echo "$(GREEN)Database initialized!$(NC)"

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	$(PYTHON) scripts/migrate_db.py
	@echo "$(GREEN)Database migrations completed!$(NC)"

db-reset: ## Reset database
	@echo "$(BLUE)Resetting database...$(NC)"
	$(PYTHON) scripts/reset_db.py
	@echo "$(GREEN)Database reset!$(NC)"

# =============================================================================
# MONITORING
# =============================================================================
monitoring: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	$(DOCKER_COMPOSE) --profile monitoring up -d
	@echo "$(GREEN)Monitoring stack started!$(NC)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3000 (admin/admin)$(NC)"

monitoring-stop: ## Stop monitoring stack
	@echo "$(BLUE)Stopping monitoring stack...$(NC)"
	$(DOCKER_COMPOSE) --profile monitoring down
	@echo "$(GREEN)Monitoring stack stopped!$(NC)"

# =============================================================================
# DEPLOYMENT
# =============================================================================
deploy-staging: ## Deploy to staging
	@echo "$(BLUE)Deploying to staging...$(NC)"
	# Add your staging deployment commands here
	@echo "$(GREEN)Staging deployment completed!$(NC)"

deploy-prod: ## Deploy to production
	@echo "$(BLUE)Deploying to production...$(NC)"
	# Add your production deployment commands here
	@echo "$(GREEN)Production deployment completed!$(NC)"

# =============================================================================
# UTILITIES
# =============================================================================
clean: ## Clean up temporary files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .benchmarks/
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-docker: ## Clean up Docker resources
	@echo "$(BLUE)Cleaning up Docker resources...$(NC)"
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f
	@echo "$(GREEN)Docker cleanup completed!$(NC)"

clean-k8s: ## Clean up Kubernetes resources
	@echo "$(BLUE)Cleaning up Kubernetes resources...$(NC)"
	$(KUBECTL) delete namespace crisis-detector --ignore-not-found=true
	@echo "$(GREEN)Kubernetes cleanup completed!$(NC)"

# =============================================================================
# HEALTH CHECKS
# =============================================================================
health: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -f http://localhost:8000/health || echo "$(RED)Health check failed!$(NC)"
	@echo "$(GREEN)Health check completed!$(NC)"

health-detailed: ## Check detailed application health
	@echo "$(BLUE)Checking detailed application health...$(NC)"
	curl -f http://localhost:8000/health/detailed || echo "$(RED)Detailed health check failed!$(NC)"
	@echo "$(GREEN)Detailed health check completed!$(NC)"

# =============================================================================
# DOCUMENTATION
# =============================================================================
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	# Add documentation generation commands here
	@echo "$(GREEN)Documentation generated!$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	# Add documentation serving commands here
	@echo "$(GREEN)Documentation server started!$(NC)"

# =============================================================================
# BACKUP AND RESTORE
# =============================================================================
backup: ## Create backup
	@echo "$(BLUE)Creating backup...$(NC)"
	$(PYTHON) scripts/backup.py
	@echo "$(GREEN)Backup created!$(NC)"

restore: ## Restore from backup
	@echo "$(BLUE)Restoring from backup...$(NC)"
	$(PYTHON) scripts/restore.py
	@echo "$(GREEN)Restore completed!$(NC)"

# =============================================================================
# DEVELOPMENT WORKFLOW
# =============================================================================
dev-setup: install install-models generate-data train ## Complete development setup
	@echo "$(GREEN)Development setup completed!$(NC)"

ci: lint type-check security test ## Run CI pipeline
	@echo "$(GREEN)CI pipeline completed!$(NC)"

pre-commit: format-fix lint type-check test ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks completed!$(NC)"

# =============================================================================
# QUICK COMMANDS
# =============================================================================
quick-start: dev-setup dev ## Quick start for new developers
	@echo "$(GREEN)Quick start completed!$(NC)"

quick-test: format-fix lint test ## Quick test run
	@echo "$(GREEN)Quick test completed!$(NC)"

quick-deploy: build up-prod ## Quick deployment
	@echo "$(GREEN)Quick deployment completed!$(NC)"
