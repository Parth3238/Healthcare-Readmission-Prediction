# Makefile for Healthcare Readmission Prediction Project

.PHONY: help install train evaluate run-web test docker-build docker-run clean

help:
	@echo "Healthcare Readmission Prediction - Available Commands"
	@echo "======================================================"
	@echo "make install       - Install all dependencies"
	@echo "make train         - Train the machine learning model"
	@echo "make evaluate      - Evaluate model performance"
	@echo "make compare       - Compare multiple ML algorithms"
	@echo "make run-web       - Start the Flask web application"
	@echo "make test          - Run unit tests"
	@echo "make notebook      - Start Jupyter notebook"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-run    - Run with Docker Compose"
	@echo "make clean         - Clean temporary files"

install:
	pip install -r requirements.txt

train:
	python src/train_model.py

evaluate:
	python src/evaluate.py

compare:
	python src/model_comparison.py

run-web:
	cd web && python app.py

test:
	python tests/test_models.py

notebook:
	jupyter notebook notebooks/

docker-build:
	docker build -t healthcare-readmission .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ 2>/dev/null || true

lint:
	flake8 src/ web/ tests/ --max-line-length=100

format:
	black src/ web/ tests/ --line-length=100
