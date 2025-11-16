# ECG Digitization Project Makefile

.PHONY: help install dev-install train inference test clean format lint docs setup all-stages

# Default target
help:
	@echo "ECG Digitization Project - Available Commands:"
	@echo ""
	@echo "  install      - Install package and dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  setup        - Setup environment and directories"
	@echo "  all-stages   - Train all stages sequentially"
	@echo "  stage0       - Train Stage 0 (Normalization)"
	@echo "  stage1       - Train Stage 1 (Grid Detection)"
	@echo "  stage2       - Train Stage 2 (Signal Digitization)"
	@echo "  inference    - Run inference on test data"
	@echo "  test         - Run test suite"
	@echo "  format       - Format code with black and isort"
	@echo "  lint         - Run code linting"
	@echo "  clean        - Clean build artifacts and cache"
	@echo "  docs         - Generate documentation"
	@echo "  sync-configs - Synchronize configuration files"

# Installation
install:
	pip install -r requirements.txt
	python -m pip install -e .

dev-install: install
	pip install pytest pytest-cov black isort mypy flake8 pre-commit

# Setup
setup:
	python main.py setup --config configs/base.yaml
	mkdir -p data/{train,val,test}/{images,annotations,series}
	mkdir -p outputs/{checkpoints,logs,predictions,visualizations}
	cp -n configs/base.yaml configs/local_config.yaml 2>/dev/null || true
	@echo "Environment setup completed!"

sync-configs:
	python scripts/sync_configs.py

# Training targets
all-stages:
	@echo "Training all stages..."
	$(MAKE) stage0
	$(MAKE) stage1
	$(MAKE) stage2
	@echo "All stages training completed!"

stage0:
	@echo "Training Stage 0..."
	python main.py train --config configs/stage0_config.yaml --mode stage0

stage1:
	@echo "Training Stage 1..."
	python main.py train --config configs/stage1_config.yaml --mode stage1

stage2:
	@echo "Training Stage 2..."
	python main.py train --config configs/stage2_config.yaml --mode stage2

resume-stage0:
	python main.py train --config configs/stage0_config.yaml --mode stage0 --resume outputs/checkpoints/stage0/latest.pth

resume-stage1:
	python main.py train --config configs/stage1_config.yaml --mode stage1 --resume outputs/checkpoints/stage1/latest.pth

resume-stage2:
	python main.py train --config configs/stage2_config.yaml --mode stage2 --resume outputs/checkpoints/stage2/latest.pth

# Inference targets
inference:
	python main.py inference --config configs/inference_config.yaml --mode pipeline

inference-single:
	@read -p "Enter path to ECG image: " img_path; \
	python main.py inference --config configs/inference_config.yaml --input "$$img_path"

inference-batch:
	@read -p "Enter path to directory: " dir_path; \
	python main.py inference --config configs/inference_config.yaml --input "$$dir_path" --output outputs/batch_results/

# Evaluation
evaluate:
	python main.py evaluate --config configs/inference_config.yaml

evaluate-stage0:
	python main.py evaluate --config configs/stage0_config.yaml --model outputs/checkpoints/stage0/best.pth

evaluate-stage1:
	python main.py evaluate --config configs/stage1_config.yaml --model outputs/checkpoints/stage1/best.pth

evaluate-stage2:
	python main.py evaluate --config configs/stage2_config.yaml --model outputs/checkpoints/stage2/best.pth

# Testing
test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -x -v

test-cov:
	python -m pytest tests/ --cov=src --cov-report=html:htmlcov --cov-report=term

test-specific:
	@read -p "Enter test file pattern (e.g., test_models.py): " pattern; \
	python -m pytest tests/"$$pattern" -v

# Code quality
format:
	black src/ tests/ data/ models/ utils/ engines/ main.py train.py inference.py
	isort src/ tests/ data/ models/ utils/ engines/ main.py train.py inference.py

format-check:
	black --check src/ tests/ data/ models/ utils/ engines/ main.py train.py inference.py
	isort --check-only src/ tests/ data/ models/ utils/ engines/ main.py train.py inference.py

lint:
	flake8 src/ tests/ data/ models/ utils/ engines/ main.py train.py inference.py
	mypy src/ models/ utils/ engines/

type-check:
	mypy src/ models/ utils/ engines/ --ignore-missing-imports

pre-commit: format lint test
	@echo "Pre-commit checks completed!"

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf outputs/logs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete

clean-all: clean
	rm -rf outputs/checkpoints/*
	rm -rf outputs/predictions/*
	rm -rf outputs/visualizations/*

# Documentation
docs:
	@echo "Documentation is available in the docs/ directory"
	@echo "Main documentation: README.md"
	@echo "Architecture: docs/architecture.md"
	@echo "API Reference: docs/api_reference.md"

# Data management
download-data:
	@echo "Data download functionality should be implemented here"
	@echo "This would download training/test data from specified source"

prepare-data:
	@echo "Data preparation functionality"
	python scripts/prepare_data.py

validate-data:
	python scripts/validate_data.py

# Performance profiling
profile:
	python -m cProfile -o profile_output.prof main.py inference --config configs/inference_config.yaml --input data/test/sample.png

memory-profile:
	python -m memory_profiler main.py inference --config configs/inference_config.yaml --input data/test/sample.png

# Development helpers
check: format lint test
	@echo "All checks passed!"

quick-test:
	python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

gpu-check:
	python -c "import torch; print('GPU count:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Environment management
create-env:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"

setup-precommit:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Docker targets (if applicable)
docker-build:
	docker build -t ecg-digitization .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs ecg-digitization

# Continuous integration helpers
ci-test: ci-lint ci-test-run

ci-lint:
	flake8 src/ tests/ data/ models/ utils/ engines/
	mypy src/ models/ utils/ engines/

ci-test-run:
	python -m pytest tests/ --cov=src --cov-report=xml

ci-build: ci-lint ci-test-run
	@echo "CI build completed successfully"

# Training automation
train-all-auto:
	@echo "Starting automated training pipeline..."
	$(MAKE) setup
	$(MAKE) validate-data
	$(MAKE) all-stages
	$(MAKE) evaluate
	@echo "Training pipeline completed!"

# Demo and examples
demo:
	@echo "Running demo..."
	python examples/demo_inference.py

demo-visualization:
	python examples/visualize_results.py

# Export and deployment
export-models:
	python scripts/export_models.py

create-package:
	python setup.py sdist bdist_wheel

# Git helpers
git-init:
	git init
	git add .
	git commit -m "Initial commit"

git-status:
	@echo "Git status:"
	@git status
	@echo ""
	@echo "Untracked files:"
	@git ls-files --others --exclude-standard

# Backup and restore
backup-configs:
	mkdir -p backups
	cp configs/*.yaml backups/configs_backup_$(shell date +%Y%m%d_%H%M%S)/

restore-configs:
	@read -p "Enter backup date (YYYYMMDD_HHMMSS): " backup_date; \
	cp -r backups/configs_backup_"$$backup_date"/configs/. configs/