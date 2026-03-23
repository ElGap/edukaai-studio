.PHONY: help install install-dev test lint format clean build run docs

# Default target
help:
	@echo "EdukaAI Studio - Available commands:"
	@echo ""
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run all linters"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make build        - Build package for distribution"
	@echo "  make run          - Run the application"
	@echo "  make setup        - Setup pre-commit hooks"
	@echo "  make check        - Run all checks (lint + test)"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/edukaai_studio --cov-report=html --cov-report=term

# Linting and formatting
lint:
	flake8 src/edukaai_studio --max-line-length=100 --extend-ignore=E203,W503
	black --check src/edukaai_studio
	mypy src/edukaai_studio --ignore-missing-imports

format:
	black src/edukaai_studio
	isort src/edukaai_studio --profile=black

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m build
	 twine check dist/*

# Running
run:
	python run.py

# Development setup
setup: install-dev
	pre-commit install
	@echo "Pre-commit hooks installed!"

# All checks
check: lint test
	@echo "All checks passed!"

# Documentation
docs:
	@echo "Documentation generation not yet implemented"
	@echo "See docs/ directory for manual documentation"
