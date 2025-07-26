# Makefile for common tasks
install:
	pip install -r requirements.txt
lint:
	flake8 src tests
format:
	black src tests
precommit:
	pre-commit install
run:
	python src/train.py
test:
	pytest tests
