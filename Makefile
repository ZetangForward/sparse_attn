# Makefile for SparseAttn

.PHONY: help install test clean format docs

help:
	@echo "SparseAttn Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install     Install the package"
	@echo "  make test        Run tests"
	@echo "  make format      Format code with black"
	@echo "  make clean       Clean build artifacts"

install:
	pip install -e .

test:
	python -m pytest tests/ -v

format:
	black sparseattn/
	black tests/
	black examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docs:
	@echo "Documentation is located in the docs/ directory"