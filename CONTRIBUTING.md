# Contributing to SparseAttn

Thank you for your interest in contributing to SparseAttn! This document provides guidelines and best practices for contributors.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Documentation](#documentation)
6. [Pull Request Process](#pull-request-process)
7. [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix

```bash
git clone https://github.com/your-username/SparseAttn.git
cd SparseAttn
git checkout -b feature/your-feature-name
```

## Development Setup

1. Install the package in development mode:

```bash
pip install -e .
```

2. Install development dependencies:

```bash
pip install -r requirements.txt
```

## Code Style

We follow the PEP 8 style guide for Python code. Additionally:

1. Use meaningful variable and function names
2. Write docstrings for all public functions and classes
3. Keep functions focused and small
4. Use type hints where possible

Code formatting is handled by Black. To format your code:

```bash
make format
```

## Testing

We use pytest for testing. All new functionality should include appropriate tests.

1. Run all tests:

```bash
make test
```

2. Run specific test files:

```bash
python -m pytest tests/test_sparseattn.py -v
```

3. Write tests for new functionality in the `tests/` directory
4. Ensure all tests pass before submitting a pull request

## Documentation

1. Update the README.md if you're adding new features or changing functionality
2. Add docstrings to all new functions and classes
3. Update examples in the `examples/` directory if needed
4. Add or update documentation in the `docs/` directory

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Run tests to ensure nothing is broken
3. Add tests for new functionality
4. Update documentation as needed
5. Submit a pull request with a clear title and description

### Pull Request Guidelines

- Keep pull requests focused on a single feature or bug fix
- Write a clear description of the changes
- Reference any related issues
- Ensure all CI checks pass

## Reporting Issues

When reporting issues, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected vs. actual behavior
4. System information (OS, Python version, CUDA version)
5. Any relevant error messages or logs

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.