# AGENTS.md

This file provides guidance to agentic coding tools like Claude Code, Codex, Copilot and OpenCode when working with code in this repository.

## Project

CBIR (Content-Based Image Retrieval) system for a master's thesis (TecGraf PUC / Embraer). Domain: vessel images from cameras monitoring Guanabara Bay. Goal: given an image of class A, verify it clusters near other class-A images in vector space. Dataset: ~4M images extracted from camera feeds (multi-distance, variable resolution), ~200K hand-labeled with class imbalance across vessel types. The system supports experimentation with different embedding models and similarity search algorithms, enabling heuristics for automatic labeling to augment the dataset and improve downstream segmentation models.

## Development Environment

- Python 3.13, managed via `.python-version`
- `uv` as package manager
- Run: `uv run main.py`
- Add dependencies: `uv add <package>`
- Lint: `ruff`
- Type check: `mypy`
- Test: `pytest`

## Architecture

- **Feature extraction**: pluggable models (ResNet, VGG, ViT, etc.) via PyTorch
- **Vector database**: Milvus for storing/searching embeddings
- **API**: FastAPI for programmatic access
- **CLI**: Typer for command-line interaction
- **Dashboard**: Streamlit for visualizing vector space and search results
- **Experiment tracking**: MLflow

## Deployment

Docker Compose with two profiles:

- **Experimentation**: all dev dependencies, MLflow tracking, CLI/API/dashboard
- **Execution**: minimal dependencies, API/CLI only
