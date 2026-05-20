# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`openllm-selector` is a Python tool for researchers to filter, compare, and select open Large Language Models (LLMs) based on characteristics like size, open weights, checkpoints, and modality. It is in early development (v0.0.1).

## Setup

```bash
pip install -e '.[dev]'
```

This installs the package in editable mode with dev dependencies: `pytest`, `streamlit`, `plotly`, `requests`.

## Common Commands

```bash
# Run tests
pytest

# Run a single test file
pytest tests/test_example.py

# Run a single test
pytest tests/test_example.py::test_function_name
```

## Architecture

The package source lives in `src/openllm_selector/` (src-layout). The `__init__.py` re-exports everything from submodules via `from .example import *`.

Planned features include:
- A curated LLM database with filtering/comparison
- Access to foundational and recent papers per model
- Visualization of model clusters (likely via Streamlit + Plotly)

The `dev` extras (`streamlit`, `plotly`, `requests`) suggest the intended UI will be a Streamlit app consuming data via HTTP requests.

## Build

```bash
pip install hatchling
python -m hatchling build
```

Uses `hatchling` as the build backend. The wheel target is configured to package `openllm_selector` from the `src/` directory.
