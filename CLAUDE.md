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

## Rules

**Git:** Never merge branches or create pull requests. Commit and push only.

**openness_score:** This field is computed dynamically inside `load_models()` by `compute_openness_score()` and injected into each record at load time. It must never be stored in `models.json`.

**Model data:** All values in `models.json` were manually verified against primary sources (HuggingFace model pages and foundational papers). Any change to model data must be verified against the same primary sources before committing.

**App data access:** Model data must never be duplicated or hardcoded in app code. All model data must be read from the database through the caching wrappers in `app/utils.py` (e.g. `cached_load_models()`, `cached_get_families()`).

## AI Usage
Claude Code was used to assist with code generation, documentation, and unit tests.
Prompts followed a consistent pattern: specify the function name, its parameters,
expected behaviour, and edge cases to test. All generated code was reviewed and
tested before committing.