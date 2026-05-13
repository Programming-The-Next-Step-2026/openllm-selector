# openllm-selector

A tool to help researchers pick the right open LLM (Large Language Model) for their study.

## About

Choosing the right open LLM for research is hard given the rapidly growing landscape of available models. This package provides a curated database of 18 open LLMs with key characteristics — size, modality, license, openness score, training data availability, intermediate checkpoints, and links to foundational papers — so researchers can filter and compare models without wading through leaderboards and blog posts.

## Installation

```bash
pip install git+https://github.com/Programming-The-Next-Step-2026/openllm-selector.git@week-2
```

## Quick start

```python
import openllm_selector as o

# Load the full database
models = o.load_models()
print(f"{len(models)} models available")

# Find fully open models small enough to run locally
local_open = o.filter_models(max_size_b=10, min_openness=5)
for m in local_open:
    print(m["name"], m["huggingface_id"])
# OLMo 7B       allenai/OLMo-7B
# OLMo 2 7B     allenai/OLMo-2-1124-7B
# Pythia 6.9B   EleutherAI/pythia-6.9b

# Search by name, family, or organization
results = o.search("eleuther")
print([m["name"] for m in results])
# ['Pythia 6.9B', 'GPT-NeoX 20B']

# Rank everything from most to least open
ranked = o.rank_by_openness()
for m in ranked:
    print(f"{m['openness_score']}  {m['name']}")
```

## API reference

All functions are importable directly from `openllm_selector`.

### `load_models() -> list[dict]`

Returns all 18 model records from the database. Each record contains:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Model name, e.g. `"Llama 3.1 8B"` |
| `family` | `str` | Model family, e.g. `"LLaMA"` |
| `organization` | `str` | Releasing organization |
| `size_b` | `float` | Parameter count in billions |
| `context_window` | `int` | Maximum context length in tokens |
| `modality` | `list[str]` | Supported modalities, e.g. `["text"]` or `["text", "image"]` |
| `architecture` | `str` | Model architecture: `"decoder-only"`, `"encoder-only"`, `"encoder-decoder"`, or `"mixture-of-experts"` |
| `license` | `str` | License name |
| `open_weights` | `bool` | Weights publicly downloadable |
| `open_training_data` | `bool` | Training dataset publicly released |
| `intermediate_checkpoints` | `bool` | Mid-training checkpoints available |
| `open_code` | `bool` | Training code publicly available |
| `foundational_paper` | `str` | arXiv URL of the model's paper |
| `huggingface_id` | `str` | HuggingFace model identifier |
| `openness_score` | `int` | 1–5 composite openness rating (see below) |

**Openness score scale:**

| Score | Meaning |
|---|---|
| 5 | Weights + training data + intermediate checkpoints + code + permissive license |
| 4 | Mostly open — weights and training data/code available, one element missing or mildly restrictive license |
| 3 | Open weights with permissive license (Apache 2.0 / MIT), closed training details |
| 2 | Open weights with restricted or custom license |
| 1 | Heavily restricted — not currently assigned to any model in the database |

---

### `get_model(name: str) -> dict | None`

Look up a single model by name (case-insensitive, exact match). Returns `None` if the name is not found.

```python
model = o.get_model("pythia 6.9b")
print(model["foundational_paper"])
# https://arxiv.org/abs/2304.01373
```

---

### `filter_models(**kwargs) -> list[dict]`

Filter the database by any combination of criteria. All parameters are keyword-only and optional; omitting a parameter means it is not filtered on.

| Parameter | Type | Description |
|---|---|---|
| `modality` | `str` | Required modality, e.g. `"text"` or `"image"` (case-insensitive) |
| `min_size_b` | `float` | Minimum size in billions (inclusive) |
| `max_size_b` | `float` | Maximum size in billions (inclusive) |
| `min_openness` | `int` | Minimum openness score, default `1` |
| `max_openness` | `int` | Maximum openness score, default `5` |
| `open_weights` | `bool` | Match on `open_weights` field |
| `open_training_data` | `bool` | Match on `open_training_data` field |
| `intermediate_checkpoints` | `bool` | Match on `intermediate_checkpoints` field |
| `open_code` | `bool` | Match on `open_code` field |
| `organization` | `str` | Substring match on organization (case-insensitive) |
| `family` | `str` | Exact family name match (case-insensitive) |
| `license` | `str` | Substring match on license (case-insensitive) |
| `architecture` | `str` | Exact architecture match (case-insensitive): `"decoder-only"`, `"encoder-only"`, `"encoder-decoder"`, or `"mixture-of-experts"` |
| `min_context_window` | `int` | Minimum context window in tokens (inclusive) |
| `max_context_window` | `int` | Maximum context window in tokens (inclusive) |

```python
# Apache-licensed multimodal models under 10 B parameters
o.filter_models(modality="image", license="Apache", max_size_b=10)

# Models with intermediate checkpoints for studying training dynamics
o.filter_models(intermediate_checkpoints=True)

# Only mixture-of-experts models
o.filter_models(architecture="mixture-of-experts")

# Models that support long-context tasks (≥ 32 k tokens)
o.filter_models(min_context_window=32768)
```

---

### `search(query: str) -> list[dict]`

Free-text search against `name`, `family`, and `organization` (case-insensitive substring match).

```python
o.search("Google")   # returns Gemma 2B and Gemma 2 9B
o.search("mistral")  # returns Mistral 7B and Mixtral 8x7B
```

---

### `rank_by_openness(models=None, *, descending=True) -> list[dict]`

Sort models by `openness_score`. Pass a pre-filtered list to rank a subset, or omit it to rank the full database. Set `descending=False` to reverse the order.

```python
# Most open models first
o.rank_by_openness()

# Least open within the Mistral family
o.rank_by_openness(o.filter_models(family="Mistral"), descending=False)
```

---

### `get_families() -> list[str]`

Returns a sorted list of all unique model family names in the database (e.g. `"Falcon"`, `"LLaMA"`, `"OLMo"`).

---

### `get_organizations() -> list[str]`

Returns a sorted list of all unique organization names in the database (e.g. `"EleutherAI"`, `"Meta"`, `"Mistral AI"`).
