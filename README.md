# openllm-selector

A tool to help researchers pick the right open LLM for their study.

Choosing the right open LLM for research is hard given the rapidly growing landscape of available models. Most comparison tools ask "which model scores highest on MMLU?" — that is not a useful question for research. What matters is: can I reproduce this model's training? Is the license compatible with my institution's data sharing agreement? Does it support the languages in my corpus? Will it fit on the GPUs I have access to?

`openllm-selector` is a curated database of 33 open LLMs with a queryable Python API and an interactive Streamlit app. Every record tracks the characteristics that actually drive research decisions rather than benchmark scores.

## Database fields

Each model record contains 25 fields:

| Field | Type | Description |
|---|---|---|
| `name`, `family`, `organization`, `country_of_origin` | str | Model identity |
| `release_year` | int | Year of public release |
| `size_b` | float | Model size in billions of parameters |
| `training_tokens_b` | float \| None | Pre-training token count in billions; `None` when undisclosed |
| `context_window` | int | Maximum context length in tokens |
| `modality` | list[str] | Supported modalities (`"text"` and/or `"image"`) |
| `architecture` | str | `decoder-only`, `encoder-decoder`, or `mixture-of-experts` |
| `license` | str | License name |
| `open_weights` | bool | Model weights are publicly available |
| `open_training_data` | bool | Training data is publicly available |
| `intermediate_checkpoints` | bool | Intermediate training checkpoints have been released |
| `open_code` | bool | Training code is publicly available |
| `multilingual` | bool | Officially supports more than one language |
| `num_languages` | int | Number of officially supported languages |
| `languages` | list[str] | Names of officially supported languages |
| `has_instruct_version` | bool | An instruction-tuned variant exists (or the model is itself instruction-tuned) |
| `model_type` | str | Model release type: `"base"`, `"instruct"`, or `"reasoning"` |
| `has_think_version` | bool | A chain-of-thought / think variant exists (or the model is itself a reasoning model) |
| `notes` | str *(optional)* | Additional context; present only for models where extra clarification is needed (e.g. post-trained models where `training_tokens_b` is null for structural reasons) |
| `foundational_paper` | str | URL of the foundational paper (arXiv for most models; non-arXiv for GPT-J 6B, Grok-1, Mixtral 8x22B, and Sarvam 30B) |
| `huggingface_id` | str | HuggingFace model identifier |
| `openness_score` | int | Computed 0–5 score: sum of `open_weights` + `open_training_data` + `intermediate_checkpoints` + `open_code` + permissive license (Apache 2.0 or MIT) |

Languages reflect officially supported languages as documented by the model creators, not partial or limited capabilities (e.g. Falcon supports German, Spanish and French officially, but has only limited capabilities in several other languages which are not included).

## Installation

```bash
pip install git+https://github.com/Programming-The-Next-Step-2026/openllm-selector.git@week-3
```

To run the interactive Streamlit app locally:

```bash
streamlit run app/app.py
```

## Python API

```python
import openllm_selector as o

# Filter by any combination of fields
candidates = o.filter_models(intermediate_checkpoints=True, max_size_b=10)
ranked = o.rank_by_openness(candidates)

# Look up a single model
model = o.get_model("OLMo 2 7B")

# Filter by officially supported language
hindi_models = o.filter_models(language="Hindi")

# Browse all supported languages
languages = o.get_languages()

# Filter by model type or think version availability
reasoning_models = o.filter_models(model_type="reasoning")
think_models = o.filter_models(has_think_version=True)

# Fetch recent arXiv papers mentioning a model
papers = o.fetch_recent_papers("OLMo", max_results=3)
```

## Documentation

See [docs/vignette.qmd](docs/vignette.qmd) for a full walkthrough covering both the Streamlit app and the Python API, with five realistic researcher scenarios.
