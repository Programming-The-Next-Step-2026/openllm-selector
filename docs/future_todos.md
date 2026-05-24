# Future TODOs and known issues

Outstanding issues and deferred features for `openllm-selector`, ordered by severity.

---

## High

### Database coverage is limited to 18 models

**Files:** `src/openllm_selector/data/models.json`

**Detail:** The database currently covers 18 models selected to represent the major open-weight families available as of early 2024. Many important newer models are missing, including OLMo 3 32B, DeepSeek-R1, Qwen2.5, Gemma 3/4, Llama 3.3, Mistral Small 3.1, and others. The filter and comparison logic becomes less useful as the database falls behind the current model landscape.

**Suggested fix:** Add new models as they achieve stable releases. Each addition requires manually verifying all fields against the HuggingFace model page and foundational paper per the data verification rule in CLAUDE.md.

---

## Medium

### `training_tokens_b` is null for Mistral 7B and Mixtral 8x7B

**Files:** `src/openllm_selector/data/models.json`

**Detail:** Mistral AI has not publicly disclosed the pre-training token counts for Mistral 7B v0.1 or Mixtral 8x7B. These fields are stored as `null` and are excluded from any filter that sets a `min_training_tokens_b` or `max_training_tokens_b` bound. This is a confirmed non-disclosure with no prospect of resolution unless Mistral AI publishes the data.

**Suggested fix:** No action possible until Mistral AI discloses the figures. When they do, update both values and cross-check against the release blog post or paper before committing.

---

### Model type (base vs. instruct) distinction not represented

**Files:** `src/openllm_selector/data/models.json`, `src/openllm_selector/database.py`, `app/components/sidebar.py`

**Detail:** `has_instruct_version` is a boolean that only indicates whether an instruct variant *exists*, not whether the record itself is a base or instruct model. Phi-3 Mini 4K and LLaVA 1.5 7B are instruction-tuned models released without a separate base variant, but this is currently surfaced only as a profile card caption rather than a queryable field. Researchers looking specifically for base models have no direct filter for this.

**Suggested fix:** Add a `model_type` field with values `"base"` or `"instruct"` to all records. Add a corresponding filter to `filter_models()` and a checkbox or radio button to the sidebar. Update tests.

---

### `has_think_version` deferred — no thinking models in the database

**Files:** `src/openllm_selector/data/models.json`, `src/openllm_selector/database.py`

**Detail:** A `has_think_version` boolean was considered during the new-fields research phase but not added because all 18 current models would have the value `False`, making the field meaningless as a filter. The relevant models (DeepSeek-R1, QwQ-32B, and similar reasoning-focused releases) are not yet in the database.

**Suggested fix:** Add at least one thinking/reasoning model to the database first (e.g. DeepSeek-R1). At that point, add `has_think_version` as a stored boolean field, update `filter_models()`, and add a sidebar checkbox. The field only becomes a useful discriminator once both values are represented.

---

### `num_languages` for Falcon 7B and Falcon 40B understates partial capabilities

**Files:** `src/openllm_selector/data/models.json`

**Detail:** `num_languages` is set to 4 for both Falcon models (English, German, Spanish, French — the four languages with full official support per the TII technical report). The HuggingFace model pages note limited capabilities in 7 additional languages (e.g. Portuguese, Italian, Dutch, Polish, Arabic, Chinese, Russian), which are intentionally excluded from the `languages` list and `num_languages` count to stay consistent with the database policy of tracking officially supported languages only.

**Suggested fix:** No change needed to the data. If a partial-capability tier is added in the future (e.g. a `languages_partial` list), Falcon 7B and 40B would be the primary candidates to populate it.

---

## Low

### Ecosystem support (Ollama/GGUF availability) not tracked

**Files:** `src/openllm_selector/data/models.json`, `src/openllm_selector/database.py`

**Detail:** Whether a model is available via Ollama or as a GGUF quantisation is practically relevant for researchers running models locally, but this information changes frequently as the community adds new quantisations and Ollama updates its library. Hardcoding snapshot values would require constant maintenance and would frequently be stale.

**Suggested fix:** Defer indefinitely, or implement as a live lookup against the Ollama API (`https://ollama.com/api/tags`) rather than a stored field, so it always reflects current availability. This would be a separate optional function alongside `fetch_recent_papers()`.
