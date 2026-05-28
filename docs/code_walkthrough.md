# openllm-selector — Code Walkthrough

Quick reference for the demo. One section per feature: file + function, how it works, how to extend it.

---

## App entry point

**`app/app.py`**

The entire app is ~20 lines. On every rerun Streamlit calls this top to bottom: set page config → render sidebar (returns filter state) → compute filtered list → render scatter + grid side-by-side in two columns → if `st.session_state.selected_model` is set, render the profile card below.

**To extend:** add a new tab or view by wrapping `render_scatter`/`render_grid` in `st.tabs(...)`.

---

## Caching strategy

**`app/utils.py`**

Streamlit reruns the entire script from top to bottom on every user interaction — every checkbox tick, every slider move, every row click. Without caching, that would mean reading `models.json` from disk and rebuilding Python dicts on every interaction, and fetching arXiv over the network every time a profile card opens.

`utils.py` is the single place where all database and network calls are wrapped with `@st.cache_data`. No component module is allowed to call `load_models()` or `fetch_recent_papers()` directly.

**How `@st.cache_data` works.** Streamlit serialises the function's arguments into a cache key. On the first call with a given key it runs the function and stores the return value. On subsequent calls with the same key it returns the stored value immediately, skipping the function body. The cache lives in memory for the lifetime of the server process.

**Five cached wrappers and their TTLs:**

| Wrapper | TTL | Why |
|---|---|---|
| `cached_load_models()` | none (session lifetime) | `models.json` never changes while the app is running |
| `cached_get_families()` | none | derived from `models.json`; same rationale |
| `cached_get_organizations()` | none | same |
| `cached_get_languages()` | none | same |
| `cached_fetch_recent_papers(model_name, max_results)` | 3600 s (1 hour) | arXiv results change over time; TTL balances freshness against rate limits |

**Cache key details.** `cached_load_models()` takes no arguments so there is exactly one cache entry — the full model list — shared across every component that calls it. `cached_fetch_recent_papers` is keyed on `(model_name, max_results)`, so each model gets its own cache entry. Opening OLMo 2 7B and then Llama 3.1 8B costs two network calls; opening OLMo 2 7B a second time within an hour costs zero.

**`get_filtered_models()` is not cached.** Filtering is pure Python list comprehension over the already-cached model list — it completes in under a millisecond and its output depends on the current widget state, which changes on every rerun. Caching it would require a complex hashable key covering all active filters and would save no meaningful time.

**To extend:** if you add a new database call (e.g. fetching model cards from HuggingFace), add a new `@st.cache_data(ttl=…)` wrapper in `utils.py` and call only that wrapper from components. Never call the underlying function directly from a component.

---

## 1. Loading models

**`src/openllm_selector/database.py` → `load_models()`**

Opens `data/models.json` (41 records), iterates the list, and calls `compute_openness_score()` on each record, injecting the result as an `openness_score` key before returning. The score is never stored in JSON — always derived at load time.

`compute_openness_score()` sums five boolean criteria: `open_weights`, `open_training_data`, `intermediate_checkpoints`, `open_code`, and permissive license (`"Apache"` or `"MIT"` in the `license` string). Result is 0–5.

**To extend:** add a new scoring criterion by adding another `bool(model["some_field"])` term to the sum in `compute_openness_score()`, and add that field to every record in `models.json`.

In the app, `load_models()` is always called through `cached_load_models()` — see the caching strategy section above.

---

## 2. Filtering models

**`src/openllm_selector/database.py` → `filter_models()`**

A single function with ~30 keyword-only parameters, all optional. Calls `load_models()` internally, then walks every model through a cascade of `if … continue` guards — one guard per parameter. Only models that pass every active guard are appended to `results`. String comparisons are case-insensitive; multivalue fields like `languages` use `any(...)`.

The app never calls `filter_models()` directly. It goes through `app/utils.py → get_filtered_models()`, which handles three things that `filter_models()` can't do in one call: OR-logic multiselects (family, org, architecture, etc.), exclusion lists, and free-text search — all applied as list comprehensions on top of the `filter_models()` output.

**To extend:** add a new filterable field by adding a parameter to `filter_models()` (with the guard in the loop) and a corresponding widget in `render_sidebar()`. The sidebar returns the value in `filter_args` and `get_filtered_models()` unpacks it automatically.

---

## 3. Ranking by openness

**`src/openllm_selector/database.py` → `rank_by_openness()`**

`sorted(models, key=lambda m: m["openness_score"], reverse=descending)`. Called at the end of `get_filtered_models()` so the results grid always shows the most open models first by default.

**To extend:** add a secondary sort key (e.g. `training_tokens_b` as a tiebreaker) by changing the lambda: `key=lambda m: (m["openness_score"], m["training_tokens_b"] or 0)`.

---

## 4. Sidebar filters

**`app/components/sidebar.py` → `render_sidebar()`**

Calls `cached_load_models()` to derive slider bounds dynamically (min/max size, year, tokens; unique context window values for the `select_slider`). Renders five sections inside `st.sidebar`: openness checkboxes, model characteristic multiselects, exclusion filters expander, range sliders, and a reset button.

The reset button works by iterating `_SIDEBAR_KEYS` — a flat list of every widget key — and popping them from `st.session_state`, then calling `st.rerun()`.

Returns three dicts: `filter_args` (unpacked into `filter_models()`), `multiselect_filters` (OR logic, handled in `get_filtered_models()`), and `exclude_filters` (exclusion logic, also in `get_filtered_models()`). Only actively-set filters are included in `filter_args` — unchecked checkboxes are simply absent from the dict.

**To extend:** add a new widget, give it an `sb_` key, add that key to `_SIDEBAR_KEYS`, and include its value in whichever return dict fits (boolean → `filter_args`, categorical → `multiselect_filters` or `exclude_filters`).

---

## 5. Results grid

**`app/components/grid.py` → `render_grid()`**

Builds a `pd.DataFrame` from the filtered list, keeping only the eight display columns defined in `_GRID_COLUMNS`. Formats `training_tokens_b` as `"N/A"` when `None`. Renders a `st.dataframe` with `selection_mode="single-row"` and `on_select="rerun"`.

When a row is selected, writes the model name to `st.session_state.selected_model` and sets `st.session_state.selection_source = "grid"`. The guarded `else` branch only clears `selected_model` when the grid itself was the selection source — so clicking a scatter bubble (which sets `selection_source = "scatter"`) doesn't get clobbered on the next rerun when the grid has no highlighted row.

**To extend:** add a column by appending its name to `_GRID_COLUMNS` and adding a `st.column_config` entry in `_COLUMN_CONFIG`.

---

## 6. Scatter plot

**`app/components/scatter.py` → `render_scatter()` and `_build_figure()`**

`render_scatter()` renders two axis selectboxes, builds the figure, handles the click event, then renders the chart.

`_build_figure()` constructs a Plotly Express scatter. Bubble colour encodes `openness_score` (Viridis 1–5). Bubble area encodes `size_b` capped at 100 B so outliers like BLOOM 176B and DeepSeek-R1 don't dwarf everything. Log scale is applied when `context_window`, `training_tokens_b`, or `num_languages` is on an axis — these span multiple orders of magnitude. Release year gets ±0.15 jitter (seeded at 42) to separate overlapping bubbles. A highlight ring is drawn as a separate `go.Scatter` trace for the selected model, since Plotly's built-in selection state resets on every rerun.

**Bubble click** works in two rerun cycles. `on_select="rerun"` triggers an immediate full rerun the moment a bubble is clicked, storing the selection in `st.session_state["scatter"]`. In the *next* rerun, `render_scatter()` reads `st.session_state.get("scatter")` *before* calling `st.plotly_chart()`, extracts the model name from `point["customdata"][0]` (set via `custom_data=["name"]`), writes it to `selected_model`, sets `selection_source = "scatter"`, and calls `st.rerun()` to open the profile card. The guard `name != st.session_state.get("selected_model")` prevents an infinite rerun loop.

**To extend:** add a new axis option by appending to `_AXIS_OPTIONS` and `_AXIS_LABELS`, and adding the field to the `log_x`/`log_y` condition if it spans orders of magnitude.

---

## 7. Model profile card

**`app/components/profile.py` → `render_profile()`**

Called from `app.py` when `st.session_state.selected_model` is set. Looks up the model from `cached_load_models()` (no disk hit), renders a bordered container with: header (name, org, country, year) + close button; left column (family, architecture, license, languages, paper and HuggingFace links); right column (size, context window, training tokens metrics, instruct/think availability); openness badge row (five ✅/❌ badges); and a "Recent arXiv papers" expander.

The close button calls `_close()`, which clears `selected_model`, pops the `grid`, `scatter`, and `selection_source` keys from session state, and reruns.

**To extend:** add a new field to the card by adding a `st.metric()` or `st.markdown()` line in `render_profile()`. No changes needed elsewhere.

---

## 8. arXiv paper fetch

**`src/openllm_selector/database.py` → `fetch_recent_papers()`**

Queries the arXiv Atom API at `http://arxiv.org/api/query` with `search_query=all:"<model_name>"`, sorted by submission date descending. Parses the Atom XML with `xml.etree.ElementTree` and returns a list of dicts with `title`, `authors`, `summary`, `published`, and `arxiv_url`.

In the app, all calls go through `app/utils.py → cached_fetch_recent_papers()`, which wraps `fetch_recent_papers()` in `@st.cache_data(ttl=3600)` — results are cached for one hour per model name, so switching between models in a session doesn't hammer the API. A 429 response is caught in `render_profile()` and surfaced as a user-visible warning rather than an uncaught exception.

**To extend:** change `max_results` (default 3) to show more papers, or switch `search_query` from `all:` to `ti:` to restrict to title-only matches (reduces false positives for short or common model names).

---

## File map

```
src/openllm_selector/
  database.py          load_models, filter_models, rank_by_openness,
                       fetch_recent_papers, compute_openness_score
  data/models.json     41 model records (no openness_score stored here)

app/
  app.py               entry point — wires sidebar → filter → scatter/grid → profile
  utils.py             @st.cache_data wrappers + get_filtered_models()
  components/
    sidebar.py         render_sidebar()  — all filter widgets
    grid.py            render_grid()     — sortable results table
    scatter.py         render_scatter(), _build_figure()
    profile.py         render_profile(), _close()
```
