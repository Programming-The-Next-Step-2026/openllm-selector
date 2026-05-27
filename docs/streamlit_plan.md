# Streamlit app — implementation plan

## Overview

A single-page Streamlit app that exposes the full `openllm_selector` API through an interactive UI. The target user is a researcher who wants to narrow down 18 models to the one or two that fit their study, without writing Python. The app has five logical sections that work together: search bar, filter sidebar, scatter plot, results grid, and model profile card.

---

## File layout

```
app/
  app.py              # entry point — layout, state wiring, section calls
  components/
    sidebar.py        # all filter widgets, returns a dict of filter args
    scatter.py        # Plotly scatter, returns selected model name or None
    grid.py           # ranked results table, returns selected model name or None
    profile.py        # model detail card + arXiv papers
  utils.py            # caching wrappers, search+filter composition
```

Run with: `streamlit run app/app.py`

Add `streamlit` and `plotly` to `[project.optional-dependencies] dev` in `pyproject.toml` (already present).

---

## State management

Use `st.session_state` for anything that persists across reruns. Define exactly two keys:

| Key | Type | Purpose |
|---|---|---|
| `selected_model` | `str \| None` | Name of the model whose profile card is open |
| `search_query` | `str` | Current value of the search bar |

Initialize both at the top of `app.py` before any widget is rendered:

```python
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
```

---

## Caching

Wrap all calls to the package in `utils.py` with `@st.cache_data`:

```python
@st.cache_data
def cached_load_models():
    return load_models()

@st.cache_data
def cached_get_families():
    return get_families()

@st.cache_data
def cached_get_organizations():
    return get_organizations()

@st.cache_data(ttl=3600)          # network call — refresh hourly
def cached_fetch_recent_papers(model_name, max_results=3):
    return fetch_recent_papers(model_name, max_results)
```

Never call `load_models()` directly from a component — always go through `utils.py` so rerenders don't hit disk.

---

## Filter + search composition

`filter_models()` and `search()` operate independently; they must be composed manually:

```python
def get_filtered_models(filter_args: dict, query: str) -> list[dict]:
    results = filter_models(**filter_args)
    if query.strip():
        q = query.lower()
        results = [
            m for m in results
            if q in m["name"].lower()
            or q in m["family"].lower()
            or q in m["organization"].lower()
        ]
    return rank_by_openness(results)
```

Apply `filter_models()` first (it narrows by field constraints), then apply the substring search as a post-filter on the already-reduced set, then rank. This keeps all three operations decoupled and testable.

---

## Section 1 — Search bar

**Widget:** `st.text_input` placed above the main content area, full width.

```python
query = st.text_input(
    "Search models",
    placeholder="e.g. OLMo, EleutherAI, Mistral …",
    key="search_query",
)
```

Using `key="search_query"` binds it directly to session state; no explicit assignment needed.

**Behaviour:** Any non-empty string filters the current result set to models where the query appears in name, family, or organization (case-insensitive substring). An empty string shows all models that pass the sidebar filters.

**Edge case:** If the search produces zero results, show `st.info("No models match your search.")` rather than an empty grid.

---

## Section 2 — Filter sidebar

All widgets live inside `with st.sidebar:`. The function `render_sidebar() -> dict` returns a dict ready to unpack into `filter_models(**filter_args)`.

### Dropdowns (multiselect — allow filtering to multiple values at once)

| Widget | Populated by | `filter_models` arg |
|---|---|---|
| Family | `get_families()` → 13 values | Pass to a manual post-filter (family is exact-match only; multiselect means OR across selections) |
| Organization | `get_organizations()` → 12 values | Same — post-filter OR logic |
| Architecture | Hard-coded: `decoder-only`, `encoder-decoder`, `mixture-of-experts` | Same |
| Country of origin | Hard-coded: `China`, `France`, `United Arab Emirates`, `United States` | Same |
| License | Hard-coded from DB: 7 distinct values | Same |

`filter_models()` only supports a single value per field. For multiselect with OR semantics, apply a post-filter after `filter_models()`:

```python
if selected_families:
    results = [m for m in results if m["family"] in selected_families]
```

Leave the selection empty (default) to mean "no restriction". Show a count chip next to each label when the selection is non-empty: `f"Family ({len(selected_families)} selected)"`.

Use `st.multiselect` for all five dropdowns. Do not pre-select any value by default.

### Checkboxes

Five boolean toggles, all default unchecked (= no filter applied). When checked, pass `True` to `filter_models()`. Do not expose a "False" filter in the UI — researchers generally want to include or ignore a criterion, not actively exclude models that have a feature.

```
□ Open weights
□ Open training data
□ Intermediate checkpoints
□ Open code
□ Multilingual
```

Group them under a `st.expander("Openness filters", expanded=True)`.

### Sliders

All sliders use `st.slider` with a range tuple `(min, max)`.

| Slider | Range | Step | `filter_models` args |
|---|---|---|---|
| Model size (B params) | 2.0 – 176.0 | 0.5 | `min_size_b`, `max_size_b` |
| Openness score | 1 – 5 | 1 | `min_openness`, `max_openness` |
| Context window (tokens) | 2 048 – 131 072 | 2 048 | `min_context_window`, `max_context_window` |
| Release year | 2022 – 2024 | 1 | `min_release_year`, `max_release_year` |

**Context window note:** The range spans two orders of magnitude (2 K – 131 K). A linear slider is misleading — a researcher dragging to "50 K" skips most of the interesting variation. Display the value with a thousands separator (`f"{v:,}"`) and add a brief label clarifying the unit. Consider bucketed `st.select_slider` as an alternative: `["2 048", "4 096", "8 192", "32 768", "131 072"]`.

**Only pass a slider value to `filter_models()` when it differs from the full range.** This avoids accidental over-filtering when the user hasn't touched a slider:

```python
if size_range != (2.0, 176.0):
    filter_args["min_size_b"] = size_range[0]
    filter_args["max_size_b"] = size_range[1]
```

### Reset button

Place a `st.button("Reset all filters")` at the bottom of the sidebar. On click, clear `st.session_state.selected_model` and rerun with default widget values by calling `st.rerun()`. Use `st.session_state` keys on every widget so their values can be reset programmatically.

---

## Section 3 — Scatter plot

**Library:** Plotly Express via `st.plotly_chart(fig, use_container_width=True, on_select="rerun")`.

The `on_select="rerun"` argument (Streamlit ≥ 1.33) causes a rerun when the user clicks a point, returning click data via `st.session_state[chart_key].selection`.

### Axis selectors

Two `st.selectbox` widgets placed in a two-column row above the chart:

```python
col1, col2 = st.columns(2)
x_axis = col1.selectbox("X axis", ["size_b", "context_window", "release_year", "openness_score"])
y_axis = col2.selectbox("Y axis", ["openness_score", "size_b", "context_window", "release_year"])
```

Map internal field names to display labels for axis titles:

```python
AXIS_LABELS = {
    "size_b": "Size (B parameters)",
    "context_window": "Context window (tokens)",
    "release_year": "Release year",
    "openness_score": "Openness score (1–5)",
}
```

### Plot construction

```python
fig = px.scatter(
    df,                                     # pd.DataFrame of filtered models
    x=x_axis,
    y=y_axis,
    color="openness_score",
    color_continuous_scale="Viridis",
    hover_name="name",
    hover_data=["organization", "license", "architecture", "multilingual"],
    size="size_b",                          # bubble size encodes parameter count
    size_max=40,
    labels=AXIS_LABELS,
)
```

Highlight the currently selected model (if any) by adding a second trace with a larger marker and contrasting border — do not rely on Plotly's built-in selection state, which resets on rerender.

### Click handling

```python
event = st.plotly_chart(fig, ..., on_select="rerun", key="scatter")
if event and event.selection and event.selection.points:
    clicked_name = event.selection.points[0]["hovertext"]  # hover_name maps here
    st.session_state.selected_model = clicked_name
```

---

## Section 4 — Results grid

Display the filtered + ranked list as an interactive `st.dataframe` with single-row selection enabled (Streamlit ≥ 1.35):

```python
selection = st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
    key="grid",
    column_config={
        "name": st.column_config.TextColumn("Model", width="medium"),
        "openness_score": st.column_config.NumberColumn("Openness", format="%d ⭐"),
        "size_b": st.column_config.NumberColumn("Size (B)", format="%.1f"),
        "context_window": st.column_config.NumberColumn("Context (tokens)", format="%d"),
        "release_year": st.column_config.NumberColumn("Year"),
        "multilingual": st.column_config.CheckboxColumn("Multilingual"),
        "architecture": st.column_config.TextColumn("Architecture"),
        "license": st.column_config.TextColumn("License"),
    },
)
```

Columns to display: `name`, `openness_score`, `size_b`, `context_window`, `release_year`, `multilingual`, `architecture`, `license`. Omit internal fields (`huggingface_id`, `foundational_paper`, boolean open_* flags) from the grid — they belong in the profile card.

On row selection:
```python
if selection.selection.rows:
    row_idx = selection.selection.rows[0]
    st.session_state.selected_model = display_df.iloc[row_idx]["name"]
```

Show a result count above the grid: `st.caption(f"{len(filtered)} of 18 models")`.

---

## Section 5 — Model profile card

Rendered in a `st.container()` below the grid, or in a `st.sidebar` expander — below the grid is preferred so it does not compete with the filter sidebar for space.

Only render if `st.session_state.selected_model is not None`.

### Layout

```python
model = get_model(st.session_state.selected_model)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(model["name"])
    st.caption(f"{model['organization']} · {model['country_of_origin']} · {model['release_year']}")
    st.markdown(f"**Family:** {model['family']}  \n**Architecture:** {model['architecture']}")
    st.markdown(f"**License:** {model['license']}")
    st.markdown(f"[Foundational paper ↗]({model['foundational_paper']})  ·  "
                f"[HuggingFace ↗](https://huggingface.co/{model['huggingface_id']})")

with col_right:
    st.metric("Openness score", f"{model['openness_score']} / 5")
    st.metric("Size", f"{model['size_b']} B")
    st.metric("Context window", f"{model['context_window']:,} tokens")
```

Below the two-column layout, render a `st.table` or a list of badge-style indicators for the boolean fields:

```
✅ Open weights    ✅ Open training data    ✅ Intermediate checkpoints
✅ Open code       ✅ Multilingual
```

Use `"✅"` / `"❌"` for visual clarity.

### Recent arXiv papers

Wrap the network call in an expander to avoid blocking the render:

```python
with st.expander("Recent arXiv papers", expanded=True):
    with st.spinner("Fetching papers…"):
        try:
            papers = cached_fetch_recent_papers(model["name"], max_results=3)
        except Exception:
            st.warning("Could not reach arXiv. Check your network connection.")
            papers = []

    if not papers:
        st.info("No recent papers found.")
    for p in papers:
        st.markdown(f"**[{p['title']}]({p['arxiv_url']})**")
        st.caption(f"{p['published'][:10]}  ·  {', '.join(p['authors'][:3])}")
        with st.expander("Abstract"):
            st.write(p["summary"])
```

### Close button

```python
if st.button("✕ Close", key="close_profile"):
    st.session_state.selected_model = None
    st.rerun()
```

---

## app.py wiring

```python
import streamlit as st
from utils import get_filtered_models
from components.sidebar import render_sidebar
from components.scatter import render_scatter
from components.grid import render_grid
from components.profile import render_profile

st.set_page_config(page_title="openllm-selector", layout="wide")
st.title("openllm-selector")
st.caption("Find the right open LLM for your research.")

# --- state init ---
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# --- search ---
query = st.text_input("Search models", key="search_query",
                      placeholder="e.g. OLMo, EleutherAI, Mistral …")

# --- sidebar → filter args ---
filter_args, multiselect_filters = render_sidebar()

# --- compose filters ---
filtered = get_filtered_models(filter_args, multiselect_filters, query)

# --- scatter + grid side by side ---
col_plot, col_grid = st.columns([1, 1])
with col_plot:
    render_scatter(filtered)
with col_grid:
    render_grid(filtered)

# --- profile card ---
if st.session_state.selected_model:
    st.divider()
    render_profile(st.session_state.selected_model)
```

---

## Implementation order

1. **`utils.py`** — caching wrappers and `get_filtered_models()`. No UI; fully testable.
2. **`components/sidebar.py`** — returns `(filter_args, multiselect_filters)`. Smoke-test by printing the dict.
3. **`components/grid.py`** — render the ranked `st.dataframe` and wire row selection to `session_state.selected_model`.
4. **`components/profile.py`** — static fields first, then add the arXiv expander once the card layout is confirmed.
5. **`components/scatter.py`** — Plotly scatter with axis selectors and click handling last, since it has the most Streamlit-version-specific behaviour.
6. **`app.py`** — assemble sections and verify the full interaction loop.

---

## Technical caveats

- **`on_select="rerun"` requires Streamlit ≥ 1.33** (scatter) **and ≥ 1.35** (dataframe). Pin the version in `pyproject.toml`: `streamlit >= 1.35`.
- **Scatter click data field name:** The `hovertext` key in `event.selection.points` corresponds to `hover_name`. Verify this against the installed Plotly/Streamlit version during development — the key has changed between minor releases.
- **BLOOM 176B** is an outlier at 176 B parameters and will visually dominate any linear size axis. Consider capping the bubble `size_max` and noting in the tooltip that bubble area is not to scale.
- **Context window log scale:** Plotly supports `log_x=True` / `log_y=True` in `px.scatter`. Enable this automatically when the context window axis is selected.
- **arXiv rate limiting:** `fetch_recent_papers()` makes a real HTTP request. The `ttl=3600` cache prevents repeated calls for the same model within a session, but rapid profile-card switching for different models can still trigger multiple requests. Add a brief `st.spinner` and catch `requests.exceptions.RequestException` gracefully.
- **No pandas dependency yet:** The `st.dataframe` call requires a DataFrame. Add `pandas` to the dev dependencies and convert the list of dicts with `pd.DataFrame(filtered)` inside `grid.py`. Keep `pandas` out of the main `dependencies` in `pyproject.toml` since the core library doesn't use it.
