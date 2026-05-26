"""Filter sidebar for the openllm-selector Streamlit app."""

import streamlit as st

from utils import (
    cached_get_families,
    cached_get_languages,
    cached_get_organizations,
    cached_load_models,
)

# Hard-coded controlled vocabularies (derived from models.json; update if DB grows).
_ARCHITECTURES = ["decoder-only", "encoder-decoder", "mixture-of-experts"]
_COUNTRIES = ["China", "France", "India", "Switzerland", "United Arab Emirates", "United States"]
_LICENSES = [
    "Apache 2.0",
    "BigScience RAIL",
    "DeepSeek License",
    "Gemma Terms of Use",
    "Llama 2 Community License",
    "Llama 3 Community License",
    "MIT",
]
_MODALITIES = ["image", "text"]
_MODEL_TYPES = ["base", "instruct", "reasoning"]

# All widget keys — enumerated here so the reset button can clear them
# without knowing implementation details of each widget.
_SIDEBAR_KEYS = [
    "sb_families",
    "sb_orgs",
    "sb_architectures",
    "sb_countries",
    "sb_licenses",
    "sb_open_weights",
    "sb_open_training_data",
    "sb_intermediate_checkpoints",
    "sb_open_code",
    "sb_permissive_license",
    "sb_multilingual",
    "sb_size_range",
    "sb_ctx_range",
    "sb_year_range",
    "sb_training_tokens_range",
    "sb_has_instruct_version",
    "sb_has_think_version",
    "sb_model_type",
    "sb_language",
    "sb_excl_families",
    "sb_excl_orgs",
    "sb_excl_countries",
    "sb_excl_licenses",
]


def _multiselect_label(label: str, selected: list) -> str:
    if selected:
        return f"{label} ({len(selected)} selected)"
    return label


def render_sidebar() -> tuple[dict, dict, dict]:
    """Render all filter widgets and return the active filter state.

    Slider bounds are computed dynamically from the current model database via
    ``cached_load_models()`` so they automatically cover any new models added
    to models.json without requiring code changes.

    Returns
    -------
    filter_args : dict
        Ready to unpack as ``filter_models(**filter_args)``. Only contains
        keys for filters the user has actively set (i.e. not at their default).
    multiselect_filters : dict
        Maps each categorical field to the list of selected values.
        Handled with OR logic in ``get_filtered_models()``.
        Keys: family, organization, architecture, country_of_origin, license,
        model_type. An empty list means no restriction for that field.
    exclude_filters : dict
        Maps each categorical field to the list of values to exclude.
        Applied as a post-filter in ``get_filtered_models()``.
        Keys: family, organization, country_of_origin, license.
        An empty list means no exclusion for that field.
    """
    # Compute slider bounds dynamically; cached_load_models() is free after first render.
    _models = cached_load_models()

    _size_min = float(min(m["size_b"] for m in _models))
    _size_max = float(max(m["size_b"] for m in _models))
    _size_default = (_size_min, _size_max)

    # Context window: unique discrete values used as select_slider options.
    _ctx_options = sorted({m["context_window"] for m in _models})
    _ctx_default = (_ctx_options[0], _ctx_options[-1])

    _year_min = int(min(m["release_year"] for m in _models))
    _year_max = int(max(m["release_year"] for m in _models))
    _year_default = (_year_min, _year_max)

    # Exclude models whose training token count was not publicly disclosed.
    _disclosed = [m["training_tokens_b"] for m in _models if m["training_tokens_b"] is not None]
    _tokens_min = int(min(_disclosed))
    _tokens_max = int(max(_disclosed))
    _tokens_default = (_tokens_min, _tokens_max)

    with st.sidebar:
        st.header("Filters")

        with st.expander("Openness filters", expanded=True):
            open_weights = st.checkbox("Open weights", key="sb_open_weights")
            open_training_data = st.checkbox("Open training data", key="sb_open_training_data")
            intermediate_checkpoints = st.checkbox(
                "Intermediate checkpoints", key="sb_intermediate_checkpoints"
            )
            open_code = st.checkbox("Open code", key="sb_open_code")
            permissive_license = st.checkbox("Permissive license (Apache 2.0 / MIT)", key="sb_permissive_license")

        st.subheader("Model characteristics")

        families = st.multiselect(
            _multiselect_label("Family", st.session_state.get("sb_families", [])),
            options=cached_get_families(),
            key="sb_families",
        )
        orgs = st.multiselect(
            _multiselect_label("Organization", st.session_state.get("sb_orgs", [])),
            options=cached_get_organizations(),
            key="sb_orgs",
        )
        architectures = st.multiselect(
            _multiselect_label("Architecture", st.session_state.get("sb_architectures", [])),
            options=_ARCHITECTURES,
            key="sb_architectures",
        )
        countries = st.multiselect(
            _multiselect_label("Country of origin", st.session_state.get("sb_countries", [])),
            options=_COUNTRIES,
            key="sb_countries",
        )
        licenses = st.multiselect(
            _multiselect_label("License", st.session_state.get("sb_licenses", [])),
            options=_LICENSES,
            key="sb_licenses",
        )
        model_types = st.multiselect(
            "Model type",
            options=_MODEL_TYPES,
            placeholder="Choose options",
            key="sb_model_type",
        )
        language = st.selectbox(
            "Language (official support only)",
            options=cached_get_languages(),
            index=None,
            placeholder="Choose options",
            key="sb_language",
        )
        has_instruct_version = st.checkbox("Instruct version available", key="sb_has_instruct_version")
        has_think_version = st.checkbox("Think version available", key="sb_has_think_version")
        multilingual = st.checkbox("Multilingual", key="sb_multilingual")

        with st.expander("Exclusion filters", expanded=False):
            excl_families = st.multiselect(
                _multiselect_label("Exclude family", st.session_state.get("sb_excl_families", [])),
                options=cached_get_families(),
                key="sb_excl_families",
            )
            excl_orgs = st.multiselect(
                _multiselect_label("Exclude organization", st.session_state.get("sb_excl_orgs", [])),
                options=cached_get_organizations(),
                key="sb_excl_orgs",
            )
            excl_countries = st.multiselect(
                _multiselect_label("Exclude country of origin", st.session_state.get("sb_excl_countries", [])),
                options=_COUNTRIES,
                key="sb_excl_countries",
            )
            excl_licenses = st.multiselect(
                _multiselect_label("Exclude license", st.session_state.get("sb_excl_licenses", [])),
                options=_LICENSES,
                key="sb_excl_licenses",
            )

        st.subheader("Ranges")

        size_range = st.slider(
            "Size (B parameters)",
            min_value=_size_min,
            max_value=_size_max,
            value=_size_default,
            step=0.5,
            key="sb_size_range",
        )
        ctx_range = st.select_slider(
            "Context window (tokens)",
            options=_ctx_options,
            value=_ctx_default,
            format_func=lambda x: f"{x:,}",
            key="sb_ctx_range",
        )
        year_range = st.slider(
            "Release year",
            min_value=_year_min,
            max_value=_year_max,
            value=_year_default,
            step=1,
            key="sb_year_range",
        )
        training_tokens_range = st.slider(
            "Training tokens (B)",
            min_value=_tokens_min,
            max_value=_tokens_max,
            value=_tokens_default,
            step=100,
            key="sb_training_tokens_range",
        )

        st.divider()
        if st.button("Reset all filters", width="stretch"):
            for key in _SIDEBAR_KEYS:
                st.session_state.pop(key, None)
            st.session_state.selected_model = None
            st.session_state.pop("grid", None)
            st.session_state.pop("scatter", None)
            st.rerun()

    # Build filter_args — only include keys the user actively set.
    filter_args: dict = {}

    # Boolean flags: only add when checked (True). Never add False — that
    # would exclude models that simply don't meet the criterion rather than
    # leaving them visible.
    if open_weights:
        filter_args["open_weights"] = True
    if open_training_data:
        filter_args["open_training_data"] = True
    if intermediate_checkpoints:
        filter_args["intermediate_checkpoints"] = True
    if open_code:
        filter_args["open_code"] = True
    if permissive_license:
        filter_args["permissive_license"] = True
    if has_instruct_version:
        filter_args["has_instruct_version"] = True
    if multilingual:
        filter_args["multilingual"] = True
    if language:
        filter_args["language"] = language
    if has_think_version:
        filter_args["has_think_version"] = True

    # Range sliders: only add when the user narrowed from the full range.
    if size_range != _size_default:
        filter_args["min_size_b"] = size_range[0]
        filter_args["max_size_b"] = size_range[1]
    if ctx_range != _ctx_default:
        filter_args["min_context_window"] = ctx_range[0]
        filter_args["max_context_window"] = ctx_range[1]
    if year_range != _year_default:
        filter_args["min_release_year"] = year_range[0]
        filter_args["max_release_year"] = year_range[1]
    if training_tokens_range != _tokens_default:
        filter_args["min_training_tokens_b"] = float(training_tokens_range[0])
        filter_args["max_training_tokens_b"] = float(training_tokens_range[1])

    multiselect_filters = {
        "family": families,
        "organization": orgs,
        "architecture": architectures,
        "country_of_origin": countries,
        "license": licenses,
        "model_type": model_types,
    }

    exclude_filters = {
        "family": excl_families,
        "organization": excl_orgs,
        "country_of_origin": excl_countries,
        "license": excl_licenses,
    }

    return filter_args, multiselect_filters, exclude_filters
