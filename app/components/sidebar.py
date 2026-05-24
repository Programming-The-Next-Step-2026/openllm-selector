"""Filter sidebar for the openllm-selector Streamlit app."""

import streamlit as st

from utils import cached_get_families, cached_get_languages, cached_get_organizations

# Hard-coded controlled vocabularies (derived from models.json; update if DB grows).
_ARCHITECTURES = ["decoder-only", "encoder-decoder", "mixture-of-experts"]
_COUNTRIES = ["China", "France", "United Arab Emirates", "United States"]
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

# Context window filter uses discrete buckets rather than a linear slider
# because the range spans two orders of magnitude (2 K – 131 K).
_CTX_OPTIONS = [2_048, 4_096, 8_192, 32_768, 131_072]
_CTX_MIN = _CTX_OPTIONS[0]
_CTX_MAX = _CTX_OPTIONS[-1]

# Full-range sentinels used to detect whether the user touched a slider.
_SIZE_RANGE_DEFAULT = (2.0, 176.0)
_OPENNESS_RANGE_DEFAULT = (1, 5)
_YEAR_RANGE_DEFAULT = (2022, 2024)
_TRAINING_TOKENS_RANGE_DEFAULT = (300, 15000)
_NUM_LANGUAGES_RANGE_DEFAULT = (1, 46)

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
    "sb_openness_range",
    "sb_ctx_range",
    "sb_year_range",
    "sb_training_tokens_range",
    "sb_num_languages_range",
    "sb_has_instruct_version",
    "sb_language",
    "sb_excl_families",
    "sb_excl_orgs",
    "sb_excl_architectures",
    "sb_excl_countries",
    "sb_excl_licenses",
    "sb_excl_modalities",
]


def _multiselect_label(label: str, selected: list) -> str:
    if selected:
        return f"{label} ({len(selected)} selected)"
    return label


def render_sidebar() -> tuple[dict, dict, dict]:
    """Render all filter widgets and return the active filter state.

    Returns
    -------
    filter_args : dict
        Ready to unpack as ``filter_models(**filter_args)``. Only contains
        keys for filters the user has actively set (i.e. not at their default).
    multiselect_filters : dict
        Maps each categorical field to the list of selected values.
        Handled with OR logic in ``get_filtered_models()``.
        Keys: family, organization, architecture, country_of_origin, license.
        An empty list means no restriction for that field.
    exclude_filters : dict
        Maps each categorical field to the list of values to exclude.
        Applied as a post-filter in ``get_filtered_models()``.
        Keys: family, organization, architecture, country_of_origin, license,
        modality. An empty list means no exclusion for that field.
    """
    with st.sidebar:
        st.header("Filters")

        # ------------------------------------------------------------------ #
        # Categorical multiselects (OR logic)                                 #
        # ------------------------------------------------------------------ #
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
        multilingual = st.checkbox("Multilingual", key="sb_multilingual")
        has_instruct_version = st.checkbox("Instruct version available", key="sb_has_instruct_version")
        language = st.selectbox(
            "Language (official support only)",
            options=[""] + cached_get_languages(),
            format_func=lambda x: "All languages" if x == "" else x,
            key="sb_language",
        )

        # ------------------------------------------------------------------ #
        # Boolean checkboxes                                                  #
        # ------------------------------------------------------------------ #
        with st.expander("Openness filters", expanded=True):
            open_weights = st.checkbox("Open weights", key="sb_open_weights")
            open_training_data = st.checkbox("Open training data", key="sb_open_training_data")
            intermediate_checkpoints = st.checkbox(
                "Intermediate checkpoints", key="sb_intermediate_checkpoints"
            )
            open_code = st.checkbox("Open code", key="sb_open_code")
            permissive_license = st.checkbox("Permissive license (Apache 2.0 / MIT)", key="sb_permissive_license")

        # ------------------------------------------------------------------ #
        # Exclusion filters                                                    #
        # ------------------------------------------------------------------ #
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
            excl_architectures = st.multiselect(
                _multiselect_label("Exclude architecture", st.session_state.get("sb_excl_architectures", [])),
                options=_ARCHITECTURES,
                key="sb_excl_architectures",
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
            excl_modalities = st.multiselect(
                _multiselect_label("Exclude modality", st.session_state.get("sb_excl_modalities", [])),
                options=_MODALITIES,
                key="sb_excl_modalities",
            )

        # ------------------------------------------------------------------ #
        # Range sliders                                                        #
        # ------------------------------------------------------------------ #
        st.subheader("Ranges")

        size_range = st.slider(
            "Size (B parameters)",
            min_value=_SIZE_RANGE_DEFAULT[0],
            max_value=_SIZE_RANGE_DEFAULT[1],
            value=_SIZE_RANGE_DEFAULT,
            step=0.5,
            key="sb_size_range",
        )
        openness_range = st.slider(
            "Openness score",
            min_value=_OPENNESS_RANGE_DEFAULT[0],
            max_value=_OPENNESS_RANGE_DEFAULT[1],
            value=_OPENNESS_RANGE_DEFAULT,
            step=1,
            key="sb_openness_range",
        )
        ctx_range = st.select_slider(
            "Context window (tokens)",
            options=_CTX_OPTIONS,
            value=(_CTX_MIN, _CTX_MAX),
            format_func=lambda x: f"{x:,}",
            key="sb_ctx_range",
        )
        year_range = st.slider(
            "Release year",
            min_value=_YEAR_RANGE_DEFAULT[0],
            max_value=_YEAR_RANGE_DEFAULT[1],
            value=_YEAR_RANGE_DEFAULT,
            step=1,
            key="sb_year_range",
        )
        training_tokens_range = st.slider(
            "Training tokens (B)",
            min_value=_TRAINING_TOKENS_RANGE_DEFAULT[0],
            max_value=_TRAINING_TOKENS_RANGE_DEFAULT[1],
            value=_TRAINING_TOKENS_RANGE_DEFAULT,
            step=100,
            key="sb_training_tokens_range",
        )
        num_languages_range = st.slider(
            "Languages supported",
            min_value=_NUM_LANGUAGES_RANGE_DEFAULT[0],
            max_value=_NUM_LANGUAGES_RANGE_DEFAULT[1],
            value=_NUM_LANGUAGES_RANGE_DEFAULT,
            step=1,
            key="sb_num_languages_range",
        )

        # ------------------------------------------------------------------ #
        # Reset                                                                #
        # ------------------------------------------------------------------ #
        st.divider()
        if st.button("Reset all filters", width="stretch"):
            for key in _SIDEBAR_KEYS:
                st.session_state.pop(key, None)
            st.session_state.selected_model = None
            st.session_state.pop("grid", None)
            st.session_state.pop("scatter", None)
            st.rerun()

    # ---------------------------------------------------------------------- #
    # Build filter_args — only include keys for non-default values           #
    # ---------------------------------------------------------------------- #
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

    # Range sliders: only add when the user narrowed from the full range.
    if size_range != _SIZE_RANGE_DEFAULT:
        filter_args["min_size_b"] = size_range[0]
        filter_args["max_size_b"] = size_range[1]
    if openness_range != _OPENNESS_RANGE_DEFAULT:
        filter_args["min_openness"] = openness_range[0]
        filter_args["max_openness"] = openness_range[1]
    if ctx_range != (_CTX_MIN, _CTX_MAX):
        filter_args["min_context_window"] = ctx_range[0]
        filter_args["max_context_window"] = ctx_range[1]
    if year_range != _YEAR_RANGE_DEFAULT:
        filter_args["min_release_year"] = year_range[0]
        filter_args["max_release_year"] = year_range[1]
    if training_tokens_range != _TRAINING_TOKENS_RANGE_DEFAULT:
        filter_args["min_training_tokens_b"] = float(training_tokens_range[0])
        filter_args["max_training_tokens_b"] = float(training_tokens_range[1])
    if num_languages_range != _NUM_LANGUAGES_RANGE_DEFAULT:
        filter_args["min_num_languages"] = num_languages_range[0]
        filter_args["max_num_languages"] = num_languages_range[1]

    multiselect_filters = {
        "family": families,
        "organization": orgs,
        "architecture": architectures,
        "country_of_origin": countries,
        "license": licenses,
    }

    exclude_filters = {
        "family": excl_families,
        "organization": excl_orgs,
        "architecture": excl_architectures,
        "country_of_origin": excl_countries,
        "license": excl_licenses,
        "modality": excl_modalities,
    }

    return filter_args, multiselect_filters, exclude_filters
