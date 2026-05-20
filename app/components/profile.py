"""Model profile card for the openllm-selector Streamlit app."""

import streamlit as st

from utils import cached_fetch_recent_papers, cached_load_models

_BOOL_FIELDS = [
    ("open_weights", "Open weights"),
    ("open_training_data", "Open training data"),
    ("intermediate_checkpoints", "Intermediate checkpoints"),
    ("open_code", "Open code"),
    ("multilingual", "Multilingual"),
]


def _lookup(name: str) -> dict | None:
    """Return the model record from the cache, or None if not found."""
    return next(
        (m for m in cached_load_models() if m["name"].lower() == name.lower()),
        None,
    )


def _close() -> None:
    st.session_state.selected_model = None
    st.session_state.pop("grid", None)
    st.session_state.pop("scatter", None)
    st.rerun()


def render_profile(model_name: str) -> None:
    """Render the full detail card for a single model.

    Reads ``model_name`` from the caller and looks it up in the cached
    database. Writes close-button state back to ``st.session_state``.

    Parameters
    ----------
    model_name:
        Exact model name as stored in the database (e.g. ``"OLMo 7B"``).
    """
    model = _lookup(model_name)

    if model is None:
        st.warning(f"Model '{model_name}' not found in database.")
        _close()
        return

    with st.container(border=True):

        # ------------------------------------------------------------------ #
        # Header row: title + close button                                    #
        # ------------------------------------------------------------------ #
        title_col, close_col = st.columns([6, 1])
        with title_col:
            st.subheader(model["name"])
            st.caption(
                f"{model['organization']} · "
                f"{model['country_of_origin']} · "
                f"{model['release_year']}"
            )
        with close_col:
            if st.button("✕ Close", key="close_profile", width="stretch"):
                _close()

        # ------------------------------------------------------------------ #
        # Main layout: details (left) + key metrics (right)                  #
        # ------------------------------------------------------------------ #
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown(
                f"**Family:** {model['family']}  \n"
                f"**Architecture:** {model['architecture']}  \n"
                f"**License:** {model['license']}"
            )
            st.markdown(
                f"[Foundational paper ↗]({model['foundational_paper']})  ·  "
                f"[HuggingFace ↗](https://huggingface.co/{model['huggingface_id']})"
            )

        with col_right:
            st.metric("Openness score", f"{model['openness_score']} / 5")
            st.metric("Size", f"{model['size_b']} B")
            st.metric("Context window", f"{model['context_window']:,} tokens")

        # ------------------------------------------------------------------ #
        # Boolean feature badges                                              #
        # ------------------------------------------------------------------ #
        st.divider()
        badge_cols = st.columns(len(_BOOL_FIELDS))
        for col, (field, label) in zip(badge_cols, _BOOL_FIELDS):
            icon = "✅" if model[field] else "❌"
            col.markdown(f"{icon} {label}")

        # ------------------------------------------------------------------ #
        # Recent arXiv papers                                                 #
        # ------------------------------------------------------------------ #
        st.divider()
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
                st.caption(
                    f"{p['published'][:10]}  ·  "
                    f"{', '.join(p['authors'][:3])}"
                )
                with st.expander("Abstract"):
                    st.write(p["summary"])
