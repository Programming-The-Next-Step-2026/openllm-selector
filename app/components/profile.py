"""Model profile card for the openllm-selector Streamlit app."""

import streamlit as st

from utils import cached_fetch_recent_papers, cached_load_models

_BOOL_FIELDS = [
    ("open_weights", "Open weights"),
    ("open_training_data", "Open training data"),
    ("intermediate_checkpoints", "Intermediate checkpoints"),
    ("open_code", "Open code"),
]

_PERMISSIVE_LICENSES = {"Apache 2.0", "MIT"}


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
    st.session_state.pop("selection_source", None)
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

        # Header: title and close button
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

        # Two-column layout: details left, key metrics right
        col_left, col_right = st.columns([2, 1])

        with col_left:
            langs = model.get("languages", [])
            st.markdown(
                f"**Family:** {model['family']}  \n"
                f"**Architecture:** {model['architecture']}  \n"
                f"**License:** {model['license']}  \n"
                f"**Languages:** {', '.join(langs)}"
            )
            st.markdown(
                f"[Foundational paper ↗]({model['foundational_paper']})  ·  "
                f"[HuggingFace ↗](https://huggingface.co/{model['huggingface_id']})"
            )

        with col_right:
            st.metric("Size", f"{model['size_b']} B")
            st.metric("Context window", f"{model['context_window']:,} tokens")
            tokens = model["training_tokens_b"]
            st.metric(
                "Training tokens",
                f"{tokens:,.0f} B" if tokens is not None else "Undisclosed",
            )
            if model.get("notes"):
                st.caption(model["notes"])
            st.markdown(
                f"{'✅' if model['has_instruct_version'] else '❌'} Instruct version available"
            )
            if model["model_type"] == "instruct":
                st.caption("This model is itself an instruct model.")
            st.markdown(
                f"{'✅' if model['has_think_version'] else '❌'} Think version available"
            )
            if model["model_type"] == "reasoning":
                st.caption("This model is itself a reasoning model.")

        # Openness badge row
        st.divider()
        st.caption(f"Openness score: {model['openness_score']} / 5")
        permissive = model["license"] in _PERMISSIVE_LICENSES
        badges = _BOOL_FIELDS + [("_permissive_license", "Permissive license")]
        badge_cols = st.columns(len(badges))
        for col, (field, label) in zip(badge_cols, badges):
            if field == "_permissive_license":
                icon = "✅" if permissive else "❌"
            else:
                icon = "✅" if model[field] else "❌"
            col.markdown(f"{icon} {label}")

        st.divider()
        with st.expander("Recent arXiv papers", expanded=True):
            with st.spinner("Fetching papers…"):
                try:
                    papers = cached_fetch_recent_papers(model["name"], max_results=3)
                except Exception as exc:
                    import requests as _req
                    if isinstance(exc, _req.exceptions.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                        st.warning("429 Too Many Requests — arXiv rate limit reached, please wait a few minutes and try again.")
                    else:
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
