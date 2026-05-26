"""Results grid for the openllm-selector Streamlit app."""

import pandas as pd
import streamlit as st

from utils import cached_load_models

_GRID_COLUMNS = [
    "name",
    "openness_score",
    "size_b",
    "context_window",
    "training_tokens_b",
    "release_year",
    "architecture",
    "license",
]

_COLUMN_CONFIG = {
    "name": st.column_config.TextColumn("Model", width="medium"),
    "openness_score": st.column_config.NumberColumn("Openness", format="%d ⭐"),
    "size_b": st.column_config.NumberColumn("Size (B)", format="%.1f"),
    "context_window": st.column_config.NumberColumn("Context (tokens)", format="%d"),
    "training_tokens_b": st.column_config.TextColumn("Training tokens (B)"),
    "release_year": st.column_config.NumberColumn("Year"),
    "architecture": st.column_config.TextColumn("Architecture", width=150),
    "license": st.column_config.TextColumn("License"),
}


def render_grid(filtered: list[dict]) -> None:
    """Render the ranked model results table.

    Displays a count caption, an interactive single-row-selectable dataframe,
    and writes the selected model name to ``st.session_state.selected_model``
    so the profile card can pick it up.

    Parameters
    ----------
    filtered:
        Pre-filtered and ranked list of model dicts from get_filtered_models().
        May be empty, in which case an info message is shown instead of the table.
    """
    total = len(cached_load_models())
    st.caption(f"{len(filtered)} of {total} models")
    st.caption("Sorted by openness score by default. Click any column header to re-sort.")

    if not filtered:
        st.info("No models match your search.")
        # Close any open profile card whose model was just filtered out.
        if st.session_state.get("selected_model"):
            st.session_state.selected_model = None
            st.session_state.pop("grid", None)
        return

    # If the currently selected model is no longer in the filtered results,
    # clear it *before* rendering — so the profile card doesn't show a stale
    # record on this rerun, and the grid renders with no row pre-highlighted.
    selected = st.session_state.get("selected_model")
    filtered_names = {m["name"] for m in filtered}
    if selected and selected not in filtered_names:
        st.session_state.selected_model = None
        st.session_state.pop("grid", None)

    display_df = pd.DataFrame(filtered)[_GRID_COLUMNS].copy()
    display_df["training_tokens_b"] = display_df["training_tokens_b"].apply(
        lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.0f}"
    )

    selection = st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="grid",
        column_config=_COLUMN_CONFIG,
    )

    if selection.selection.rows:
        row_idx = selection.selection.rows[0]
        if row_idx < len(display_df):
            st.session_state.selected_model = display_df.iloc[row_idx]["name"]
    else:
        # Row was deselected — close the profile card.
        st.session_state.selected_model = None
