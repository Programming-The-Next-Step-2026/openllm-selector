"""Scatter plot component for the openllm-selector Streamlit app."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_AXIS_OPTIONS = ["context_window", "release_year", "training_tokens_b", "num_languages"]

_AXIS_LABELS = {
    "context_window": "Context window (tokens)",
    "release_year": "Release year",
    "training_tokens_b": "Training tokens (B)",
    "num_languages": "Languages supported",
}

# Default axes: context window (x) vs training tokens (y).
_X_DEFAULT_IDX = 0   # context_window
_Y_DEFAULT_IDX = 2   # training_tokens_b


def _build_figure(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    selected_name: str | None,
) -> go.Figure:
    """Construct the Plotly figure for the current filter state.

    Bubble area encodes model size (size_b). Log scale is applied
    automatically when the context_window axis is selected, because that
    field spans two orders of magnitude (2 K – 131 K tokens).

    BLOOM 176B is a size outlier (~88× the smallest model); size_max caps
    the largest bubble at 40 px radius so smaller models remain visible.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered model records (all fields present).
    x_axis, y_axis : str
        Column names for the axes.
    selected_name : str or None
        Name of the currently selected model, used to draw a highlight ring.
    """
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color="openness_score",
        color_continuous_scale="Viridis",
        range_color=[1, 5],
        hover_name="name",
        custom_data=["name"],       # reliable access in click-event point dicts
        hover_data={
            "openness_score": False,
            "size_b": False,
        },
        size="size_b",
        size_max=40,
        opacity=0.7,
        labels=_AXIS_LABELS,
        # Log scale for context_window: linear scale would compress 2 K–32 K
        # into a tiny band while 131 K dominates.
        log_x=(x_axis == "context_window"),
        log_y=(y_axis == "context_window"),
    )

    fig.update_traces(hovertemplate="%{hovertext}<extra></extra>")

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Openness", thickness=12),
        uirevision="scatter",       # preserve zoom/pan when data updates
    )

    # Release year is an integer field; dtick=1 prevents Plotly from inserting
    # fractional ticks (2022.5 etc.) when the range is narrow.
    if x_axis == "release_year":
        fig.update_xaxes(dtick=1)
    if y_axis == "release_year":
        fig.update_yaxes(dtick=1)

    # num_languages is an integer; tickformat="d" prevents decimal ticks
    # while letting Plotly pick sensible spacing across the 1–46 range.
    if x_axis == "num_languages":
        fig.update_xaxes(tickformat="d")
    if y_axis == "num_languages":
        fig.update_yaxes(tickformat="d")

    # Draw a highlight ring at the selected model's position.
    # Using a separate go.Scatter trace instead of Plotly's built-in selection
    # state, which resets on every rerender.
    if selected_name:
        sel = df[df["name"] == selected_name]
        if not sel.empty:
            fig.add_trace(
                go.Scatter(
                    x=sel[x_axis],
                    y=sel[y_axis],
                    mode="markers",
                    marker=dict(
                        size=22,
                        color="rgba(0,0,0,0)",          # transparent fill
                        line=dict(color="white", width=3),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    return fig


def render_scatter(filtered: list[dict]) -> None:
    """Render the axis selectors and interactive scatter plot.

    Clicking a point writes the model name to
    ``st.session_state.selected_model`` to open the profile card.

    Parameters
    ----------
    filtered:
        Pre-filtered and ranked list of model dicts from get_filtered_models().
    """
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox(
        "X axis",
        _AXIS_OPTIONS,
        index=_X_DEFAULT_IDX,
        format_func=_AXIS_LABELS.get,
        key="scatter_x",
    )
    y_axis = col2.selectbox(
        "Y axis",
        _AXIS_OPTIONS,
        index=_Y_DEFAULT_IDX,
        format_func=_AXIS_LABELS.get,
        key="scatter_y",
    )

    if not filtered:
        st.info("No models to display.")
        return

    df = pd.DataFrame(filtered)
    selected = st.session_state.get("selected_model")
    fig = _build_figure(df, x_axis, y_axis, selected)

    event = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        key="scatter",
    )

    # Extract the clicked model name from the event.
    # customdata[0] is the first element of the custom_data=["name"] list —
    # more reliable than "hovertext", which has changed key names across
    # Plotly/Streamlit minor versions.
    if event and event.selection and event.selection.points:
        point = event.selection.points[0]
        name = (point.get("customdata") or [None])[0] or point.get("hovertext")
        if name:
            st.session_state.selected_model = name

    note = "Bubble area encodes model size (B parameters). BLOOM 176B is capped at the maximum bubble size."
    if x_axis == "training_tokens_b" or y_axis == "training_tokens_b":
        note += " Models with undisclosed training token counts are hidden."
    st.caption(note)
