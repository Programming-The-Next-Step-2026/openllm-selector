"""Scatter plot component for the openllm-selector Streamlit app."""

import numpy as np
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

# Default axes: training tokens (x) vs context window (y).
_X_DEFAULT_IDX = 2   # training_tokens_b
_Y_DEFAULT_IDX = 0   # context_window


def _build_figure(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    selected_name: str | None,
) -> go.Figure:
    """Construct the Plotly figure for the current filter state.

    Bubble area encodes model size (size_b). Log scale is applied
    automatically when ``context_window`` or ``training_tokens_b`` is
    selected — both fields span multiple orders of magnitude where a linear
    scale would compress most models into a narrow band.

    Bubble area encodes ``size_b`` capped at 100 B so that outliers such as
    BLOOM 176B and DeepSeek-R1 671B don't dwarf every other model. All models
    above 100 B display at the same maximum bubble size.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered model records (all fields present).
    x_axis, y_axis : str
        Column names for the axes.
    selected_name : str or None
        Name of the currently selected model, used to draw a highlight ring.
    """
    df = df.copy()
    df["size_display"] = df["size_b"].clip(upper=100)

    # Add ±0.15 jitter to release_year to separate overlapping bubbles.
    # A fixed seed ensures the offsets are stable across rerenders.
    rng = np.random.default_rng(42)
    x_col, y_col = x_axis, y_axis
    if x_axis == "release_year":
        df["_x_plot"] = df["release_year"] + rng.uniform(-0.15, 0.15, len(df))
        x_col = "_x_plot"
    if y_axis == "release_year":
        df["_y_plot"] = df["release_year"] + rng.uniform(-0.15, 0.15, len(df))
        y_col = "_y_plot"

    # Ensure jittered column names resolve to the same axis label.
    plot_labels = dict(_AXIS_LABELS)
    if x_col != x_axis:
        plot_labels[x_col] = _AXIS_LABELS[x_axis]
    if y_col != y_axis:
        plot_labels[y_col] = _AXIS_LABELS[y_axis]

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="openness_score",
        color_continuous_scale="Viridis",
        range_color=[1, 5],
        hover_name="name",
        custom_data=["name"],       # reliable access in click-event point dicts
        hover_data={
            "openness_score": False,
            "size_b": False,
            "size_display": False,
        },
        size="size_display",
        size_max=40,
        opacity=0.7,
        labels=plot_labels,
        # Log scale for fields that span multiple orders of magnitude;
        # linear scale would compress most models into a narrow band.
        log_x=(x_axis in ("context_window", "training_tokens_b", "num_languages")),
        log_y=(y_axis in ("context_window", "training_tokens_b", "num_languages")),
    )

    fig.update_traces(hovertemplate="%{hovertext}<extra></extra>")

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Openness",
            thickness=12,
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1", "2", "3", "4", "5"],
        ),
        uirevision="scatter",       # preserve zoom/pan when data updates
    )

    # Release year: dtick=1 prevents fractional ticks (e.g. 2022.5) on a
    # narrow integer range; rangemode guards against any incidental negative
    # padding Plotly might apply during bubble layout.
    if x_axis == "release_year":
        fig.update_xaxes(dtick=1, rangemode="nonnegative")
    if y_axis == "release_year":
        fig.update_yaxes(dtick=1, rangemode="nonnegative")

    # Suppress sub-1 ticks on the log-scale num_languages axis.
    # range=[0, None] sets the minimum to 10^0=1; rangemode and explicit
    # tickvals together ensure no 0.1/0.5 labels appear despite Plotly padding.
    _lang_tick_vals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    if x_axis == "num_languages":
        fig.update_xaxes(rangemode="nonnegative", range=[0, None], tickvals=_lang_tick_vals)
    if y_axis == "num_languages":
        fig.update_yaxes(rangemode="nonnegative", range=[0, None], tickvals=_lang_tick_vals)

    # Draw a highlight ring at the selected model's position.
    # Using a separate go.Scatter trace instead of Plotly's built-in selection
    # state, which resets on every rerender.
    if selected_name:
        sel = df[df["name"] == selected_name]
        if not sel.empty:
            fig.add_trace(
                go.Scatter(
                    x=sel[x_col],
                    y=sel[y_col],
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

    note = "Bubble area encodes model size; models over 100 B are capped. Use fullscreen for detail."
    if x_axis == "training_tokens_b" or y_axis == "training_tokens_b":
        note += " Models with undisclosed training token counts are hidden."
    st.caption(note)
