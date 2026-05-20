"""openllm-selector Streamlit app — entry point."""

import streamlit as st

from components.grid import render_grid
from components.profile import render_profile
from components.scatter import render_scatter
from components.sidebar import render_sidebar
from utils import get_filtered_models

st.set_page_config(page_title="openllm-selector", layout="wide")
st.title("openllm-selector")
st.caption("Find the right open LLM for your research.")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

query = st.text_input(
    "Search models",
    key="search_query",
    placeholder="e.g. OLMo, EleutherAI, Mistral …",
)

filter_args, multiselect_filters = render_sidebar()
filtered = get_filtered_models(filter_args, multiselect_filters, query)

col_plot, col_grid = st.columns([1, 1])
with col_plot:
    render_scatter(filtered)
with col_grid:
    render_grid(filtered)

if st.session_state.selected_model:
    st.divider()
    render_profile(st.session_state.selected_model)
