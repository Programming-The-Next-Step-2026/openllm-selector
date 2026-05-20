"""Caching wrappers and filter/search composition for the Streamlit app.

All database calls that are safe to cache go through this module so that
component modules never hit disk or the network on every rerender.
"""

import streamlit as st

from openllm_selector import (
    fetch_recent_papers,
    filter_models,
    get_families,
    get_organizations,
    load_models,
    rank_by_openness,
)


@st.cache_data
def cached_load_models() -> list[dict]:
    return load_models()


@st.cache_data
def cached_get_families() -> list[str]:
    return get_families()


@st.cache_data
def cached_get_organizations() -> list[str]:
    return get_organizations()


@st.cache_data(ttl=3600)
def cached_fetch_recent_papers(model_name: str, max_results: int = 3) -> list[dict]:
    return fetch_recent_papers(model_name, max_results)


def get_filtered_models(
    filter_args: dict,
    multiselect_filters: dict,
    query: str,
) -> list[dict]:
    """Apply all active filters in order, then rank by openness.

    Parameters
    ----------
    filter_args:
        Keyword arguments unpacked directly into filter_models(). Only keys
        that differ from their default (i.e. the user actively set them)
        should be present — the sidebar is responsible for this invariant.
    multiselect_filters:
        Mapping of field name → list of selected values. Each field is
        applied with OR semantics: a model matches if its field value is in
        the list. An empty list means no restriction for that field.
        Expected keys: family, organization, architecture,
        country_of_origin, license.
    query:
        Free-text search string matched against name, family, and
        organization (case-insensitive substring). Empty string = no filter.

    Returns
    -------
    list[dict]
        Filtered models sorted by openness_score descending.
    """
    results = filter_models(**filter_args)

    for field, values in multiselect_filters.items():
        if values:
            results = [m for m in results if m[field] in values]

    if query.strip():
        q = query.lower()
        results = [
            m for m in results
            if q in m["name"].lower()
            or q in m["family"].lower()
            or q in m["organization"].lower()
        ]

    return rank_by_openness(results)
