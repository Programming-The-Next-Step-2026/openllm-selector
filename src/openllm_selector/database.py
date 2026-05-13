"""Functions to load, filter, and query the open LLM database."""

import json
import pathlib

_DATA_FILE = pathlib.Path(__file__).parent / "data" / "models.json"


def load_models() -> list[dict]:
    """Load the full model database from disk.

    Returns
    -------
    list[dict]
        All model records. Each record contains: name, family, organization,
        size_b, context_window, modality, architecture, license, open_weights,
        open_training_data, intermediate_checkpoints, open_code,
        foundational_paper, huggingface_id, and openness_score.
    """
    with _DATA_FILE.open() as fh:
        return json.load(fh)


def get_model(name: str) -> dict | None:
    """Retrieve a single model record by name (case-insensitive, exact match).

    Parameters
    ----------
    name : str
        The model name to look up, e.g. ``"OLMo 7B"``.

    Returns
    -------
    dict or None
        The model record if found, ``None`` otherwise.

    Examples
    --------
    >>> model = get_model("Pythia 6.9B")
    >>> model["openness_score"]
    5
    """
    target = name.lower()
    for model in load_models():
        if model["name"].lower() == target:
            return model
    return None


def filter_models(
    *,
    modality: str | None = None,
    min_size_b: float | None = None,
    max_size_b: float | None = None,
    min_openness: int = 1,
    max_openness: int = 5,
    open_weights: bool | None = None,
    open_training_data: bool | None = None,
    intermediate_checkpoints: bool | None = None,
    open_code: bool | None = None,
    organization: str | None = None,
    family: str | None = None,
    license: str | None = None,
    architecture: str | None = None,
    min_context_window: int | None = None,
    max_context_window: int | None = None,
) -> list[dict]:
    """Filter models by one or more criteria.

    All parameters are optional; omitting a parameter means that field is not
    filtered on. String comparisons are case-insensitive. For ``modality``, a
    model matches if the supplied value appears anywhere in its modality list.
    For ``organization`` and ``license``, substring matching is used. For
    ``family`` and ``architecture``, an exact match (case-insensitive) is required.

    Parameters
    ----------
    modality : str, optional
        Required modality string, e.g. ``"text"`` or ``"image"``.
    min_size_b : float, optional
        Minimum model size in billions of parameters (inclusive).
    max_size_b : float, optional
        Maximum model size in billions of parameters (inclusive).
    min_openness : int
        Minimum openness score on the 1–5 scale (inclusive). Default ``1``.
    max_openness : int
        Maximum openness score on the 1–5 scale (inclusive). Default ``5``.
    open_weights : bool, optional
        If provided, keep only models whose ``open_weights`` field matches.
    open_training_data : bool, optional
        If provided, keep only models whose ``open_training_data`` matches.
    intermediate_checkpoints : bool, optional
        If provided, keep only models whose ``intermediate_checkpoints`` matches.
    open_code : bool, optional
        If provided, keep only models whose ``open_code`` field matches.
    organization : str, optional
        Substring to match against ``organization`` (case-insensitive).
    family : str, optional
        Exact family name to match (case-insensitive), e.g. ``"LLaMA"``.
    license : str, optional
        Substring to match against ``license`` (case-insensitive),
        e.g. ``"Apache"`` to match ``"Apache 2.0"``.
    architecture : str, optional
        Exact architecture to match (case-insensitive). One of
        ``"decoder-only"``, ``"encoder-only"``, ``"encoder-decoder"``, or
        ``"mixture-of-experts"``.
    min_context_window : int, optional
        Minimum context window in tokens (inclusive).
    max_context_window : int, optional
        Maximum context window in tokens (inclusive).

    Returns
    -------
    list[dict]
        Models satisfying all supplied criteria.

    Examples
    --------
    >>> fully_open = filter_models(min_openness=5)
    >>> small_open = filter_models(max_size_b=8, open_training_data=True)
    >>> multimodal = filter_models(modality="image")
    >>> moe = filter_models(architecture="mixture-of-experts")
    >>> long_context = filter_models(min_context_window=32768)
    """
    results = []
    for m in load_models():
        if modality is not None:
            if modality.lower() not in [mod.lower() for mod in m["modality"]]:
                continue
        if min_size_b is not None and m["size_b"] < min_size_b:
            continue
        if max_size_b is not None and m["size_b"] > max_size_b:
            continue
        if not (min_openness <= m["openness_score"] <= max_openness):
            continue
        if open_weights is not None and m["open_weights"] != open_weights:
            continue
        if open_training_data is not None and m["open_training_data"] != open_training_data:
            continue
        if intermediate_checkpoints is not None and m["intermediate_checkpoints"] != intermediate_checkpoints:
            continue
        if open_code is not None and m["open_code"] != open_code:
            continue
        if organization is not None and organization.lower() not in m["organization"].lower():
            continue
        if family is not None and m["family"].lower() != family.lower():
            continue
        if license is not None and license.lower() not in m["license"].lower():
            continue
        if architecture is not None and m["architecture"].lower() != architecture.lower():
            continue
        if min_context_window is not None and m["context_window"] < min_context_window:
            continue
        if max_context_window is not None and m["context_window"] > max_context_window:
            continue
        results.append(m)
    return results


def get_families() -> list[str]:
    """Return all unique model family names in the database, sorted alphabetically.

    Returns
    -------
    list[str]
        Sorted list of family names, e.g. ``["Falcon", "Gemma", "LLaMA", ...]``.

    Examples
    --------
    >>> families = get_families()
    >>> "OLMo" in families
    True
    """
    return sorted({m["family"] for m in load_models()})


def get_organizations() -> list[str]:
    """Return all unique organization names in the database, sorted alphabetically.

    Returns
    -------
    list[str]
        Sorted list of organization names.

    Examples
    --------
    >>> orgs = get_organizations()
    >>> "Meta" in orgs
    True
    """
    return sorted({m["organization"] for m in load_models()})


def rank_by_openness(
    models: list[dict] | None = None,
    *,
    descending: bool = True,
) -> list[dict]:
    """Sort models by openness score.

    The openness score (1–5) reflects how openly the model was released:

    * **5** – Fully open: weights, training data, intermediate checkpoints,
      and training code are all publicly available under a permissive license.
    * **4** – Mostly open: weights plus training data and/or code, but
      missing one element or carrying a mildly restrictive license.
    * **3** – Open weights with a permissive license (e.g. Apache 2.0 / MIT),
      but closed training details.
    * **2** – Open weights with a restricted or custom license.
    * **1** – Heavily restricted access or weights-only release under a
      non-commercial or proprietary license.

    Parameters
    ----------
    models : list[dict], optional
        Subset of models to rank. If ``None``, the full database is ranked.
    descending : bool
        If ``True`` (default), the most open models appear first.

    Returns
    -------
    list[dict]
        Models sorted by ``openness_score``.

    Examples
    --------
    >>> top = rank_by_openness()[:3]
    >>> [m["name"] for m in top]  # all score 5
    ['OLMo 7B', 'OLMo 2 7B', 'Pythia 6.9B']
    """
    if models is None:
        models = load_models()
    return sorted(models, key=lambda m: m["openness_score"], reverse=descending)


def search(query: str) -> list[dict]:
    """Search models by a free-text query matched against name, family, and organization.

    The search is case-insensitive and returns any model where ``query`` is a
    substring of the ``name``, ``family``, or ``organization`` field.

    Parameters
    ----------
    query : str
        Search term, e.g. ``"llama"``, ``"Google"``, or ``"OLMo"``.

    Returns
    -------
    list[dict]
        Models whose name, family, or organization contains ``query``.

    Examples
    --------
    >>> results = search("eleuther")
    >>> [m["name"] for m in results]
    ['Pythia 6.9B', 'GPT-NeoX 20B']
    """
    q = query.lower()
    return [
        m for m in load_models()
        if q in m["name"].lower()
        or q in m["family"].lower()
        or q in m["organization"].lower()
    ]
