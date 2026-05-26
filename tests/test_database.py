"""Tests for openllm_selector.database."""

from unittest.mock import MagicMock, patch

import pytest
import requests as req

from openllm_selector.database import (
    compute_openness_score,
    fetch_recent_papers,
    filter_models,
    get_families,
    get_languages,
    get_model,
    get_organizations,
    load_models,
    rank_by_openness,
    search,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_models():
    return load_models()


# ---------------------------------------------------------------------------
# New fields: training_tokens_b, num_languages, has_instruct_version
# ---------------------------------------------------------------------------

class TestNewFields:

    # --- field presence and types ---

    def test_training_tokens_b_present(self, all_models):
        for m in all_models:
            assert "training_tokens_b" in m

    def test_training_tokens_b_is_float_or_none(self, all_models):
        for m in all_models:
            assert m["training_tokens_b"] is None or isinstance(m["training_tokens_b"], float), (
                f"{m['name']} training_tokens_b should be float or None"
            )

    def test_training_tokens_b_null_models(self, all_models):
        null_names = {m["name"] for m in all_models if m["training_tokens_b"] is None}
        assert null_names == {
            "Mistral 7B", "Mixtral 8x7B", "LLaVA 1.5 7B", "DeepSeek-R1",
            "Mixtral 8x22B", "GPT-OSS 20B", "Grok-1",
        }

    def test_training_tokens_b_positive_when_present(self, all_models):
        for m in all_models:
            if m["training_tokens_b"] is not None:
                assert m["training_tokens_b"] > 0, f"{m['name']} training_tokens_b must be positive"

    def test_known_training_tokens_values(self):
        assert get_model("OLMo 7B")["training_tokens_b"] == 2500.0
        assert get_model("Pythia 6.9B")["training_tokens_b"] == 300.0
        assert get_model("Llama 3.1 8B")["training_tokens_b"] == 15000.0
        assert get_model("DeepSeek-LLM 7B")["training_tokens_b"] == 2000.0
        assert get_model("Mistral 7B")["training_tokens_b"] is None

    def test_num_languages_present(self, all_models):
        for m in all_models:
            assert "num_languages" in m

    def test_num_languages_is_positive_int(self, all_models):
        for m in all_models:
            assert isinstance(m["num_languages"], int), (
                f"{m['name']} num_languages should be int"
            )
            assert m["num_languages"] >= 1, f"{m['name']} num_languages must be >= 1"

    def test_known_num_languages_values(self):
        assert get_model("BLOOM 176B")["num_languages"] == 46
        assert get_model("Qwen2 7B")["num_languages"] == 27
        assert get_model("Llama 3.1 8B")["num_languages"] == 8
        assert get_model("Mixtral 8x7B")["num_languages"] == 5
        assert get_model("Falcon 7B")["num_languages"] == 4
        assert get_model("Yi 1.5 9B")["num_languages"] == 2
        assert get_model("OLMo 7B")["num_languages"] == 1

    def test_multilingual_models_have_more_than_one_language(self, all_models):
        for m in all_models:
            if m["multilingual"]:
                assert m["num_languages"] > 1, (
                    f"{m['name']} is multilingual but num_languages={m['num_languages']}"
                )

    def test_non_multilingual_models_have_one_language(self, all_models):
        for m in all_models:
            if not m["multilingual"]:
                assert m["num_languages"] == 1, (
                    f"{m['name']} is not multilingual but num_languages={m['num_languages']}"
                )

    def test_has_instruct_version_present(self, all_models):
        for m in all_models:
            assert "has_instruct_version" in m

    def test_has_instruct_version_is_bool(self, all_models):
        for m in all_models:
            assert isinstance(m["has_instruct_version"], bool), (
                f"{m['name']} has_instruct_version should be bool"
            )

    def test_known_has_instruct_version_values(self):
        assert get_model("Pythia 6.9B")["has_instruct_version"] is False
        assert get_model("GPT-NeoX 20B")["has_instruct_version"] is False
        assert get_model("OLMo 7B")["has_instruct_version"] is True
        assert get_model("Llama 2 7B")["has_instruct_version"] is True
        assert get_model("Mistral 7B")["has_instruct_version"] is False

    # --- filter_models: training_tokens_b ---

    def test_filter_min_training_tokens(self):
        results = filter_models(min_training_tokens_b=10000)
        names = {m["name"] for m in results}
        assert "Llama 3.1 8B" in names
        assert all(m["training_tokens_b"] >= 10000 for m in results)

    def test_filter_max_training_tokens(self):
        results = filter_models(max_training_tokens_b=500)
        assert all(m["training_tokens_b"] <= 500 for m in results)
        names = {m["name"] for m in results}
        assert "Pythia 6.9B" in names   # 300
        assert "BLOOM 176B" in names    # 366
        assert "GPT-NeoX 20B" in names  # 472

    def test_filter_training_tokens_range(self):
        results = filter_models(min_training_tokens_b=2000, max_training_tokens_b=2000)
        names = {m["name"] for m in results}
        assert "Llama 2 7B" in names
        assert "DeepSeek-LLM 7B" in names
        assert all(m["training_tokens_b"] == 2000 for m in results)

    def test_filter_training_tokens_excludes_nulls(self):
        results = filter_models(min_training_tokens_b=1)
        names = {m["name"] for m in results}
        assert "Mistral 7B" not in names
        assert "Mixtral 8x7B" not in names
        assert "LLaVA 1.5 7B" not in names

    def test_filter_max_training_tokens_excludes_nulls(self):
        results = filter_models(max_training_tokens_b=99999)
        names = {m["name"] for m in results}
        assert "Mistral 7B" not in names
        assert "Mixtral 8x7B" not in names
        assert "LLaVA 1.5 7B" not in names

    def test_filter_training_tokens_boundary_inclusive(self):
        assert any(m["name"] == "Pythia 6.9B" for m in filter_models(min_training_tokens_b=300))
        assert any(m["name"] == "Pythia 6.9B" for m in filter_models(max_training_tokens_b=300))

    def test_filter_training_tokens_impossible_range_empty(self):
        assert filter_models(min_training_tokens_b=5000, max_training_tokens_b=1000) == []

    # --- filter_models: num_languages ---

    def test_filter_min_num_languages(self):
        results = filter_models(min_num_languages=5)
        assert all(m["num_languages"] >= 5 for m in results)
        names = {m["name"] for m in results}
        assert "BLOOM 176B" in names
        assert "Qwen2 7B" in names
        assert "Llama 3.1 8B" in names
        assert "Mixtral 8x7B" in names

    def test_filter_max_num_languages(self):
        results = filter_models(max_num_languages=1)
        assert all(m["num_languages"] == 1 for m in results)
        assert len(results) == 17  # all English-only models

    def test_filter_num_languages_range(self):
        results = filter_models(min_num_languages=2, max_num_languages=5)
        assert all(2 <= m["num_languages"] <= 5 for m in results)
        names = {m["name"] for m in results}
        assert "Falcon 7B" in names     # 4
        assert "Mixtral 8x7B" in names  # 5
        assert "Yi 1.5 9B" in names     # 2
        assert "BLOOM 176B" not in names  # 46

    def test_filter_num_languages_bloom(self):
        results = filter_models(min_num_languages=40)
        assert len(results) == 3
        names = {m["name"] for m in results}
        assert names == {"BLOOM 176B", "Qwen3 8B", "Apertus 8B"}

    def test_filter_num_languages_boundary_inclusive(self):
        assert any(m["name"] == "Mixtral 8x7B" for m in filter_models(min_num_languages=5))
        assert any(m["name"] == "Mixtral 8x7B" for m in filter_models(max_num_languages=5))

    # --- filter_models: has_instruct_version ---

    def test_filter_has_instruct_version_false(self):
        results = filter_models(has_instruct_version=False)
        names = {m["name"] for m in results}
        assert names == {
            "Pythia 6.9B", "GPT-NeoX 20B", "Mistral 7B",
            "GPT-J 6B", "Grok-1", "Phi-2",
        }

    def test_filter_has_instruct_version_true(self):
        results = filter_models(has_instruct_version=True)
        assert len(results) == 27
        assert all(m["has_instruct_version"] is True for m in results)

    def test_filter_has_instruct_and_openness(self):
        results = filter_models(has_instruct_version=True, min_openness=5)
        names = {m["name"] for m in results}
        assert "OLMo 7B" in names
        assert "OLMo 2 7B" in names
        assert "Pythia 6.9B" not in names  # instruct=False

    def test_filter_has_instruct_and_training_tokens(self):
        results = filter_models(has_instruct_version=False, max_training_tokens_b=500)
        names = {m["name"] for m in results}
        assert "Pythia 6.9B" in names
        assert "GPT-NeoX 20B" in names

    # --- languages field ---

    def test_languages_field_present(self, all_models):
        for m in all_models:
            assert "languages" in m, f"{m['name']} missing languages field"

    def test_languages_is_list_of_strings(self, all_models):
        for m in all_models:
            assert isinstance(m["languages"], list), (
                f"{m['name']} languages should be a list"
            )
            assert all(isinstance(lang, str) for lang in m["languages"]), (
                f"{m['name']} languages list should contain only strings"
            )

    def test_languages_non_empty(self, all_models):
        for m in all_models:
            assert len(m["languages"]) >= 1, f"{m['name']} languages must not be empty"

    def test_non_multilingual_models_have_only_english(self, all_models):
        for m in all_models:
            if not m["multilingual"]:
                assert m["languages"] == ["English"], (
                    f"{m['name']} is not multilingual but languages={m['languages']}"
                )

    def test_multilingual_models_have_multiple_languages(self, all_models):
        for m in all_models:
            if m["multilingual"]:
                assert len(m["languages"]) > 1, (
                    f"{m['name']} is multilingual but has only {len(m['languages'])} language(s)"
                )

    def test_known_languages_bloom(self):
        bloom = get_model("BLOOM 176B")
        assert len(bloom["languages"]) == 46
        assert "English" in bloom["languages"]
        assert "French" in bloom["languages"]
        assert "Yoruba" in bloom["languages"]
        assert "Akkadian" in bloom["languages"]

    def test_known_languages_falcon(self):
        falcon = get_model("Falcon 7B")
        assert set(falcon["languages"]) == {"English", "German", "Spanish", "French"}
        assert get_model("Falcon 40B")["languages"] == get_model("Falcon 7B")["languages"]

    def test_known_languages_mixtral(self):
        mixtral = get_model("Mixtral 8x7B")
        assert set(mixtral["languages"]) == {"English", "French", "Italian", "German", "Spanish"}

    def test_known_languages_llama31(self):
        llama = get_model("Llama 3.1 8B")
        assert len(llama["languages"]) == 8
        assert "Hindi" in llama["languages"]
        assert "Thai" in llama["languages"]
        assert "Portuguese" in llama["languages"]

    def test_known_languages_english_only(self):
        for name in ["OLMo 7B", "Pythia 6.9B", "Mistral 7B", "Gemma 2B", "LLaVA 1.5 7B"]:
            assert get_model(name)["languages"] == ["English"], (
                f"{name} should have languages=['English']"
            )

    def test_known_languages_chinese_models(self):
        for name in ["Qwen2 7B", "Yi 1.5 9B", "DeepSeek-LLM 7B"]:
            langs = get_model(name)["languages"]
            assert "English" in langs
            assert "Chinese" in langs

    # --- get_languages ---

    def test_get_languages_returns_sorted_list(self):
        langs = get_languages()
        assert langs == sorted(langs)

    def test_get_languages_contains_english(self):
        assert "English" in get_languages()

    def test_get_languages_contains_bloom_languages(self):
        langs = get_languages()
        for lang in ["Akkadian", "Yoruba", "Welsh", "Swahili"]:
            assert lang in langs, f"{lang} missing from get_languages()"

    def test_get_languages_no_duplicates(self):
        langs = get_languages()
        assert len(langs) == len(set(langs))

    # --- filter_models: language ---

    def test_filter_language_english(self):
        results = filter_models(language="English")
        assert len(results) == 32  # all models except Apertus 8B (FineWeb-2 languages, no English)

    def test_filter_language_hindi(self):
        results = filter_models(language="Hindi")
        names = {m["name"] for m in results}
        assert names == {"BLOOM 176B", "Llama 3.1 8B", "Gemma 3 27B", "Qwen3 8B", "Sarvam 30B", "Apertus 8B"}

    def test_filter_language_thai(self):
        results = filter_models(language="Thai")
        names = {m["name"] for m in results}
        assert names == {"BLOOM 176B", "Llama 3.1 8B", "Qwen2.5 7B", "Gemma 3 27B", "Qwen3 8B", "Apertus 8B"}

    def test_filter_language_italian(self):
        results = filter_models(language="Italian")
        names = {m["name"] for m in results}
        assert names == {
            "Mixtral 8x7B", "Llama 3.1 8B", "Qwen2.5 7B", "Mixtral 8x22B",
            "Gemma 3 27B", "Qwen3 8B", "Apertus 8B",
        }

    def test_filter_language_case_insensitive(self):
        assert filter_models(language="english") == filter_models(language="English")
        assert filter_models(language="FRENCH") == filter_models(language="French")

    def test_filter_language_no_match_returns_empty(self):
        assert filter_models(language="Klingon") == []

    def test_filter_language_combined_with_other_filters(self):
        results = filter_models(language="French", min_openness=4)
        assert all("French" in m["languages"] for m in results)
        assert all(m["openness_score"] >= 4 for m in results)

    # --- model_type field ---

    def test_model_type_present(self, all_models):
        for m in all_models:
            assert "model_type" in m, f"{m['name']} missing model_type field"

    def test_model_type_is_string(self, all_models):
        for m in all_models:
            assert isinstance(m["model_type"], str), (
                f"{m['name']} model_type should be str"
            )

    def test_model_type_valid_values(self, all_models):
        valid = {"base", "instruct", "reasoning"}
        for m in all_models:
            assert m["model_type"] in valid, (
                f"{m['name']} has invalid model_type '{m['model_type']}'"
            )

    def test_known_model_type_instruct(self):
        assert get_model("Phi-3 Mini 4K")["model_type"] == "instruct"
        assert get_model("LLaVA 1.5 7B")["model_type"] == "instruct"

    def test_known_model_type_reasoning(self):
        assert get_model("DeepSeek-R1")["model_type"] == "reasoning"

    def test_known_model_type_base(self):
        for name in ["OLMo 7B", "Llama 3.1 8B", "Pythia 6.9B", "BLOOM 176B"]:
            assert get_model(name)["model_type"] == "base", (
                f"{name} should have model_type='base'"
            )

    def test_filter_model_type_base(self):
        results = filter_models(model_type="base")
        assert all(m["model_type"] == "base" for m in results)
        assert len(results) == 28

    def test_filter_model_type_instruct(self):
        results = filter_models(model_type="instruct")
        names = {m["name"] for m in results}
        assert names == {"Phi-3 Mini 4K", "LLaVA 1.5 7B"}

    def test_filter_model_type_reasoning(self):
        results = filter_models(model_type="reasoning")
        assert len(results) == 3
        names = {m["name"] for m in results}
        assert names == {"DeepSeek-R1", "GPT-OSS 20B", "Sarvam 30B"}

    def test_filter_model_type_case_insensitive(self):
        assert filter_models(model_type="BASE") == filter_models(model_type="base")
        assert filter_models(model_type="Instruct") == filter_models(model_type="instruct")

    def test_filter_model_type_no_match_returns_empty(self):
        assert filter_models(model_type="unknown") == []

    # --- has_think_version field ---

    def test_has_think_version_present(self, all_models):
        for m in all_models:
            assert "has_think_version" in m, f"{m['name']} missing has_think_version field"

    def test_has_think_version_is_bool(self, all_models):
        for m in all_models:
            assert isinstance(m["has_think_version"], bool), (
                f"{m['name']} has_think_version should be bool"
            )

    def test_known_has_think_version_true(self):
        assert get_model("OLMo 3 32B")["has_think_version"] is True
        assert get_model("DeepSeek-R1")["has_think_version"] is True
        assert get_model("Phi-4")["has_think_version"] is True

    def test_known_has_think_version_false(self):
        for name in ["OLMo 7B", "OLMo 2 7B", "Pythia 6.9B", "Llama 3.1 8B"]:
            assert get_model(name)["has_think_version"] is False, (
                f"{name} should have has_think_version=False"
            )

    def test_filter_has_think_version_true(self):
        results = filter_models(has_think_version=True)
        names = {m["name"] for m in results}
        assert names == {"OLMo 3 32B", "DeepSeek-R1", "Phi-4", "Qwen3 8B", "GPT-OSS 20B", "Sarvam 30B"}

    def test_filter_has_think_version_false(self):
        results = filter_models(has_think_version=False)
        assert len(results) == 27
        assert all(m["has_think_version"] is False for m in results)

    def test_filter_has_think_version_combined(self):
        results = filter_models(has_think_version=True, open_weights=True)
        assert all(m["has_think_version"] is True for m in results)
        assert all(m["open_weights"] is True for m in results)

    # --- notes field (optional) ---

    def test_notes_present_on_deepseek_r1(self):
        model = get_model("DeepSeek-R1")
        assert "notes" in model

    def test_notes_absent_or_none_on_standard_model(self):
        model = get_model("OLMo 7B")
        assert model.get("notes") is None

    def test_notes_non_empty_string_when_present(self, all_models):
        for m in all_models:
            if "notes" in m:
                assert isinstance(m["notes"], str) and m["notes"].strip(), (
                    f"{m['name']} has an empty or non-string notes field"
                )


# ---------------------------------------------------------------------------
# compute_openness_score
# ---------------------------------------------------------------------------

class TestComputeOpennessScore:
    def test_fully_open_model_scores_5(self):
        # All booleans True + Apache license
        model = {
            "open_weights": True, "open_training_data": True,
            "intermediate_checkpoints": True, "open_code": True,
            "license": "Apache 2.0",
        }
        assert compute_openness_score(model) == 5

    def test_mit_license_counts_as_permissive(self):
        model = {
            "open_weights": True, "open_training_data": True,
            "intermediate_checkpoints": True, "open_code": True,
            "license": "MIT",
        }
        assert compute_openness_score(model) == 5

    def test_non_permissive_license_does_not_contribute(self):
        model = {
            "open_weights": True, "open_training_data": True,
            "intermediate_checkpoints": True, "open_code": True,
            "license": "Llama 2 Community License",
        }
        assert compute_openness_score(model) == 4

    def test_only_open_weights_non_permissive_scores_1(self):
        model = {
            "open_weights": True, "open_training_data": False,
            "intermediate_checkpoints": False, "open_code": False,
            "license": "Gemma Terms of Use",
        }
        assert compute_openness_score(model) == 1

    def test_only_open_weights_permissive_scores_2(self):
        model = {
            "open_weights": True, "open_training_data": False,
            "intermediate_checkpoints": False, "open_code": False,
            "license": "Apache 2.0",
        }
        assert compute_openness_score(model) == 2

    def test_all_false_scores_0(self):
        model = {
            "open_weights": False, "open_training_data": False,
            "intermediate_checkpoints": False, "open_code": False,
            "license": "Proprietary",
        }
        assert compute_openness_score(model) == 0

    def test_olmo_7b_scores_5(self):
        assert compute_openness_score(get_model("OLMo 7B")) == 5

    def test_llama2_7b_scores_1(self):
        assert compute_openness_score(get_model("Llama 2 7B")) == 1

    def test_score_is_injected_by_load_models(self):
        models = load_models()
        assert all("openness_score" in m for m in models)

    def test_score_matches_direct_computation(self):
        for m in load_models():
            assert m["openness_score"] == compute_openness_score(m)


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------

class TestLoadModels:
    def test_returns_list(self, all_models):
        assert isinstance(all_models, list)

    def test_nonempty(self, all_models):
        assert len(all_models) > 0

    def test_each_record_has_required_keys(self, all_models):
        required = {
            "name", "family", "organization", "country_of_origin",
            "release_year", "size_b", "training_tokens_b", "context_window",
            "modality", "architecture", "license", "open_weights",
            "open_training_data", "intermediate_checkpoints", "open_code",
            "multilingual", "num_languages", "languages", "has_instruct_version",
            "model_type", "has_think_version", "foundational_paper",
            "huggingface_id", "openness_score",
        }
        for model in all_models:
            assert required <= model.keys(), f"{model['name']} is missing fields"

    def test_openness_scores_in_range(self, all_models):
        for model in all_models:
            assert 1 <= model["openness_score"] <= 5, (
                f"{model['name']} has openness_score {model['openness_score']}"
            )

    def test_size_b_positive(self, all_models):
        for model in all_models:
            assert model["size_b"] > 0, f"{model['name']} has non-positive size_b"

    def test_modality_is_list(self, all_models):
        for model in all_models:
            assert isinstance(model["modality"], list), (
                f"{model['name']} modality should be a list"
            )

    def test_boolean_fields_are_bool(self, all_models):
        bool_fields = ["open_weights", "open_training_data", "intermediate_checkpoints", "open_code"]
        for model in all_models:
            for field in bool_fields:
                assert isinstance(model[field], bool), (
                    f"{model['name']}.{field} should be bool"
                )

    _NON_ARXIV_PAPERS = {
        "Mixtral 8x22B",  # no arXiv paper exists; links to Mistral blog
        "GPT-J 6B",       # no dedicated arXiv paper; links to mesh-transformer-jax GitHub repo
        "Grok-1",         # no arXiv paper; links to xAI blog post
        "Sarvam 30B",     # no arXiv paper; links to Sarvam AI blog post
    }

    def test_foundational_paper_is_arxiv_url(self, all_models):
        for model in all_models:
            if model["name"] in self._NON_ARXIV_PAPERS:
                continue
            assert model["foundational_paper"].startswith("https://arxiv.org/"), (
                f"{model['name']} foundational_paper is not an arXiv URL"
            )

    def test_context_window_is_positive_int(self, all_models):
        for model in all_models:
            assert isinstance(model["context_window"], int), (
                f"{model['name']} context_window should be int"
            )
            assert model["context_window"] > 0, (
                f"{model['name']} context_window must be positive"
            )

    def test_architecture_values_are_valid(self, all_models):
        valid = {"decoder-only", "encoder-only", "encoder-decoder", "mixture-of-experts"}
        for model in all_models:
            assert model["architecture"] in valid, (
                f"{model['name']} has unknown architecture '{model['architecture']}'"
            )

    def test_mixtral_is_mixture_of_experts(self, all_models):
        mixtral = next(m for m in all_models if m["name"] == "Mixtral 8x7B")
        assert mixtral["architecture"] == "mixture-of-experts"

    def test_llava_is_encoder_decoder(self, all_models):
        llava = next(m for m in all_models if m["name"] == "LLaVA 1.5 7B")
        assert llava["architecture"] == "encoder-decoder"

    def test_names_are_unique(self, all_models):
        names = [m["name"] for m in all_models]
        assert len(names) == len(set(names)), "Duplicate model names found"

    def test_returns_independent_copy_each_call(self):
        first = load_models()
        first[0]["name"] = "MUTATED"
        second = load_models()
        assert second[0]["name"] != "MUTATED"


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_exact_match(self):
        model = get_model("OLMo 7B")
        assert model is not None
        assert model["name"] == "OLMo 7B"

    def test_case_insensitive(self):
        assert get_model("olmo 7b") is not None
        assert get_model("OLMO 7B") is not None
        assert get_model("OlMo 7b") is not None

    def test_returns_none_for_unknown_name(self):
        assert get_model("GPT-4") is None
        assert get_model("") is None
        assert get_model("nonexistent model xyz") is None

    def test_partial_name_does_not_match(self):
        # "OLMo" alone should not match "OLMo 7B" (exact match required)
        assert get_model("OLMo") is None

    def test_returned_record_has_correct_fields(self):
        model = get_model("Pythia 6.9B")
        assert model["openness_score"] == 5
        assert model["organization"] == "EleutherAI"
        assert model["open_training_data"] is True
        assert model["intermediate_checkpoints"] is True


# ---------------------------------------------------------------------------
# filter_models
# ---------------------------------------------------------------------------

class TestFilterModels:
    def test_no_filters_returns_all(self, all_models):
        assert len(filter_models()) == len(all_models)

    # --- modality ---

    def test_filter_text_modality(self):
        results = filter_models(modality="text")
        assert len(results) > 0
        assert all("text" in m["modality"] for m in results)

    def test_filter_image_modality(self):
        results = filter_models(modality="image")
        assert len(results) > 0
        assert all("image" in m["modality"] for m in results)

    def test_filter_modality_case_insensitive(self):
        lower = filter_models(modality="text")
        upper = filter_models(modality="TEXT")
        assert [m["name"] for m in lower] == [m["name"] for m in upper]

    def test_filter_nonexistent_modality_returns_empty(self):
        assert filter_models(modality="video") == []

    # --- size ---

    def test_filter_max_size(self):
        results = filter_models(max_size_b=8.0)
        assert all(m["size_b"] <= 8.0 for m in results)

    def test_filter_min_size(self):
        results = filter_models(min_size_b=20.0)
        assert all(m["size_b"] >= 20.0 for m in results)

    def test_filter_size_range(self):
        results = filter_models(min_size_b=5.0, max_size_b=10.0)
        assert all(5.0 <= m["size_b"] <= 10.0 for m in results)

    def test_filter_size_range_no_matches(self):
        # No model has exactly 1 B parameters in the DB
        assert filter_models(min_size_b=1.0, max_size_b=1.5) == []

    def test_size_boundary_inclusive(self):
        # OLMo 7B has size_b = 7.0; both bounds should include it
        assert any(m["name"] == "OLMo 7B" for m in filter_models(min_size_b=7.0))
        assert any(m["name"] == "OLMo 7B" for m in filter_models(max_size_b=7.0))

    def test_filter_impossible_size_range_returns_empty(self):
        assert filter_models(min_size_b=100.0, max_size_b=50.0) == []

    # --- openness ---

    def test_filter_min_openness_5(self):
        results = filter_models(min_openness=5)
        assert all(m["openness_score"] == 5 for m in results)
        assert len(results) > 0

    def test_filter_max_openness_2(self):
        results = filter_models(max_openness=2)
        assert all(m["openness_score"] <= 2 for m in results)

    def test_filter_openness_range(self):
        results = filter_models(min_openness=3, max_openness=4)
        assert all(3 <= m["openness_score"] <= 4 for m in results)

    def test_filter_openness_score_1_returns_models(self):
        # Llama 2 7B: only open_weights=True, non-permissive license → score 1
        results = filter_models(min_openness=1, max_openness=1)
        assert len(results) > 0
        assert all(m["openness_score"] == 1 for m in results)

    # --- boolean flags ---

    def test_filter_open_training_data_true(self):
        results = filter_models(open_training_data=True)
        assert all(m["open_training_data"] is True for m in results)
        assert len(results) > 0

    def test_filter_open_training_data_false(self):
        results = filter_models(open_training_data=False)
        assert all(m["open_training_data"] is False for m in results)
        assert len(results) > 0

    def test_filter_intermediate_checkpoints_true(self):
        results = filter_models(intermediate_checkpoints=True)
        assert all(m["intermediate_checkpoints"] is True for m in results)
        assert len(results) > 0

    def test_filter_open_code_true(self):
        results = filter_models(open_code=True)
        assert all(m["open_code"] is True for m in results)

    # --- multilingual ---

    def test_multilingual_field_is_bool(self, all_models):
        for m in all_models:
            assert isinstance(m["multilingual"], bool), (
                f"{m['name']} multilingual should be bool"
            )

    def test_filter_multilingual_true(self):
        results = filter_models(multilingual=True)
        assert len(results) > 0
        assert all(m["multilingual"] is True for m in results)

    def test_filter_multilingual_false(self):
        results = filter_models(multilingual=False)
        assert len(results) > 0
        assert all(m["multilingual"] is False for m in results)

    def test_multilingual_true_and_false_partition_all(self, all_models):
        true_set = {m["name"] for m in filter_models(multilingual=True)}
        false_set = {m["name"] for m in filter_models(multilingual=False)}
        all_names = {m["name"] for m in all_models}
        assert true_set | false_set == all_names
        assert true_set & false_set == set()

    def test_multilingual_true_includes_bloom(self):
        results = filter_models(multilingual=True)
        assert any(m["name"] == "BLOOM 176B" for m in results)

    def test_multilingual_true_includes_mixtral(self):
        results = filter_models(multilingual=True)
        assert any(m["name"] == "Mixtral 8x7B" for m in results)

    def test_multilingual_false_excludes_bloom(self):
        results = filter_models(multilingual=False)
        assert all(m["name"] != "BLOOM 176B" for m in results)

    def test_combined_multilingual_and_openness(self):
        results = filter_models(multilingual=True, min_openness=3)
        assert all(m["multilingual"] is True and m["openness_score"] >= 3 for m in results)

    def test_combined_multilingual_and_size(self):
        results = filter_models(multilingual=True, max_size_b=10.0)
        assert all(m["multilingual"] is True and m["size_b"] <= 10.0 for m in results)

    # --- organization ---

    def test_filter_organization_exact(self):
        results = filter_models(organization="Meta")
        assert all("Meta" in m["organization"] for m in results)
        assert len(results) > 0

    def test_filter_organization_substring(self):
        results = filter_models(organization="allen")
        assert len(results) > 0
        assert all("allen" in m["organization"].lower() for m in results)

    def test_filter_organization_case_insensitive(self):
        lower = filter_models(organization="meta")
        upper = filter_models(organization="META")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_filter_organization_no_match_returns_empty(self):
        assert filter_models(organization="Anthropic") == []

    # --- family ---

    def test_filter_family_exact(self):
        results = filter_models(family="LLaMA")
        assert all(m["family"] == "LLaMA" for m in results)
        assert len(results) > 0

    def test_filter_family_case_insensitive(self):
        lower = filter_models(family="llama")
        upper = filter_models(family="LLAMA")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_filter_family_no_match_returns_empty(self):
        assert filter_models(family="Anthropic") == []

    def test_filter_family_is_exact_not_substring(self):
        # "OLMo" should not match "OLMo 7B" as a family name,
        # but "OLMo" IS a family, so this should work; "OL" should not.
        assert filter_models(family="OL") == []

    # --- license ---

    def test_filter_license_apache(self):
        results = filter_models(license="Apache")
        assert all("Apache" in m["license"] for m in results)
        assert len(results) > 0

    def test_filter_license_case_insensitive(self):
        lower = filter_models(license="apache")
        upper = filter_models(license="APACHE")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_filter_license_no_match_returns_empty(self):
        assert filter_models(license="WTFPL") == []

    # --- combined filters ---

    def test_combined_size_and_openness(self):
        results = filter_models(max_size_b=10.0, min_openness=5)
        assert all(m["size_b"] <= 10.0 and m["openness_score"] == 5 for m in results)

    def test_combined_family_and_open_weights(self):
        results = filter_models(family="LLaMA", open_weights=True)
        assert all(m["family"] == "LLaMA" and m["open_weights"] is True for m in results)

    def test_combined_contradictory_filters_returns_empty(self):
        # All models in DB have open_weights=True; asking for False gives nothing
        assert filter_models(open_weights=False) == []

    def test_combined_filters_narrow_to_one(self):
        # LLaVA is the only multimodal model
        results = filter_models(modality="image", max_size_b=10.0)
        assert len(results) == 1
        assert results[0]["name"] == "LLaVA 1.5 7B"

    # --- country_of_origin ---

    def test_filter_country_exact(self):
        results = filter_models(country_of_origin="France")
        assert all(m["country_of_origin"] == "France" for m in results)
        assert len(results) > 0

    def test_filter_country_returns_correct_models(self):
        results = filter_models(country_of_origin="France")
        names = {m["name"] for m in results}
        assert "BLOOM 176B" in names
        assert "Mistral 7B" in names
        assert "Mixtral 8x7B" in names

    def test_filter_country_case_insensitive(self):
        lower = filter_models(country_of_origin="france")
        upper = filter_models(country_of_origin="FRANCE")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_filter_country_no_match_returns_empty(self):
        assert filter_models(country_of_origin="Germany") == []

    def test_filter_country_is_exact_not_substring(self):
        # "United" alone should not match "United States" or "United Arab Emirates"
        assert filter_models(country_of_origin="United") == []

    def test_filter_country_china(self):
        results = filter_models(country_of_origin="China")
        names = {m["name"] for m in results}
        assert "Qwen2 7B" in names
        assert "Yi 1.5 9B" in names
        assert "DeepSeek-LLM 7B" in names
        assert all(m["country_of_origin"] == "China" for m in results)

    def test_filter_country_uae(self):
        results = filter_models(country_of_origin="United Arab Emirates")
        names = {m["name"] for m in results}
        assert "Falcon 7B" in names
        assert "Falcon 40B" in names
        assert all(m["country_of_origin"] == "United Arab Emirates" for m in results)

    def test_combined_country_and_openness(self):
        results = filter_models(country_of_origin="United States", min_openness=5)
        assert all(
            m["country_of_origin"] == "United States" and m["openness_score"] == 5
            for m in results
        )
        assert len(results) > 0

    def test_combined_country_and_architecture(self):
        results = filter_models(country_of_origin="France", architecture="mixture-of-experts")
        assert len(results) == 2
        names = {m["name"] for m in results}
        assert names == {"Mixtral 8x7B", "Mixtral 8x22B"}

    # --- context window ---

    def test_filter_min_context_window(self):
        results = filter_models(min_context_window=8192)
        assert all(m["context_window"] >= 8192 for m in results)
        assert len(results) > 0

    def test_filter_max_context_window(self):
        results = filter_models(max_context_window=4096)
        assert all(m["context_window"] <= 4096 for m in results)
        assert len(results) > 0

    def test_filter_context_window_range(self):
        results = filter_models(min_context_window=4096, max_context_window=8192)
        assert all(4096 <= m["context_window"] <= 8192 for m in results)
        assert len(results) > 0

    def test_filter_context_window_boundary_inclusive(self):
        # Mistral 7B has context_window = 8192; both bounds should include it
        assert any(m["name"] == "Mistral 7B" for m in filter_models(min_context_window=8192))
        assert any(m["name"] == "Mistral 7B" for m in filter_models(max_context_window=8192))

    def test_filter_context_window_no_matches(self):
        # No model in the DB has a context window between 9000 and 10000
        assert filter_models(min_context_window=9000, max_context_window=10000) == []

    def test_filter_context_window_impossible_range_returns_empty(self):
        assert filter_models(min_context_window=100000, max_context_window=1000) == []

    def test_filter_long_context_models(self):
        # Only Llama 3.1 8B and Qwen2 7B have context_window >= 131072
        results = filter_models(min_context_window=131072)
        names = {m["name"] for m in results}
        assert "Llama 3.1 8B" in names
        assert "Qwen2 7B" in names

    def test_filter_short_context_models(self):
        results = filter_models(max_context_window=2048)
        assert all(m["context_window"] <= 2048 for m in results)
        # OLMo 7B, Pythia 6.9B, BLOOM, GPT-NeoX, Falcon 7B, Falcon 40B
        assert len(results) >= 6

    def test_combined_context_window_and_openness(self):
        results = filter_models(min_context_window=4096, min_openness=5)
        assert all(
            m["context_window"] >= 4096 and m["openness_score"] == 5
            for m in results
        )

    def test_combined_context_window_and_size(self):
        results = filter_models(min_context_window=8192, max_size_b=10.0)
        assert all(
            m["context_window"] >= 8192 and m["size_b"] <= 10.0
            for m in results
        )

    # --- release_year ---

    def test_release_year_is_valid_int(self, all_models):
        for m in all_models:
            assert isinstance(m["release_year"], int), (
                f"{m['name']} release_year should be int"
            )
            assert 2020 <= m["release_year"] <= 2030, (
                f"{m['name']} release_year {m['release_year']} seems wrong"
            )

    def test_filter_min_release_year(self):
        results = filter_models(min_release_year=2024)
        assert all(m["release_year"] >= 2024 for m in results)
        assert len(results) > 0

    def test_filter_max_release_year(self):
        results = filter_models(max_release_year=2022)
        assert all(m["release_year"] <= 2022 for m in results)
        assert len(results) > 0

    def test_filter_release_year_range(self):
        results = filter_models(min_release_year=2023, max_release_year=2023)
        assert all(m["release_year"] == 2023 for m in results)
        assert len(results) > 0

    def test_filter_release_year_boundary_inclusive(self):
        year_2022_models = filter_models(min_release_year=2022, max_release_year=2022)
        names = {m["name"] for m in year_2022_models}
        assert "BLOOM 176B" in names
        assert "GPT-NeoX 20B" in names

    def test_filter_release_year_no_match(self):
        assert filter_models(min_release_year=2020, max_release_year=2020) == []

    def test_filter_release_year_impossible_range_returns_empty(self):
        assert filter_models(min_release_year=2025, max_release_year=2022) == []

    def test_combined_release_year_and_family(self):
        results = filter_models(min_release_year=2024, family="LLaMA")
        assert all(
            m["release_year"] >= 2024 and m["family"] == "LLaMA"
            for m in results
        )
        assert len(results) > 0

    # --- exclude_* ---

    def test_exclude_family(self, all_models):
        results = filter_models(exclude_family="LLaMA")
        assert all(m["family"] != "LLaMA" for m in results)
        assert len(results) == len(all_models) - len(filter_models(family="LLaMA"))

    def test_exclude_family_case_insensitive(self):
        lower = filter_models(exclude_family="llama")
        upper = filter_models(exclude_family="LLAMA")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_exclude_family_nonexistent_leaves_all(self, all_models):
        results = filter_models(exclude_family="Anthropic")
        assert len(results) == len(all_models)

    def test_exclude_organization(self, all_models):
        results = filter_models(exclude_organization="Meta")
        assert all("meta" not in m["organization"].lower() for m in results)
        assert len(results) < len(all_models)

    def test_exclude_organization_case_insensitive(self):
        lower = filter_models(exclude_organization="meta")
        upper = filter_models(exclude_organization="META")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_exclude_license(self, all_models):
        results = filter_models(exclude_license="Apache")
        assert all("apache" not in m["license"].lower() for m in results)
        assert len(results) < len(all_models)

    def test_exclude_license_case_insensitive(self):
        lower = filter_models(exclude_license="apache")
        upper = filter_models(exclude_license="APACHE")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_exclude_architecture(self, all_models):
        results = filter_models(exclude_architecture="decoder-only")
        assert all(m["architecture"] != "decoder-only" for m in results)
        assert len(results) < len(all_models)

    def test_exclude_architecture_case_insensitive(self):
        lower = filter_models(exclude_architecture="decoder-only")
        upper = filter_models(exclude_architecture="DECODER-ONLY")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_exclude_country_of_origin(self, all_models):
        results = filter_models(exclude_country_of_origin="China")
        assert all(m["country_of_origin"] != "China" for m in results)
        assert len(results) == len(all_models) - len(filter_models(country_of_origin="China"))

    def test_exclude_country_of_origin_case_insensitive(self):
        lower = filter_models(exclude_country_of_origin="china")
        upper = filter_models(exclude_country_of_origin="CHINA")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_exclude_modality(self, all_models):
        results = filter_models(exclude_modality="image")
        assert all("image" not in m["modality"] for m in results)
        assert len(results) == len(all_models) - 2  # LLaVA 1.5 7B and Gemma 3 27B have image

    def test_exclude_modality_case_insensitive(self):
        lower = filter_models(exclude_modality="image")
        upper = filter_models(exclude_modality="IMAGE")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_include_and_exclude_combine(self):
        results = filter_models(
            architecture="decoder-only",
            exclude_country_of_origin="China",
        )
        assert all(m["architecture"] == "decoder-only" for m in results)
        assert all(m["country_of_origin"] != "China" for m in results)

    def test_include_and_exclude_same_field_can_empty(self):
        # Including and excluding the same family must return nothing
        results = filter_models(family="LLaMA", exclude_family="LLaMA")
        assert results == []

    def test_exclude_nonexistent_license_leaves_all(self, all_models):
        results = filter_models(exclude_license="WTFPL")
        assert len(results) == len(all_models)

    def test_exclude_model_type_base(self, all_models):
        results = filter_models(exclude_model_type="base")
        assert all(m["model_type"] != "base" for m in results)
        assert len(results) == len(all_models) - len(filter_models(model_type="base"))

    def test_exclude_model_type_instruct(self, all_models):
        results = filter_models(exclude_model_type="instruct")
        assert all(m["model_type"] != "instruct" for m in results)
        names = {m["name"] for m in results}
        assert "Phi-3 Mini 4K" not in names
        assert "LLaVA 1.5 7B" not in names

    def test_exclude_model_type_reasoning(self, all_models):
        results = filter_models(exclude_model_type="reasoning")
        assert all(m["model_type"] != "reasoning" for m in results)
        names = {m["name"] for m in results}
        assert "DeepSeek-R1" not in names
        assert "GPT-OSS 20B" not in names

    def test_exclude_model_type_case_insensitive(self):
        lower = filter_models(exclude_model_type="base")
        upper = filter_models(exclude_model_type="BASE")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_exclude_model_type_nonexistent_leaves_all(self, all_models):
        results = filter_models(exclude_model_type="unknown")
        assert len(results) == len(all_models)

    def test_exclude_model_type_combined_with_other_filters(self):
        results = filter_models(exclude_model_type="base", multilingual=True)
        assert all(m["model_type"] != "base" for m in results)
        assert all(m["multilingual"] is True for m in results)


# ---------------------------------------------------------------------------
# get_families
# ---------------------------------------------------------------------------

class TestGetFamilies:
    def test_returns_sorted_list(self):
        families = get_families()
        assert families == sorted(families)

    def test_contains_expected_families(self):
        families = get_families()
        for expected in ("OLMo", "LLaMA", "Mistral", "Falcon", "Gemma"):
            assert expected in families

    def test_no_duplicates(self):
        families = get_families()
        assert len(families) == len(set(families))

    def test_all_families_present_in_db(self, all_models):
        db_families = {m["family"] for m in all_models}
        assert set(get_families()) == db_families


# ---------------------------------------------------------------------------
# get_organizations
# ---------------------------------------------------------------------------

class TestGetOrganizations:
    def test_returns_sorted_list(self):
        orgs = get_organizations()
        assert orgs == sorted(orgs)

    def test_contains_expected_organizations(self):
        orgs = get_organizations()
        for expected in ("Meta", "EleutherAI", "Google DeepMind", "Mistral AI"):
            assert expected in orgs

    def test_no_duplicates(self):
        orgs = get_organizations()
        assert len(orgs) == len(set(orgs))

    def test_all_organizations_present_in_db(self, all_models):
        db_orgs = {m["organization"] for m in all_models}
        assert set(get_organizations()) == db_orgs


# ---------------------------------------------------------------------------
# rank_by_openness
# ---------------------------------------------------------------------------

class TestRankByOpenness:
    def test_default_descending(self, all_models):
        ranked = rank_by_openness()
        scores = [m["openness_score"] for m in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ascending(self, all_models):
        ranked = rank_by_openness(descending=False)
        scores = [m["openness_score"] for m in ranked]
        assert scores == sorted(scores)

    def test_ranks_full_db_when_none(self, all_models):
        assert len(rank_by_openness()) == len(all_models)

    def test_ranks_provided_subset(self):
        subset = filter_models(family="LLaMA")
        ranked = rank_by_openness(subset)
        assert len(ranked) == len(subset)

    def test_empty_list_returns_empty(self):
        assert rank_by_openness([]) == []

    def test_single_item_returns_same(self):
        model = get_model("OLMo 7B")
        result = rank_by_openness([model])
        assert result == [model]

    def test_does_not_mutate_input(self):
        subset = filter_models(family="LLaMA")
        original_order = [m["name"] for m in subset]
        rank_by_openness(subset)
        assert [m["name"] for m in subset] == original_order

    def test_top_ranked_have_highest_scores(self):
        ranked = rank_by_openness()
        top_score = ranked[0]["openness_score"]
        assert top_score == 5

    def test_bottom_ranked_ascending_have_lowest_score(self):
        ranked = rank_by_openness(descending=False)
        bottom_score = ranked[0]["openness_score"]
        assert bottom_score == min(m["openness_score"] for m in load_models())


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_by_name(self):
        results = search("OLMo 7B")
        names = [m["name"] for m in results]
        assert "OLMo 7B" in names

    def test_search_by_family(self):
        results = search("Pythia")
        assert all(
            "pythia" in m["name"].lower()
            or "pythia" in m["family"].lower()
            or "pythia" in m["organization"].lower()
            for m in results
        )
        assert len(results) > 0

    def test_search_by_organization(self):
        results = search("EleutherAI")
        assert all(
            "eleutherai" in m["organization"].lower()
            or "eleutherai" in m["name"].lower()
            or "eleutherai" in m["family"].lower()
            for m in results
        )
        assert len(results) > 0

    def test_case_insensitive_name(self):
        lower = search("olmo")
        upper = search("OLMO")
        mixed = search("OlMo")
        assert {m["name"] for m in lower} == {m["name"] for m in upper} == {m["name"] for m in mixed}

    def test_case_insensitive_organization(self):
        lower = search("meta")
        upper = search("META")
        assert {m["name"] for m in lower} == {m["name"] for m in upper}

    def test_partial_substring_matches(self):
        # "euth" is a substring of "EleutherAI"
        results = search("euth")
        assert len(results) > 0

    def test_no_match_returns_empty(self):
        assert search("zzznonexistent") == []

    def test_empty_string_matches_all(self, all_models):
        # Every field contains the empty string
        assert len(search("")) == len(all_models)

    def test_search_matches_across_fields(self):
        # "Allen" appears in organization, not name or family
        results = search("Allen")
        assert len(results) > 0
        assert all("allen" in m["organization"].lower() for m in results)


# ---------------------------------------------------------------------------
# fetch_recent_papers
# ---------------------------------------------------------------------------

_SAMPLE_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Using OLMo for Natural Language Inference</title>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-15T00:00:00Z</published>
    <summary>We evaluate OLMo on NLI benchmarks.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
  </entry>
  <entry>
    <title>OLMo in Low-Resource Settings</title>
    <id>http://arxiv.org/abs/2401.00002v2</id>
    <published>2024-01-10T00:00:00Z</published>
    <summary>Low-resource adaptation of OLMo.</summary>
    <author><name>Charlie Brown</name></author>
  </entry>
</feed>
"""

_EMPTY_ATOM = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _mock_response(text, status_error=None):
    mock = MagicMock()
    mock.text = text
    if status_error:
        mock.raise_for_status.side_effect = status_error
    else:
        mock.raise_for_status.return_value = None
    return mock


class TestFetchRecentPapers:
    def test_returns_list_of_dicts(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        assert isinstance(results, list)
        assert all(isinstance(p, dict) for p in results)

    def test_each_result_has_required_keys(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        required = {"title", "authors", "summary", "published", "arxiv_url"}
        for paper in results:
            assert required <= paper.keys()

    def test_parses_title_correctly(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        assert results[0]["title"] == "Using OLMo for Natural Language Inference"

    def test_parses_multiple_authors(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        assert results[0]["authors"] == ["Alice Smith", "Bob Jones"]

    def test_parses_single_author(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        assert results[1]["authors"] == ["Charlie Brown"]

    def test_parses_published_date(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        assert results[0]["published"] == "2024-01-15T00:00:00Z"

    def test_arxiv_url_uses_https(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        assert all(p["arxiv_url"].startswith("https://") for p in results)

    def test_arxiv_url_contains_arxiv_id(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        assert "2401.00001" in results[0]["arxiv_url"]

    def test_correct_number_of_results(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            results = fetch_recent_papers("OLMo")
        # _SAMPLE_ATOM contains exactly 2 entries
        assert len(results) == 2

    def test_max_results_forwarded_to_api(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            fetch_recent_papers("OLMo", max_results=7)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["max_results"] == 7

    def test_model_name_used_in_query(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_SAMPLE_ATOM)
            fetch_recent_papers("Pythia 6.9B")
        _, kwargs = mock_get.call_args
        assert "Pythia 6.9B" in kwargs["params"]["search_query"]

    def test_empty_feed_returns_empty_list(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(_EMPTY_ATOM)
            results = fetch_recent_papers("NonExistentModel99999")
        assert results == []

    def test_http_error_raises(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                _SAMPLE_ATOM,
                status_error=req.exceptions.HTTPError("503 Service Unavailable"),
            )
            with pytest.raises(req.exceptions.HTTPError):
                fetch_recent_papers("OLMo")

    def test_network_error_raises(self):
        with patch("openllm_selector.database.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.ConnectionError("unreachable")
            with pytest.raises(req.exceptions.ConnectionError):
                fetch_recent_papers("OLMo")
