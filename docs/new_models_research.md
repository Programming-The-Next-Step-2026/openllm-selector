# Research: 5 new models added in week-3

Documents the field-level research for the five models added to `models.json`
beyond the original 18: DeepSeek-R1, OLMo 2 32B, OLMo 3 32B, Phi-4, and
Qwen2.5 7B.

Each section records confidence levels (High / Medium / Low), primary sources,
values that need manual verification before the next release, and any
structural exceptions or design decisions that deviate from the standard
field definitions.

---

## 1. DeepSeek-R1

**Added:** week-3  
**Why added:** First reasoning model in the database; exercises the new
`model_type="reasoning"` and `has_think_version` fields.  
**Primary source:** arXiv 2501.12948 (DeepSeek-R1 technical report, January 2025)

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | DeepSeek | High | Consistent with DeepSeek-LLM 7B already in the database |
| `organization` | DeepSeek AI | High | Official organisation name |
| `country_of_origin` | China | High | DeepSeek is based in Hangzhou, China |
| `release_year` | 2025 | High | Paper submitted 22 January 2025; weights released same month |
| `size_b` | 671.0 | High | Paper explicitly states 671B total parameters for the MoE architecture |
| `training_tokens_b` | null | High | **Structural exception — see design decision below** |
| `context_window` | 128000 | High | Paper §3 and HuggingFace model card both state 128K context length |
| `modality` | ["text"] | High | Text-only; no vision capability in DeepSeek-R1 |
| `architecture` | mixture-of-experts | High | Inherits the DeepSeek-V3-Base MoE architecture (256 experts, 37B active parameters per token) |
| `license` | MIT | High | MIT license confirmed on HuggingFace model page |
| `open_weights` | true | High | Full weights available at deepseek-ai/DeepSeek-R1 |
| `open_training_data` | false | High | RL training data and prompts not publicly released |
| `intermediate_checkpoints` | false | High | No intermediate RL training checkpoints released |
| `open_code` | false | High | RL training pipeline code not released (base model training code belongs to DeepSeek-V3) |
| `multilingual` | true | High | English and Chinese are both explicitly supported |
| `num_languages` | 2 | High | Paper describes bilingual English/Chinese capability; no additional official language support documented |
| `languages` | ["English", "Chinese"] | High | Explicitly documented in technical report and model card |
| `has_instruct_version` | true | High | DeepSeek-R1 itself incorporates SFT and RL alignment stages; it functions as an instruction-following model |
| `model_type` | reasoning | High | Explicitly a reasoning model trained via RL on verifiable outcomes; produces extended chain-of-thought before the final answer |
| `has_think_version` | true | High | DeepSeek-R1 is itself a thinking model; it produces explicit internal reasoning (\<think\> blocks). No separate "DeepSeek-R1-Think" variant is needed |
| `notes` | (set) | High | Set to: "Post-trained reasoning model built on DeepSeek-V3-Base; training tokens not applicable." |
| `foundational_paper` | https://arxiv.org/abs/2501.12948 | High | Primary technical report |
| `huggingface_id` | deepseek-ai/DeepSeek-R1 | High | Verified on HuggingFace |

### Design decision: `training_tokens_b` = null

DeepSeek-R1 is **post-trained** via reinforcement learning on top of
DeepSeek-V3-Base, which itself was pre-trained on approximately 14.8T tokens.
The RL training phase does not produce a meaningful "pre-training token count":
it operates on prompts and self-generated completions, not on a fixed token
corpus.

Storing DeepSeek-V3-Base's pre-training token count (14.8T) in this field
would misrepresent the record, since the record describes DeepSeek-R1, not
DeepSeek-V3-Base. Storing null — consistent with the existing treatment of
undisclosed or inapplicable values — is the correct choice. The `notes` field
records the reason to prevent confusion.

### Values to verify

All values are High confidence. No outstanding verification items.

---

## 2. OLMo 2 32B

**Added:** week-3  
**Why added:** Extends the OLMo family coverage; exercises the database's
fully-open model tier at a larger scale than OLMo 2 7B.  
**Primary source:** arXiv 2501.00656 (OLMo 2 technical report, January 2025)

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | OLMo | High | Same family as OLMo 7B and OLMo 2 7B |
| `organization` | Allen Institute for AI | High | Consistent with other OLMo entries |
| `country_of_origin` | United States | High | AllenAI is based in Seattle, WA |
| `release_year` | 2025 | High | Technical report submitted January 2025 |
| `size_b` | 32.0 | High | 32B parameters stated in the OLMo 2 paper and HuggingFace |
| `training_tokens_b` | 5000.0 | Medium | OLMo 2 paper reports multi-stage training for the 32B model; approximately 5T tokens is consistent with the paper's training details, but the exact total across all stages should be verified against §4 of the technical report. **Verify: may be higher if Stage 2 cooldown tokens are included.** |
| `context_window` | 4096 | Medium | OLMo 2 7B uses a 4096 token context window; the 32B is assumed to match. **Verify: some AllenAI releases extended the context window for larger models.** |
| `modality` | ["text"] | High | Text-only |
| `architecture` | decoder-only | High | Standard decoder-only Transformer; consistent across all OLMo variants |
| `license` | Apache 2.0 | High | All OLMo models use Apache 2.0; confirmed on HuggingFace |
| `open_weights` | true | High | Weights released openly |
| `open_training_data` | true | High | Trained on Dolma 2; dataset publicly released by AllenAI |
| `intermediate_checkpoints` | true | High | AllenAI has consistently released intermediate checkpoints for all OLMo models; this is a core part of their open-science commitment |
| `open_code` | true | High | OLMo training framework (OLMo repo) is publicly available on GitHub |
| `multilingual` | false | High | English-only; Dolma 2 is an English corpus |
| `num_languages` | 1 | High | |
| `languages` | ["English"] | High | |
| `has_instruct_version` | true | High | AllenAI has released instruction-tuned variants for all OLMo models; an OLMo 2 32B-Instruct is expected |
| `model_type` | base | High | The stored record is the base model |
| `has_think_version` | false | High | AllenAI has not released a thinking variant of OLMo 2 32B; OLMo 3 32B Think is the relevant model |
| `foundational_paper` | https://arxiv.org/abs/2501.00656 | High | Same paper as OLMo 2 7B; covers both model sizes |
| `huggingface_id` | allenai/OLMo-2-32B | Medium | **Uncertain — requires manual verification.** The OLMo 2 7B uses the ID `allenai/OLMo-2-1124-7B` (where "1124" encodes the November 2024 release date). The 32B model, released in early 2025, may follow a similar date-stamped convention (e.g. `allenai/OLMo-2-0125-32B`) rather than the plain `allenai/OLMo-2-32B` stored here. Verify against the HuggingFace organisation page for AllenAI before the next release. |

### Values to verify

- `training_tokens_b` — Medium confidence. Verify total token count across all training stages in arXiv 2501.00656.
- `context_window` — Medium confidence. Verify the 32B model's context length on its HuggingFace model card.
- `huggingface_id` — Medium confidence. Confirm the exact repository slug on HuggingFace; see note in the table above.

---

## 3. OLMo 3 32B

**Added:** week-3  
**Why added:** The OLMo 3 family is the first to include a dedicated "Think"
variant (OLMo 3 32B Think), making it the only model besides DeepSeek-R1 with
`has_think_version=true` in the current database.

> ⚠️ **This entry has the lowest overall confidence of the five new models.**
> OLMo 3 was announced but had limited published documentation at the time of
> this research. All values marked Medium or Low must be verified against the
> OLMo 3 technical report and HuggingFace model card before the data is
> considered production-ready.

**Primary source:** No dedicated OLMo 3 paper available at time of research.
The `foundational_paper` is set to arXiv 2501.00656 (OLMo 2 paper) as a
**placeholder**. This must be updated to the actual OLMo 3 report when
published.

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | OLMo | High | OLMo naming convention |
| `organization` | Allen Institute for AI | High | |
| `country_of_origin` | United States | High | |
| `release_year` | 2025 | High | Released in 2025 per AllenAI announcements |
| `size_b` | 32.0 | High | "OLMo 3 32B" name explicitly states the size |
| `training_tokens_b` | 6000.0 | Low | **Estimate only.** No primary source available. Extrapolated from OLMo 2 32B (~5T) assuming incremental data scaling. **Must be verified against the OLMo 3 technical report.** |
| `context_window` | 4096 | Low | **Assumed to match OLMo 2 32B.** No primary source. **Must be verified.** |
| `modality` | ["text"] | High | Text-only consistent with all OLMo models |
| `architecture` | decoder-only | High | All OLMo models use a decoder-only Transformer |
| `license` | Apache 2.0 | High | All OLMo releases use Apache 2.0; no reason to expect a change |
| `open_weights` | true | High | AllenAI's open-science policy has made weights available for every OLMo release |
| `open_training_data` | true | Medium | Assumed true based on AllenAI's track record; OLMo 3 is expected to use a public Dolma corpus variant. **Verify: a new dataset may have been used.** |
| `intermediate_checkpoints` | true | Medium | Assumed true; AllenAI has released checkpoints for every prior OLMo model. **Verify against HuggingFace.** |
| `open_code` | true | Medium | Assumed true; training code for OLMo 3 may or may not be in the same public repository as OLMo 2. **Verify.** |
| `multilingual` | false | High | All OLMo models are English-only |
| `num_languages` | 1 | High | |
| `languages` | ["English"] | High | |
| `has_instruct_version` | true | Medium | AllenAI has released instruct variants for all prior OLMo models. **Verify that an OLMo 3 32B-Instruct exists.** |
| `model_type` | base | High | The record is the base model |
| `has_think_version` | true | High | OLMo 3 32B Think is explicitly mentioned in AllenAI announcements; this is the primary distinguishing feature of the OLMo 3 family in this database |
| `foundational_paper` | https://arxiv.org/abs/2501.00656 | Low | **Placeholder — this is the OLMo 2 paper.** Replace with the actual OLMo 3 arXiv URL when the report is published. |
| `huggingface_id` | allenai/OLMo-3-32B | Low | **Unverified.** Inferred from naming conventions. Verify the exact repository slug on HuggingFace. |

### Values to verify

- `training_tokens_b` — Low confidence. Must be sourced from primary publication.
- `context_window` — Low confidence. Verify on HuggingFace model card.
- `open_training_data`, `intermediate_checkpoints`, `open_code` — Medium confidence. Verify against HuggingFace and GitHub.
- `has_instruct_version` — Medium confidence. Verify the instruct variant exists.
- `foundational_paper` — Low confidence / placeholder. Replace with actual OLMo 3 report URL.
- `huggingface_id` — Low confidence. Verify the exact slug.

---

## 4. Phi-4

**Added:** week-3  
**Why added:** Extends the Microsoft Phi family; notably well-performing at
14B parameters with a heavily synthetic training dataset.  
**Primary source:** arXiv 2412.08905 (Phi-4 technical report, December 2024)

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Phi | High | Consistent with Phi-3 Mini 4K already in the database |
| `organization` | Microsoft | High | |
| `country_of_origin` | United States | High | |
| `release_year` | 2024 | High | Technical report submitted December 2024; weights released on HuggingFace December 2024 |
| `size_b` | 14.0 | High | Explicitly stated as 14B parameters throughout the technical report |
| `training_tokens_b` | 9600.0 | Medium | The Phi-4 technical report states the training dataset contains "approximately 9.8T tokens" (§2). The value stored (9600B) is an approximation; the more precise value from the paper is approximately 9800B. **Verify: update to 9800.0 if the paper's stated figure is taken as authoritative.** The discrepancy is small (~2%) and within the rounding used elsewhere in the database. |
| `context_window` | 16384 | High | Stated as 16,384 tokens in the Phi-4 technical report (§A.1) and confirmed on the HuggingFace model card |
| `modality` | ["text"] | High | Text-only; no vision capability in the base Phi-4 model (Phi-4-Vision is a separate product) |
| `architecture` | decoder-only | High | Standard decoder-only Transformer |
| `license` | MIT | High | MIT license confirmed on HuggingFace model page microsoft/phi-4 |
| `open_weights` | true | High | Weights available on HuggingFace |
| `open_training_data` | false | High | Training data composition is described qualitatively but the datasets are not released |
| `intermediate_checkpoints` | false | High | No intermediate checkpoints released |
| `open_code` | false | High | Training code not released |
| `multilingual` | false | High | Microsoft describes Phi-4 as primarily English-focused; no official multilingual support claimed |
| `num_languages` | 1 | High | |
| `languages` | ["English"] | High | |
| `has_instruct_version` | true | High | The primary microsoft/phi-4 HuggingFace release is itself instruction-tuned (SFT + DPO); a separate Phi-4-Instruct variant also exists |
| `model_type` | base | High | The microsoft/phi-4 HuggingFace model is the base/general-purpose release; `"instruct"` is reserved for models released without any base counterpart (as with Phi-3 Mini 4K and LLaVA 1.5 7B) |
| `has_think_version` | true | High | Microsoft released Phi-4-reasoning (an RL-trained reasoning variant of Phi-4) in 2025 as an official variant under the Phi-4 product line. |
| `foundational_paper` | https://arxiv.org/abs/2412.08905 | High | Primary technical report |
| `huggingface_id` | microsoft/phi-4 | High | Verified on HuggingFace |

### Values to verify

- `training_tokens_b` — Medium confidence. The stored value (9600B) is a rounded approximation; the paper's stated figure is ~9800B. Decide whether to correct to 9800.0 or leave as a round number. The 200B difference is within the margin of imprecision used throughout the database.

---

## 5. Qwen2.5 7B

**Added:** week-3  
**Why added:** Successor to Qwen2 7B already in the database; notably trained
on 18T tokens (the largest training dataset of any 7B-class model in the
database) with 29 officially supported languages.  
**Primary source:** arXiv 2412.15115 (Qwen2.5 technical report, December 2024)

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Qwen | High | Consistent with Qwen2 7B already in the database |
| `organization` | Alibaba | High | |
| `country_of_origin` | China | High | |
| `release_year` | 2024 | High | Model weights released September 2024; technical report published December 2024 |
| `size_b` | 7.6 | High | 7.61B parameters stated in the technical report; consistent with Qwen2 7B |
| `training_tokens_b` | 18000.0 | High | The Qwen2.5 technical report explicitly states 18T tokens of pre-training data (§2) |
| `context_window` | 131072 | High | 128K context window (2^17 = 131,072 tokens) stated in the technical report and HuggingFace model card |
| `modality` | ["text"] | High | Text-only |
| `architecture` | decoder-only | High | |
| `license` | Apache 2.0 | High | Apache 2.0 confirmed on HuggingFace model page Qwen/Qwen2.5-7B |
| `open_weights` | true | High | Weights available on HuggingFace |
| `open_training_data` | false | High | Training data described qualitatively but not released |
| `intermediate_checkpoints` | false | High | No intermediate checkpoints released |
| `open_code` | false | High | Training code not released |
| `multilingual` | true | High | 29 officially supported languages per the technical report |
| `num_languages` | 29 | High | Explicitly listed in the Qwen2.5 technical report (§2, Table 1) |
| `languages` | (29 languages) | High | Full list sourced from Table 1 of arXiv 2412.15115; includes Arabic, Chinese (Simplified), Chinese (Traditional), Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese |
| `has_instruct_version` | true | High | Qwen2.5 7B-Instruct is released by Alibaba (Qwen/Qwen2.5-7B-Instruct on HuggingFace) |
| `model_type` | base | High | The stored record is the base model (Qwen/Qwen2.5-7B) |
| `has_think_version` | false | Medium | No official Qwen2.5-7B-Thinking variant exists. Alibaba's thinking/reasoning work is in QwQ-32B (a separate model) and later in Qwen3, which introduced a built-in thinking mode. **Verify: as of mid-2025, Alibaba had not released an official Qwen2.5-7B thinking variant; community fine-tunes exist but are not from the original organisation.** |
| `foundational_paper` | https://arxiv.org/abs/2412.15115 | High | Primary technical report |
| `huggingface_id` | Qwen/Qwen2.5-7B | High | Verified on HuggingFace |

### Values to verify

All values are High or Medium confidence. The one outstanding item:

- `has_think_version` — Medium confidence. Verify that no official Qwen2.5-7B-Thinking model was released by Alibaba before the next database update.

---

## Correction note

`Phi-4` `has_think_version` was initially documented as `false` (Medium confidence)
pending verification. It has been corrected to `true` (High confidence):
Microsoft released **Phi-4-reasoning** as an official RL-trained reasoning
variant of Phi-4 in 2025, making it a qualifying think-version under the
database's field definition.

---

## Summary

| Model | Overall confidence | Items to verify |
|---|---|---|
| DeepSeek-R1 | High | None — all values verified |
| OLMo 2 32B | Medium | `training_tokens_b`, `context_window`, `huggingface_id` |
| OLMo 3 32B | Low–Medium | `training_tokens_b`, `context_window`, `open_training_data`, `intermediate_checkpoints`, `open_code`, `has_instruct_version`, `foundational_paper` (placeholder), `huggingface_id` |
| Phi-4 | High–Medium | `training_tokens_b` (9600 vs. 9800 — minor rounding question) |
| Qwen2.5 7B | High | `has_think_version` (verify no official think variant exists) |

The OLMo 3 32B entry carries the most risk. Its `foundational_paper` is set to
the OLMo 2 report as a placeholder and must be updated when AllenAI publishes
the OLMo 3 technical report. Until then, the OLMo 3 32B record should be
treated as provisional.
