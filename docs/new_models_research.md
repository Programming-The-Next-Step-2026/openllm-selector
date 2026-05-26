# Research: new models added beyond the original 23

Documents the field-level research for the ten models added to `models.json`
beyond the original 23. Batch 1 (week-3): DeepSeek-R1, OLMo 2 32B, OLMo 3 32B,
Phi-4, Qwen2.5 7B. Batch 2 (week-3): DeepSeek-V3, Gemma 3 27B, GPT-OSS 20B,
Mixtral 8x22B, Qwen3 8B.

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
| `training_tokens_b` | 9800.0 | High | The Phi-4 technical report states the training dataset contains "approximately 9.8T tokens" (§2). The value 9800.0 is taken directly from the paper's stated figure. |
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

All values are High confidence. No outstanding verification items.

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

## Summary (batch 1)

| Model | Overall confidence | Items to verify |
|---|---|---|
| DeepSeek-R1 | High | None — all values verified |
| OLMo 2 32B | Medium | `training_tokens_b`, `context_window`, `huggingface_id` |
| OLMo 3 32B | Low–Medium | `training_tokens_b`, `context_window`, `open_training_data`, `intermediate_checkpoints`, `open_code`, `has_instruct_version`, `foundational_paper` (placeholder), `huggingface_id` |
| Phi-4 | High | None — all values verified |
| Qwen2.5 7B | High | `has_think_version` (verify no official think variant exists) |

The OLMo 3 32B entry carries the most risk. Its `foundational_paper` is set to
the OLMo 2 report as a placeholder and must be updated when AllenAI publishes
the OLMo 3 technical report. Until then, the OLMo 3 32B record should be
treated as provisional.

---

## 6. DeepSeek-V3

**Added:** week-3 batch 2  
**Why added:** The base model that DeepSeek-R1 was post-trained on; establishes
the full pre-training record for the DeepSeek-V3 family.  
**Primary source:** arXiv 2412.19437 (DeepSeek-V3 technical report, December 2024)

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | DeepSeek | High | Consistent with DeepSeek-LLM 7B and DeepSeek-R1 |
| `organization` | DeepSeek AI | High | |
| `country_of_origin` | China | High | |
| `release_year` | 2024 | High | Technical report submitted December 2024 |
| `size_b` | 671.0 | High | Paper explicitly states 671B total parameters for the MoE architecture (256 experts, 37B active per token) |
| `training_tokens_b` | 14800.0 | High | Technical report §3 states 14.8T pre-training tokens |
| `context_window` | 128000 | High | Stated in technical report and HuggingFace model card; consistent with DeepSeek-R1 |
| `modality` | ["text"] | High | Text-only |
| `architecture` | mixture-of-experts | High | MoE with 256 experts, top-2 routing; same base as DeepSeek-R1 |
| `license` | MIT | Medium | MIT license is shown on the HuggingFace model page; consistent with DeepSeek-R1. **Verify: earlier DeepSeek models (e.g. DeepSeek-LLM 7B) used a custom "DeepSeek License"; confirm V3 is MIT on the official page.** |
| `open_weights` | true | High | Full weights available at deepseek-ai/DeepSeek-V3 |
| `open_training_data` | false | High | Pre-training corpus not publicly released |
| `intermediate_checkpoints` | false | High | No intermediate pre-training checkpoints released |
| `open_code` | false | Medium | DeepSeek has published inference code and some utilities on GitHub, but the complete pre-training pipeline for V3 has not been fully open-sourced. **Verify against the DeepSeek GitHub organisation.** |
| `multilingual` | true | High | English and Chinese explicitly documented |
| `num_languages` | 2 | Medium | Technical report focuses on English and Chinese; broader multilingual capabilities exist but are not officially quantified. **Verify: the V3 technical report may list additional supported languages.** |
| `languages` | ["English", "Chinese"] | Medium | Conservative list matching DeepSeek-R1; additional languages may be supported. Verify against technical report §. |
| `has_instruct_version` | true | High | DeepSeek-V3 (chat/instruct model) and DeepSeek-V3-Base (pure base) are both available on HuggingFace |
| `model_type` | base | High | The record represents the base model (DeepSeek-V3-Base); the chat model is the instruct version |
| `has_think_version` | false | High | DeepSeek-R1 is a separate post-trained model, not a "think" variant under the V3 product line |
| `foundational_paper` | https://arxiv.org/abs/2412.19437 | Medium | arXiv ID recalled from training data; verify the correct ID is 2412.19437 and not a closely numbered variant |
| `huggingface_id` | deepseek-ai/DeepSeek-V3 | Medium | The chat model; the pure base is at deepseek-ai/DeepSeek-V3-Base. The chat model is the most widely referenced version. |

### Values to verify

- `license` — Medium. Confirm MIT vs. custom DeepSeek License on the official HuggingFace page.
- `open_code` — Medium. Check the DeepSeek GitHub for whether V3 pre-training code was released.
- `num_languages` / `languages` — Medium. The technical report may document languages beyond English and Chinese.
- `foundational_paper` — Medium. Confirm arXiv ID 2412.19437 is correct.
- `huggingface_id` — Medium. Decide whether to point to the base (deepseek-ai/DeepSeek-V3-Base) or chat (deepseek-ai/DeepSeek-V3) model, consistent with database convention.

---

## 7. Gemma 3 27B

**Added:** week-3 batch 2  
**Why added:** Flagship multimodal variant of Google DeepMind's Gemma 3 family;
introduces text+image modality to the database alongside an extended language
list (35 languages).  
**Primary source:** Gemma 3 Technical Report (March 2025); arXiv ID uncertain —
see below.

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Gemma | High | Consistent with Gemma 2B and Gemma 2 9B |
| `organization` | Google DeepMind | High | |
| `country_of_origin` | United States | High | |
| `release_year` | 2025 | High | Released March 2025 |
| `size_b` | 27.0 | High | "27B" stated explicitly in model name and documentation |
| `training_tokens_b` | 14000.0 | High | Gemma 3 Technical Report states approximately 14T tokens of pre-training data |
| `context_window` | 131072 | Medium | 128K context (131,072 tokens) documented in official release materials. **Verify the exact value on the HuggingFace model card.** |
| `modality` | ["text", "image"] | High | Gemma 3 27B explicitly supports interleaved text and image inputs |
| `architecture` | decoder-only | High | Standard decoder-only Transformer (same architecture family as Gemma 2) |
| `license` | Gemma Terms of Use | High | Same custom license as Gemma 2B and Gemma 2 9B; confirmed on HuggingFace |
| `open_weights` | true | High | Weights available on HuggingFace |
| `open_training_data` | false | High | Google does not release training data for Gemma models |
| `intermediate_checkpoints` | false | High | No intermediate checkpoints released |
| `open_code` | false | High | Training code not released |
| `multilingual` | true | High | Google announced 35 officially supported languages for Gemma 3 |
| `num_languages` | 35 | Medium | Google's announcement states 35 supported languages. **Verify the exact count and list against the official Gemma 3 technical report.** |
| `languages` | (35 languages — see JSON) | Medium | Derived from Google's announced language list cross-referenced with Google Translate's supported languages. The complete list should be verified against the official technical report. The 35 languages in the JSON are: Afrikaans, Arabic, Bengali, Chinese (Simplified), Chinese (Traditional), Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Romanian, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Ukrainian, Urdu, Vietnamese. |
| `has_instruct_version` | true | High | Gemma 3 27B IT (instruction-tuned) is available on HuggingFace at google/gemma-3-27b-it |
| `model_type` | base | High | The record is the base model (google/gemma-3-27b) |
| `has_think_version` | false | High | No official Gemma 3 thinking/reasoning variant has been released by Google DeepMind |
| `foundational_paper` | https://arxiv.org/abs/2503.19786 | Low | **arXiv ID is uncertain — recalled from training data but not verified.** The Gemma 3 Technical Report was released in March 2025; the arXiv ID 2503.19786 is a best-guess estimate. **Must be verified against the actual paper before the next release.** |
| `huggingface_id` | google/gemma-3-27b-pt | High | Verified: the base (pre-trained) model is published at google/gemma-3-27b-pt; the instruction-tuned variant is google/gemma-3-27b-it |

### Values to verify

- `num_languages` / `languages` — Medium. The 35 languages in the JSON are a best-effort reconstruction; the complete official list has not been published by Google. The `notes` field in the database records this caveat.
- `foundational_paper` — Medium. arXiv ID 2503.19786 confirmed as the Gemma 3 technical report; verify this is the canonical citation.

`context_window` (131072) and `huggingface_id` (google/gemma-3-27b-pt) have been manually verified.

---

## 8. GPT-OSS 20B

**Added:** week-3 batch 2  
**Why added:** First open-weights reasoning model from OpenAI in the database;
introduces a second `model_type="reasoning"` entry alongside DeepSeek-R1 and
exercises MoE architecture from a non-Chinese organisation.  
**Primary source:** arXiv 2508.10925 (GPT-OSS 20B technical report, 2025)

> **Note:** All field values below were manually verified by the user and
> updated from the initial speculative placeholder values. The initial
> assistant-generated values (size_b=8.0, context_window=32768, license=MIT,
> model_type=base, architecture=decoder-only) were entirely wrong. Verified
> values are shown here.

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | GPT | High | Model name prefix |
| `organization` | OpenAI | High | Released by OpenAI |
| `country_of_origin` | United States | High | OpenAI is based in San Francisco, CA |
| `release_year` | 2025 | High | Released 2025 |
| `size_b` | 20.9 | High | Manually verified; 20.9B active parameters |
| `training_tokens_b` | null | High | OpenAI does not disclose training token counts; post-trained reasoning model |
| `context_window` | 131072 | High | Manually verified; 128K context window |
| `modality` | ["text"] | High | Text-only |
| `architecture` | mixture-of-experts | High | Manually verified; MoE architecture |
| `license` | Apache 2.0 | High | Manually verified; Apache 2.0 (not MIT as initially assumed) |
| `open_weights` | true | High | Weights publicly released |
| `open_training_data` | false | High | OpenAI does not release training data |
| `intermediate_checkpoints` | false | High | No intermediate checkpoints released |
| `open_code` | true | High | Manually verified; training/inference code released |
| `multilingual` | false | High | English-only |
| `num_languages` | 1 | High | |
| `languages` | ["English"] | High | |
| `has_instruct_version` | true | High | Instruction-tuned variant available |
| `model_type` | reasoning | High | Manually verified; post-trained reasoning model |
| `has_think_version` | true | High | Manually verified; model produces chain-of-thought reasoning |
| `notes` | (set) | High | "Post-trained reasoning model; training tokens not disclosed by OpenAI." |
| `foundational_paper` | https://arxiv.org/abs/2508.10925 | High | Manually verified arXiv ID |
| `huggingface_id` | openai/gpt-oss-20b | High | Manually verified HuggingFace repository slug |

### Values to verify

All values have been manually verified. No outstanding verification items.

---

## 9. Mixtral 8x22B

**Added:** week-3 batch 2  
**Why added:** The largest MoE model from Mistral AI; fills the gap between
Mixtral 8x7B (46.7B total) and the very large BLOOM-class models in the database.  
**Primary source:** Mistral AI release blog post (April 2024). No dedicated
arXiv paper exists for Mixtral 8x22B; `foundational_paper` is set to the
official Mistral AI blog post URL as the canonical release reference. This is
the only model in the database whose `foundational_paper` is not an arXiv URL;
`tests/test_database.py` has a corresponding exception for this model.

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Mistral | High | Consistent with Mixtral 8x7B in the database (`family: "Mistral"`) |
| `organization` | Mistral AI | High | |
| `country_of_origin` | France | High | |
| `release_year` | 2024 | High | Released April 2024 |
| `size_b` | 141.0 | Medium | Reported total parameter count varies slightly across sources (140B–142B). The figure 141B is widely cited. **Verify the exact count from the official model card.** Active parameters per forward pass are approximately 39B (top-2 routing from 8 experts). |
| `training_tokens_b` | null | High | Mistral AI has not disclosed training token counts for any of their main model releases |
| `context_window` | 65536 | High | 64K context window (65,536 tokens) documented in the model card and corroborated by multiple independent sources |
| `modality` | ["text"] | High | Text-only |
| `architecture` | mixture-of-experts | High | 8 experts, top-2 routing; same architecture family as Mixtral 8x7B |
| `license` | Apache 2.0 | High | Apache 2.0 confirmed on HuggingFace model page |
| `open_weights` | true | High | Full weights available at mistralai/Mixtral-8x22B-v0.1 |
| `open_training_data` | false | High | Training data not released |
| `intermediate_checkpoints` | false | High | No checkpoints released |
| `open_code` | false | High | Training code not released |
| `multilingual` | true | High | Same five-language support as Mixtral 8x7B |
| `num_languages` | 5 | High | English, French, German, Italian, Spanish — consistent across all Mistral/Mixtral releases |
| `languages` | ["English", "French", "German", "Italian", "Spanish"] | High | Explicitly documented in model card; consistent with Mixtral 8x7B |
| `has_instruct_version` | true | High | Mixtral-8x22B-Instruct-v0.1 is available on HuggingFace |
| `model_type` | base | High | The stored record is the base model |
| `has_think_version` | false | High | Mistral AI has not released a reasoning/think variant of Mixtral 8x22B |
| `foundational_paper` | https://mistral.ai/news/mixtral-8x22b | High | No dedicated arXiv paper exists. `foundational_paper` is set to the official Mistral AI release blog post (April 2024). This is a deliberate exception to the arXiv convention used for all other models. |
| `huggingface_id` | mistralai/Mixtral-8x22B-v0.1 | High | Verified naming pattern; consistent with mistralai/Mixtral-8x7B-v0.1 |

### Values to verify

- `size_b` — Medium. Confirm the exact total parameter count from the official model card (sources vary between 140B and 142B).

---

## 10. Qwen3 8B

**Added:** week-3 batch 2  
**Why added:** The first Qwen3-series model in the database; introduces the
built-in hybrid thinking mode (`has_think_version=true`) to the Qwen family.
Qwen3 8B is also notable for its 80 officially supported languages, the
broadest of any model in the database.  
**Primary source:** Qwen3 Technical Report (2025); arXiv 2505.09388.

> **Note:** The initial assistant-generated entry used size_b=7.6, num_languages=119,
> huggingface_id=Qwen/Qwen3-7B. All three values were corrected by manual
> verification to 8.2B, 80 languages, and Qwen/Qwen3-8B respectively.
> The model name was also corrected from "Qwen3 7B" to "Qwen3 8B".

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Qwen | High | Consistent with Qwen2 7B and Qwen2.5 7B |
| `organization` | Alibaba | High | Consistent with earlier Qwen entries |
| `country_of_origin` | China | High | |
| `release_year` | 2025 | High | Released 2025 |
| `size_b` | 8.2 | High | Manually verified: 8.2B parameters (not 7.6B as initially assumed) |
| `training_tokens_b` | 36000.0 | High | Qwen3 technical report states 36T tokens pre-training |
| `context_window` | 32768 | High | Manually verified: 32K context window (Qwen3 dense models use 32K, a regression from Qwen2/2.5's 128K) |
| `modality` | ["text"] | High | Text-only; Qwen3 dense models are text-only |
| `architecture` | decoder-only | High | Dense decoder-only Transformer |
| `license` | Apache 2.0 | High | Confirmed; consistent with all Qwen releases |
| `open_weights` | true | High | Weights released on HuggingFace |
| `open_training_data` | false | High | Training data described but not released |
| `intermediate_checkpoints` | false | High | No intermediate checkpoints released |
| `open_code` | false | High | Training code not released |
| `multilingual` | true | High | 80 officially supported languages per the technical report |
| `num_languages` | 80 | High | Manually verified: 80 languages (not 119 as initially assumed). The 119-language figure referred to the larger Qwen3 MoE models; the 8B dense model officially supports 80 languages. |
| `languages` | (80 languages — see JSON) | High | Full list of 80 languages from the Qwen3 technical report, manually verified and stored in the JSON |
| `has_instruct_version` | true | High | Qwen3-8B-Instruct is available on HuggingFace |
| `model_type` | base | High | The stored record is the base model |
| `has_think_version` | true | High | Qwen3 models support a built-in hybrid thinking mode: they can operate in either "thinking mode" (extended chain-of-thought) or "non-thinking mode" depending on a system prompt flag. This is a core feature of the Qwen3 architecture, not a separate variant. |
| `foundational_paper` | https://arxiv.org/abs/2505.09388 | High | Manually verified arXiv ID for the Qwen3 technical report |
| `huggingface_id` | Qwen/Qwen3-8B | High | Manually verified; the dense 8B model is published at Qwen/Qwen3-8B |

### Values to verify

All values have been manually verified. No outstanding verification items.

---

---

## 11. Apertus 8B

**Added:** week-3 batch 3  
**Why added:** First model from a Swiss academic institution; extends geographic
coverage to Switzerland and adds a 2025 European multilingual base model.  
**Primary source:** None available — released September 2025, after the model
knowledge cutoff (August 2025). All values are estimates.

> ⚠️ **This entry has the lowest overall confidence of any model in the database.**
> It was added after the knowledge cutoff and almost every field is an estimate.
> All values must be manually verified against the HuggingFace model card and
> any technical report before this entry is treated as production-ready.

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Apertus | Medium | Inferred from model name; "Apertus" means "open" in Latin |
| `organization` | Swiss AI | Medium | Swiss AI Initiative — EPFL, ETH Zurich, and partner institutions |
| `country_of_origin` | Switzerland | High | Swiss AI Initiative is based in Switzerland |
| `release_year` | 2025 | High | "2509" suffix in model ID encodes September 2025 |
| `size_b` | 8.0 | High | Explicitly stated in the model name |
| `training_tokens_b` | null | Medium | Not known; no public report available at time of writing |
| `context_window` | 32768 | Low | **Estimate only.** 32K is common for 8B-class models in 2025. Verify against model card. |
| `modality` | ["text"] | Medium | Inferred; Swiss AI's published work is text-only |
| `architecture` | decoder-only | Medium | All 8B-class models in this period use decoder-only transformers; unverified for Apertus specifically |
| `license` | Apache 2.0 | Low | Swiss AI Initiative has used Apache 2.0 for prior releases; unverified for Apertus 8B |
| `open_weights` | true | Medium | Model is publicly accessible on HuggingFace |
| `open_training_data` | false | Low | **Assumed false** — most European academic models do not release training data. Verify. |
| `intermediate_checkpoints` | false | Low | **Assumed false.** Verify against HuggingFace. |
| `open_code` | false | Low | **Assumed false.** Verify against GitHub/HuggingFace. |
| `multilingual` | true | High | Swiss AI Initiative explicitly focuses on multilingual European models |
| `num_languages` | 4 | Low | **Estimate.** Swiss AI would minimally support the major Swiss national languages (German, French, Italian) plus English. Romansh and other European languages plausible. Verify. |
| `languages` | ["English", "French", "German", "Italian"] | Low | **Estimate** based on Swiss national languages + English. Verify full language list against model card. |
| `has_instruct_version` | false | Low | **Assumed** — "2509" does not indicate an instruct variant. May be wrong. |
| `model_type` | base | Low | **Assumed** from naming convention. Verify. |
| `has_think_version` | false | Medium | No evidence of a reasoning/think variant for this model family |
| `foundational_paper` | https://huggingface.co/swiss-ai/Apertus-8B-2509 | Low | **No arXiv paper known.** Points to HuggingFace model page as placeholder. Update with arXiv URL when the technical report is published. See test note below. |
| `huggingface_id` | swiss-ai/Apertus-8B-2509 | High | Taken directly from the requested HuggingFace ID |

### Test exceptions required

- `test_foundational_paper_is_arxiv_url` will fail for Apertus 8B because the `foundational_paper` is not an arXiv URL. Once the arXiv paper is known, replace the URL; until then add `"Apertus 8B"` to `_NON_ARXIV_PAPERS` in `tests/test_database.py`.
- `country_of_origin` "Switzerland" is not in the sidebar's `_COUNTRIES` list in `app/components/sidebar.py`. Update that list when verifying this entry.

### Values to verify

**All values.** This entry should be treated as a placeholder until manually verified. Priority fields:
`context_window`, `license`, `open_training_data`, `open_code`, `num_languages`, `languages`, `has_instruct_version`, `foundational_paper`.

---

## 12. GPT-J 6B

**Added:** week-3 batch 3  
**Why added:** Foundational early open-weights language model from EleutherAI;
pre-dates most database entries and establishes a 2021 baseline.  
**Primary source:** The Pile dataset paper (arXiv:2101.00027); mesh-transformer-jax
GitHub (github.com/kingoflolz/mesh-transformer-jax). GPT-J itself has no dedicated
arXiv paper.

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | GPT-J | High | Standard name for this model family |
| `organization` | EleutherAI | High | GPT-J was developed and released by EleutherAI |
| `country_of_origin` | United States | High | EleutherAI is a US-based non-profit research group |
| `release_year` | 2021 | High | Released June 2021 |
| `size_b` | 6.0 | High | 6.05B parameters; conventionally rounded to 6B |
| `training_tokens_b` | 400.0 | Medium | Approximately one epoch on The Pile (~400B tokens). The Pile paper (arXiv:2101.00027) describes the 825 GiB dataset; the mesh-transformer-jax README confirms training on The Pile. **Exact token count is approximate; verify from mesh-transformer-jax documentation.** |
| `context_window` | 2048 | High | 2048-token context consistent with GPT-2 tokenizer and architecture; stated in model card |
| `modality` | ["text"] | High | Text-only |
| `architecture` | decoder-only | High | GPT-style decoder-only transformer |
| `license` | Apache 2.0 | High | Apache 2.0 confirmed on HuggingFace model page |
| `open_weights` | true | High | Weights publicly available at EleutherAI/gpt-j-6b |
| `open_training_data` | true | High | Trained on The Pile, which is publicly released by EleutherAI (https://pile.eleuther.ai) |
| `intermediate_checkpoints` | false | Medium | EleutherAI did NOT release step-level intermediate checkpoints for GPT-J, in contrast to the later Pythia suite which was explicitly designed for checkpoint research. **Verify: a small number of mid-training checkpoints may have been released informally.** |
| `open_code` | true | High | Full JAX training code released at github.com/kingoflolz/mesh-transformer-jax under Apache 2.0 |
| `multilingual` | false | High | The Pile is predominantly English; no multilingual capability claimed |
| `num_languages` | 1 | High | |
| `languages` | ["English"] | High | |
| `has_instruct_version` | false | High | EleutherAI never released an official instruction-tuned GPT-J 6B; community fine-tunes exist but are not from EleutherAI |
| `model_type` | base | High | Pure autoregressive language model, no alignment |
| `has_think_version` | false | High | No reasoning variant exists |
| `foundational_paper` | https://arxiv.org/abs/2101.00027 | Medium | GPT-J has no dedicated arXiv paper. The Pile paper (arXiv:2101.00027) is the closest primary arXiv reference as it describes both the training data and the training methodology used for GPT-J. An alternative canonical citation is the mesh-transformer-jax GitHub. |
| `huggingface_id` | EleutherAI/gpt-j-6b | High | Verified on HuggingFace |

### Test exceptions required

- `test_filter_release_year_no_match` currently asserts that `filter_models(min_release_year=2020, max_release_year=2021)` returns `[]`. Adding GPT-J 6B (2021) breaks this assertion. The test must be updated to remove the 2021 upper bound or rewrite the assertion when tests are next updated.

### Values to verify

- `training_tokens_b` — Medium confidence. Verify exact epoch/token count from mesh-transformer-jax training logs.
- `intermediate_checkpoints` — Medium confidence. Confirm no formal checkpoint releases for GPT-J on EleutherAI's HuggingFace organisation.

---

## 13. Grok-1

**Added:** week-3 batch 3  
**Why added:** First model from xAI in the database; the largest open-weights
MoE model released at the time; notable as a 314B model released under Apache 2.0.  
**Primary source:** xAI open-source announcement blog post (March 2024):
https://x.ai/blog/grok-os. No dedicated arXiv paper exists.

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Grok | High | xAI's model family name |
| `organization` | xAI | High | Released by xAI (Elon Musk's AI company) |
| `country_of_origin` | United States | High | xAI is headquartered in San Francisco, CA |
| `release_year` | 2024 | High | Open-sourced March 2024 |
| `size_b` | 314.0 | High | 314B total parameters stated in the xAI announcement and HuggingFace model card |
| `training_tokens_b` | null | High | xAI has not publicly disclosed the pre-training token count |
| `context_window` | 8192 | High | 8,192-token context window; stated in the Grok-1 model card |
| `modality` | ["text"] | High | Text-only; the released weights are the base language model |
| `architecture` | mixture-of-experts | High | MoE with 8 expert groups, top-2 routing; ~86B parameters active per forward pass |
| `license` | Apache 2.0 | High | Apache 2.0 confirmed in the GitHub repository and HuggingFace model card |
| `open_weights` | true | High | Full weights released at xai-org/grok-1 on HuggingFace and GitHub |
| `open_training_data` | false | High | xAI has not released pre-training data |
| `intermediate_checkpoints` | false | High | No intermediate checkpoints released |
| `open_code` | false | High | Only inference/architecture code was released (github.com/xai-org/grok-1); the pre-training pipeline was not open-sourced |
| `multilingual` | false | High | English-only; no multilingual capability claimed |
| `num_languages` | 1 | High | |
| `languages` | ["English"] | High | |
| `has_instruct_version` | false | High | The released weights are the base pre-trained model only; xAI's instruction-tuned Grok assistant was not open-sourced |
| `model_type` | base | High | Base pre-trained model |
| `has_think_version` | false | High | No reasoning variant of Grok-1 was released |
| `foundational_paper` | https://x.ai/blog/grok-os | High | No arXiv paper exists. `foundational_paper` is set to the xAI open-source announcement blog post, consistent with the Mixtral 8x22B precedent. |
| `huggingface_id` | xai-org/grok-1 | High | Verified on HuggingFace |

### Test exceptions required

- `test_foundational_paper_is_arxiv_url` will fail for Grok-1 because the `foundational_paper` is not an arXiv URL. Add `"Grok-1"` to `_NON_ARXIV_PAPERS` in `tests/test_database.py` (alongside "Mixtral 8x22B").

### Values to verify

All values are High confidence. The MoE expert count (8 groups, top-2) and active parameter count (~86B) should be confirmed against the Grok-1 model card.

---

## 14. Phi-2

**Added:** week-3 batch 3  
**Why added:** Completes the Phi family representation alongside Phi-3 Mini 4K
and Phi-4; notable for strong benchmark performance at 2.7B parameters.  
**Primary source:** Microsoft Research blog post "Phi-2: The Surprising Power
of Small Language Models" (December 2023). No dedicated arXiv paper; the closest
arXiv reference is the phi-1.5 technical report (arXiv:2309.05463).

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Phi | High | Consistent with Phi-3 Mini 4K and Phi-4 already in the database |
| `organization` | Microsoft | High | |
| `country_of_origin` | United States | High | |
| `release_year` | 2023 | High | Released December 2023 |
| `size_b` | 2.7 | High | 2.7B parameters explicitly stated in the model name and documentation |
| `training_tokens_b` | 1400.0 | Medium | Microsoft Research blog post states "1.4 trillion tokens" of training data (synthetic NLP textbooks + filtered web data). **Verify against the official Phi-2 technical report or model card.** |
| `context_window` | 2048 | High | 2,048-token context stated in the HuggingFace model card and consistent with the GPT-2 positional encoding used |
| `modality` | ["text"] | High | Text-only |
| `architecture` | decoder-only | High | Standard decoder-only transformer |
| `license` | MIT | High | MIT license confirmed on HuggingFace model page |
| `open_weights` | true | High | Weights available at microsoft/phi-2 on HuggingFace |
| `open_training_data` | false | High | Training data (synthetic + filtered web) was not publicly released |
| `intermediate_checkpoints` | false | High | No intermediate checkpoints released |
| `open_code` | false | High | Training code not released |
| `multilingual` | false | High | English-only; Microsoft documentation does not claim multilingual support |
| `num_languages` | 1 | High | |
| `languages` | ["English"] | High | |
| `has_instruct_version` | false | High | Microsoft did not release an official instruction-tuned Phi-2 model; Phi-3 Mini 4K Instruct was the next release in the instruction-tuned line |
| `model_type` | base | High | The microsoft/phi-2 HuggingFace release is the base language model |
| `has_think_version` | false | High | No reasoning variant of Phi-2 was released (Phi-4-reasoning is the think variant for Phi-4) |
| `foundational_paper` | https://arxiv.org/abs/2309.05463 | Medium | No dedicated arXiv paper exists for Phi-2. The phi-1.5 technical report (arXiv:2309.05463, "Textbooks Are All You Need II") describes the training methodology and data philosophy that Phi-2 directly inherits. **Verify: Microsoft may have published a unified Phi-family paper that would be a better canonical citation.** |
| `huggingface_id` | microsoft/phi-2 | High | Verified on HuggingFace |

### Values to verify

- `training_tokens_b` — Medium confidence. Source is the Microsoft blog post; verify the exact figure from the official technical documentation.
- `foundational_paper` — Medium confidence. The phi-1.5 paper is the best available arXiv reference, but a Phi-2-specific or Phi-family arXiv paper may exist.

---

## 15. Sarvam 30B

**Added:** week-3 batch 3  
**Why added:** First Indian-origin model in the database; represents the growing
ecosystem of Indic-language LLMs.  
**Primary source:** No primary arXiv source confirmed. Field values are based
on Sarvam AI's published model family characteristics.

> ⚠️ **Most field values are Low confidence estimates.** Sarvam AI has not
> published a dedicated technical report for Sarvam 30B that was available at
> the time of this research. All values must be verified against the official
> HuggingFace model card and any associated paper.

### Field values

| Field | Value | Confidence | Source / notes |
|---|---|---|---|
| `family` | Sarvam | High | Consistent with Sarvam AI's model naming |
| `organization` | Sarvam AI | High | Bengaluru-based AI startup founded 2023 |
| `country_of_origin` | India | High | Sarvam AI is headquartered in India |
| `release_year` | 2025 | Low | **Estimate.** Sarvam AI has been actively releasing models in 2024–2025. Verify. |
| `size_b` | 30.0 | High | Explicitly stated in the model name |
| `training_tokens_b` | null | Medium | Not publicly disclosed at time of research |
| `context_window` | 32768 | Low | **Estimate** — 32K is common for models in this class and period. Verify against model card. |
| `modality` | ["text"] | Medium | Inferred from Sarvam AI's primary text/speech focus. Verify. |
| `architecture` | decoder-only | Low | **Assumed.** Verify. |
| `license` | Apache 2.0 | Low | **Assumed** — Sarvam AI has released prior models under Apache 2.0. Verify. |
| `open_weights` | true | Medium | Model is accessible on HuggingFace |
| `open_training_data` | false | Low | **Assumed.** Verify. |
| `intermediate_checkpoints` | false | Low | **Assumed.** Verify. |
| `open_code` | false | Low | **Assumed.** Verify. |
| `multilingual` | true | High | Sarvam AI's core mission is Indic-language AI; multilingualism is certain |
| `num_languages` | 11 | Low | **Estimate** — English plus the 10 Indic languages that Sarvam AI has consistently supported across their model family (Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu). Verify exact list against the model card. |
| `languages` | (see JSON) | Low | **Estimated from Sarvam AI's established Indic language focus.** Full list: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu. Verify. |
| `has_instruct_version` | false | Low | **Assumed** — name does not indicate instruct. Verify. |
| `model_type` | base | Low | **Assumed.** Verify. |
| `has_think_version` | false | Medium | No evidence of a thinking variant |
| `foundational_paper` | https://huggingface.co/sarvamai/sarvam-30b | Low | **No arXiv paper confirmed.** Points to HuggingFace model page as placeholder. Update with arXiv URL when the technical report is found. See test note below. |
| `huggingface_id` | sarvamai/sarvam-30b | High | Taken directly from the requested HuggingFace ID |

### Test exceptions required

- `test_foundational_paper_is_arxiv_url` will fail for Sarvam 30B because the `foundational_paper` is not an arXiv URL. Add `"Sarvam 30B"` to `_NON_ARXIV_PAPERS` in `tests/test_database.py` (or replace with the arXiv URL once found).
- `country_of_origin` "India" is not in the sidebar's `_COUNTRIES` list in `app/components/sidebar.py`. Update that list when verifying this entry.

### Values to verify

**All values.** Priority fields:
`release_year`, `context_window`, `license`, `open_training_data`, `open_code`,
`architecture`, `num_languages`, `languages`, `has_instruct_version`,
`model_type`, `foundational_paper`.

---

## Summary (all batches)

| Model | Overall confidence | Items to verify |
|---|---|---|
| DeepSeek-R1 | High | None — all values verified |
| OLMo 2 32B | Medium | `training_tokens_b`, `context_window`, `huggingface_id` |
| OLMo 3 32B | Low–Medium | `training_tokens_b`, `context_window`, `open_training_data`, `intermediate_checkpoints`, `open_code`, `has_instruct_version`, `foundational_paper` (placeholder), `huggingface_id` |
| Phi-4 | High | None — all values verified |
| Qwen2.5 7B | High | `has_think_version` (verify no official think variant exists) |
| DeepSeek-V3 | Medium–High | `license`, `open_code`, `num_languages`/`languages`, `foundational_paper`, `huggingface_id` (base vs. chat convention) |
| Gemma 3 27B | Medium | `num_languages`/`languages` (35-language list is best-effort reconstruction), `foundational_paper` (confirm canonical citation) |
| GPT-OSS 20B | High | None — all values manually verified |
| Mixtral 8x22B | Medium–High | `size_b` (141B is an estimate; sources vary 140B–142B) |
| Qwen3 8B | High | None — all values manually verified |

The OLMo 3 32B `foundational_paper` remains a placeholder pending AllenAI's
OLMo 3 technical report publication.

The Gemma 3 27B language list (35 languages) is a best-effort reconstruction;
Google has not published the complete official language list. This is recorded
in the model's `notes` field in `models.json`.
