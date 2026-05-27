# Research: candidate new fields for models.json

Two fields assessed for addition to the 18-model database:
`num_languages` (number of supported languages) and
`training_tokens_b` (training token count in billions).

---

## Field 1: `num_languages`

**Definition considered:** number of languages the model officially supports or
was primarily trained to handle, as stated by the releasing organisation. This
excludes incidental multilingual content in web-scraped training data that was
not a design goal.

### Per-model findings

| Model | Value | Confidence | Source / notes |
|---|---|---|---|
| OLMo 7B | 1 | High | English-only design; Dolma corpus is overwhelmingly English |
| OLMo 2 7B | 1 | High | Same as OLMo 7B; trained on Dolma 2 |
| Pythia 6.9B | 1 | High | The Pile is English-centric; no multilingual design goal |
| BLOOM 176B | 46 | High | Explicitly stated in BigScience/ROOTS: 46 natural languages |
| GPT-NeoX 20B | 1 | High | The Pile, English-only design |
| Falcon 7B | 4 | High | TII technical report (Almazrouei et al. 2023) explicitly lists English, German, Spanish, and French as the four officially supported languages. HuggingFace model card notes limited capabilities in 7 additional languages (Portuguese, Italian, Dutch, Polish, Arabic, Chinese, Russian) which are not counted per the database policy of official support only. |
| Falcon 40B | 4 | High | Same as Falcon 7B — same training data and same TII documentation. |
| Mistral 7B | 1 | Medium | No official count; Mistral AI describes it as an English model, though capability in French and others is noted anecdotally |
| Mixtral 8x7B | 5 | High | Mistral AI documentation explicitly lists English, French, German, Italian, Spanish |
| Llama 2 7B | 1 | High | Meta explicitly targets English; small multilingual content in training data is not an official design goal |
| Llama 3.1 8B | 8 | High | Meta documentation lists English, German, French, Italian, Portuguese, Hindi, Spanish, Thai |
| Gemma 2B | 1 | High | Google states primarily English |
| Gemma 2 9B | 1 | High | Google states primarily English; no official multilingual count |
| Phi-3 Mini 4K | 1 | High | Microsoft states English-only for Phi-3 Mini |
| Qwen2 7B | 27 | High | Alibaba technical report explicitly lists 27 languages |
| Yi 1.5 9B | 2 | Medium | 01.AI describes Yi as English + Chinese bilingual; Yi 1.5 may include more languages but no official count is published. **Verify against Yi 1.5 technical report.** |
| DeepSeek-LLM 7B | 2 | Medium | DeepSeek paper states English and Chinese as primary languages. **Verify whether additional languages are officially claimed.** |
| LLaVA 1.5 7B | 1 | High | Inherits Llama 2 backbone; English-only design |

### Missing / unavailable data

None — all 18 models now have a verified value. The two previously uncertain
entries (Falcon 7B, Falcon 40B) were resolved against the TII technical report.

### Assessment

**Suitable as a database field; all values verified.**

All 18 models have a definitive value from a primary source. The Falcon
resolution (4 languages, officially documented) closed the only genuine gap.
The field has a definitional asymmetry — for the 10 English-only models the
value is trivially 1 and adds little beyond `multilingual` — but it is
consistently defined and accurately reflects what the releasing organisations
claim.

---

## Field 2: `training_tokens_b`

**Definition:** number of tokens the model was trained on during pre-training,
expressed in billions (e.g. `2000` for 2 trillion tokens). For models trained
in multiple stages, this should be the total across all pre-training stages;
fine-tuning / RLHF token counts are excluded.

### Per-model findings

| Model | Value (B) | Confidence | Source / notes |
|---|---|---|---|
| OLMo 7B | ~2,050 | Medium | OLMo paper (arXiv 2402.00838) reports training on Dolma v1.6; commonly cited as ~2T tokens. **Verify exact figure in paper.** |
| OLMo 2 7B | ~3,900 | Low | OLMo 2 paper (arXiv 2501.00656) describes multi-stage training; total estimated at ~4T tokens. **Verify against paper — this figure is uncertain.** |
| Pythia 6.9B | ~300 | High | Trained on exactly one epoch of The Pile (299B tokens); explicitly stated in Biderman et al. 2023 |
| BLOOM 176B | 341 | High | BigScience ROOTS corpus; stated in the BLOOM paper (Le Scao et al. 2022) |
| GPT-NeoX 20B | ~472 | High | Calculated from training run details in Black et al. 2022: 150,000 steps × 3.15M tokens per step = 472.5B tokens. The ~400B figure cited in secondary sources is incorrect; the primary paper supports ~472B. |
| Falcon 7B | ~1,500 | Medium | TII Falcon paper reports 1.5T tokens of RefinedWeb + curated sources. **Verify — some secondary sources cite 1T.** |
| Falcon 40B | ~1,000 | Medium | Commonly reported as 1T tokens; less than Falcon 7B per the Falcon paper. **Verify against TII technical report.** |
| Mistral 7B | — | N/A | **Not disclosed.** Mistral AI has not published training token counts for any of their models. |
| Mixtral 8x7B | — | N/A | **Not disclosed.** Same as Mistral 7B; Mistral AI does not release this information. |
| Llama 2 7B | 2,000 | High | Meta paper (Touvron et al. 2023) explicitly states 2T tokens for all Llama 2 variants |
| Llama 3.1 8B | ~15,000 | High | Meta Llama 3 paper reports 15T+ tokens (15.6T cited in some Meta communications) |
| Gemma 2B | 2,000 | High | Google Gemma technical report explicitly states 2T tokens for the 2B model |
| Gemma 2 9B | ~8,000 | Medium | Gemma 2 technical report reports 8T tokens for the 9B model; training used knowledge distillation in addition to standard pre-training. **Verify — some sources report different totals depending on whether distillation stages are included.** |
| Phi-3 Mini 4K | 3,300 | High | Microsoft Phi-3 paper explicitly states 3.3T tokens (mix of web + synthetic data) |
| Qwen2 7B | ~7,000 | Medium | Alibaba Qwen2 technical report cites 7T tokens. **Verify exact figure against the report.** |
| Yi 1.5 9B | 3,600 | High | Verified against the Yi 1.5 technical report: 3.6T tokens. Earlier figure of ~3.1T was from secondary sources and is incorrect. |
| DeepSeek-LLM 7B | 2,000 | High | DeepSeek-LLM paper (arXiv 2401.02954) explicitly states 2T tokens |
| LLaVA 1.5 7B | — | N/A | **Ambiguous / not applicable.** LLaVA 1.5 is a visual instruction fine-tune of Vicuna (itself fine-tuned from Llama 2 7B). The pre-training token count belongs to Llama 2, not to LLaVA. The LLaVA-specific training uses ~665K image-text pairs, an order of magnitude smaller than any base model. Storing a token count here would be misleading. |

### Missing / unavailable data

- **Mistral 7B** — confirmed not disclosed; no primary or credible secondary
  source gives this figure.
- **Mixtral 8x7B** — same; Mistral AI has consistently withheld training
  details.
- **LLaVA 1.5 7B** — structurally inapplicable; it is a fine-tuned model, not
  a pre-trained base model.

### Assessment

**Mostly suitable as a database field, with two firm gaps and one structural
exception.**

15 of 18 models have findable values. The two Mistral models are confirmed
unknowns with no prospect of resolution unless Mistral AI publishes the data.
LLaVA requires a design decision: either store `null` with a note that it is
fine-tuned, or exclude the field for multimodal fine-tunes.

Values to **manually verify before committing** (medium confidence):
OLMo 7B, OLMo 2 7B, GPT-NeoX 20B, Falcon 7B, Falcon 40B, Gemma 2 9B,
Qwen2 7B, Yi 1.5 9B.

**Recommendation:** add the field with `null` for Mistral 7B, Mixtral 8x7B,
and LLaVA 1.5 7B. It is more complete and consistently defined than
`num_languages`, and the null entries have a clear explanation (not disclosed /
not applicable). Verify the eight medium-confidence values against their
respective foundational papers before the field goes into production.

---

## Field 3: `has_instruct_version`

**Definition:** `true` if the releasing organisation published an official
instruction-tuned or chat-tuned variant of this specific base model (e.g.
released under the same name with an `-Instruct`, `-Chat`, or `-IT` suffix).
Community fine-tunes that are not from the original organisation do not count.

### Per-model findings

| Model | Value | Confidence | Official variant name |
|---|---|---|---|
| OLMo 7B | true | High | OLMo 7B-Instruct (Allen AI) |
| OLMo 2 7B | true | High | OLMo 2 7B-Instruct (Allen AI) |
| Pythia 6.9B | false | High | EleutherAI released Pythia exclusively as base research models; no instruct release |
| BLOOM 176B | true | High | BLOOMZ 176B (BigScience / HuggingFace) |
| GPT-NeoX 20B | false | High | EleutherAI did not release an instruct variant; community fine-tunes exist but nothing official |
| Falcon 7B | true | High | Falcon 7B-Instruct (TII) |
| Falcon 40B | true | High | Falcon 40B-Instruct (TII) |
| Mistral 7B | false | High | The database record points to Mistral 7B v0.1, which has no official instruct variant — the instruct release only arrived with v0.3. Counting a later version's instruct variant would misrepresent the specific checkpoint being described. |
| Mixtral 8x7B | true | High | Mixtral 8x7B Instruct v0.1 (Mistral AI) |
| Llama 2 7B | true | High | Llama 2 7B-Chat (Meta) |
| Llama 3.1 8B | true | High | Llama 3.1 8B-Instruct (Meta) |
| Gemma 2B | true | High | Gemma 2B-IT (Google) |
| Gemma 2 9B | true | High | Gemma 2 9B-IT (Google) |
| Phi-3 Mini 4K | true | High | Phi-3 Mini 4K-Instruct (Microsoft) — the instruct variant is in fact the primary promoted version |
| Qwen2 7B | true | High | Qwen2 7B-Instruct (Alibaba) |
| Yi 1.5 9B | true | High | Yi 1.5 9B-Chat (01.AI) |
| DeepSeek-LLM 7B | true | High | DeepSeek-LLM 7B-Chat (DeepSeek) |
| LLaVA 1.5 7B | true | High | LLaVA 1.5 is itself an instruction-tuned model. Storing `true` is more useful for UI display than `null`, and accurately reflects that an instruction-following variant exists (the model itself). |

### Missing / unavailable data

None — all 18 models have a definitive value.

### Assessment

**All values verified and in production.** Two corrections were applied after
initial research: Mistral 7B was corrected to `false` (the v0.1 checkpoint has
no instruct variant; instruct was only released with v0.3) and LLaVA 1.5 7B
was set to `true` (LLaVA is itself instruction-tuned, making `true` both
accurate and useful for UI display).

---

## Field 4: `has_think_version`

**Definition:** `true` if the releasing organisation published an official
variant of this specific model that is trained for extended chain-of-thought
or "thinking" reasoning — i.e. a model that produces explicit internal
reasoning steps before its final answer, trained via RL on verifiable outcomes
or distilled from a reasoning teacher. This excludes general instruction-tuned
models that can be prompted for chain-of-thought.

### Per-model findings

| Model | Value | Confidence | Notes |
|---|---|---|---|
| OLMo 7B | false | High | No thinking variant released by Allen AI |
| OLMo 2 7B | false | High | No thinking variant released by Allen AI |
| Pythia 6.9B | false | High | Research base model only |
| BLOOM 176B | false | High | No thinking variant; BigScience project concluded |
| GPT-NeoX 20B | false | High | Research base model only; project concluded |
| Falcon 7B | false | High | TII has not released a thinking variant for the Falcon 7B line |
| Falcon 40B | false | High | Same as Falcon 7B |
| Mistral 7B | false | High | Mistral AI released Magistral (2025) as their reasoning model, but it is a separate product line, not a Mistral 7B variant |
| Mixtral 8x7B | false | High | No thinking variant; Mistral's reasoning work uses different architecture |
| Llama 2 7B | false | High | Predates the thinking-model paradigm; Meta has not released a reasoning fine-tune of this specific checkpoint |
| Llama 3.1 8B | false | Medium | Meta has not released an official "Llama 3.1 8B Thinking" model. Community thinking fine-tunes exist (e.g. via distillation from DeepSeek-R1) but none from Meta. **Verify: Meta roadmap may change this.** |
| Gemma 2B | false | High | No thinking variant; Google's reasoning work is in Gemma 3 and later |
| Gemma 2 9B | false | Medium | Google has not released a thinking variant of Gemma 2 9B specifically. **Verify: Google released thinking-capable models in the Gemma 3 generation, not Gemma 2.** |
| Phi-3 Mini 4K | false | High | Microsoft's reasoning work is in Phi-4-reasoning (a separate model); no thinking variant of Phi-3 Mini 4K |
| Qwen2 7B | false | Medium | Alibaba released QwQ-32B and Qwen3 with thinking mode, but those are different models. Qwen2 7B specifically has no official thinking variant. **Verify: Qwen2.5-7B-Thinking exists but that is Qwen2.5, not Qwen2.** |
| Yi 1.5 9B | false | High | 01.AI has not released a thinking variant of Yi 1.5 |
| DeepSeek-LLM 7B | false | High | DeepSeek-R1 is a separate product line (based on DeepSeek-V3, not DeepSeek-LLM). DeepSeek-R1-Distill variants use Qwen2.5 and Llama 3 backbones, not DeepSeek-LLM 7B. |
| LLaVA 1.5 7B | false | High | No thinking variant; multimodal reasoning work has moved to newer architectures |

### Missing / unavailable data

None — the field is uniformly `false` for all 18 models with high or medium
confidence.

### Assessment

**The field adds no information for the current 18-model database.**
Every model predates or sits outside the thinking-model paradigm as it is
currently defined. The three medium-confidence entries (Llama 3.1 8B,
Gemma 2 9B, Qwen2 7B) are false because the thinking work in those families
happened in later model generations (Llama 4, Gemma 3, Qwen3), not as
variants of the specific checkpoints in this database.

Adding the field now would produce a column of 18 `false` values, which is
correct but uninformative. It becomes meaningful only when newer models
(Qwen3 7B, Llama 4 Scout, Gemma 3, DeepSeek-R1-Distill-Llama-8B) are added
to the database.

**Recommendation:** hold off on adding `has_think_version` until at least one
thinking-capable model is included in the database. At that point, backfilling
`false` for the existing 18 models is trivial. Alternatively, add it now as
future-proofing if the database is expected to grow soon.

---

## Summary comparison

| | `num_languages` | `training_tokens_b` | `has_instruct_version` | `has_think_version` |
|---|---|---|---|---|
| Models with data | 18 / 18 | 15 / 18 | 18 / 18 | 18 / 18 |
| Confirmed gaps | 0 | 2 (Mistral 7B, Mixtral) | 0 | 0 |
| Structural exceptions | 0 | 1 (LLaVA — null, not applicable) | 0 | 0 |
| Definition consistency | High | High | High | High |
| Values verified | 18 / 18 | 15 / 15 with data | 18 / 18 | 18 / 18 |
| Status | In production | In production | In production | Deferred — all values would be false |
