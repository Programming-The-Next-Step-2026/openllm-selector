# openllm-selector

A tool to help researchers pick the right open LLM for their study.

Choosing the right open LLM for research is hard given the rapidly growing landscape of available models. This package provides a curated database of 18 open LLMs with key characteristics — size, modality, license, openness score, training data availability, intermediate checkpoints, and links to foundational papers — so researchers can filter and compare models without wading through leaderboards and blog posts.

Unlike leaderboards that rank models by benchmark scores, `openllm-selector` focuses on the practical and ethical characteristics that matter for research: how openly the model was released, what data it was trained on, and whether intermediate checkpoints are available for studying training dynamics.

## Installation

```bash
pip install git+https://github.com/Programming-The-Next-Step-2026/openllm-selector.git@week-3
```

## Documentation

See [vignette/tutorial.qmd](vignette/tutorial.qmd) for a full walkthrough of the package, including example usage and a realistic researcher use case.
