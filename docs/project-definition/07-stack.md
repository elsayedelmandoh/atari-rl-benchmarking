# stack overview and tooling choices

# expertise
ml, dl, nlp, genai, agentic ai

## core stack
- language: python 3.12
- environment: conda
- entrypoint: app.py
- reusable code: src/
- tests: tests/
- notebooks: notebooks/
- docs: docs/
- data layout: data/raw, data/processed, data/samples, data/models, data/predictions, data/vectorizers
- project manager: gemini 3.5 thinking
- ide: vscode (wsl2)
- architect: claude 4.6 opus
- implementation: glm5, kimi k2.5, qwen 3.6 pro
- code review: claude 4.6 opus
- deployment: hugging face, docker

## common libraries
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pytorch
- scipy
- nltk
- transformers

## project rules
- keep dependencies minimal until a feature needs them
- add a library only when it supports a concrete use case
- keep notebooks for exploration and move stable logic into src/
- keep tests deterministic and repeatable
