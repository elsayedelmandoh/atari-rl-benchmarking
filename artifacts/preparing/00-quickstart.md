# preparing

**propose:** store fitted data-processing artifacts needed during inference.

**examples:**
- `tokenizers/` -- fitted tokenizer vocab and config files
- `vectorizers/` -- fitted tf-idf, countvectorizer, or embedding transforms

**input:** processed data from `data/processed/` drives fitting.

**output:** serialized transforms consumed by training and inference pipelines.
