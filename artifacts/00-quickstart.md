# artifacts directory

**propose:** central repository for all pipeline-generated outputs -- processing,
training, evaluation, and deployment artifacts.

**input:** processed data from `data/` drives training and inference.

**output:** frozen encoders, model weights, evaluation reports, and production
deployment assets.

**structure:**
- `preparing/` -- fitted tokenizers, scalers, vectorizers (data-processing outputs)
- `training/` -- checkpoints, model weights, training args (training outputs)
- `evaluation/` -- predictions, metrics, figures (evaluation outputs)
- `exports/` -- onnx, lora adapters, quantized models (deployment assets)

> input data (raw, processed, splits) lives in **`data/`**.
