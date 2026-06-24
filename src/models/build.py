"""propose: initialize model architecture and run training loop.
input: model name from config, optional pretrained weights.
output: initialized model ready for training."""


def load_model(model_name: str, config: dict):
    """loads and initializes the base model architecture."""
    pass


def run_training_loop(model, train_data, val_data, hparams: dict):
    """executes the supervised fine-tuning (sft) or training loop."""
    pass
