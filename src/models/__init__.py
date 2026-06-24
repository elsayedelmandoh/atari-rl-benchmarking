"""propose: models package -- model architecture, training harness, and custom implementations.
input: configs, environment, hyperparameters.
output: initialized models, trained policies, evaluation results."""

from src.models.build import load_model, run_training_loop

__all__ = ["load_model", "run_training_loop"]
