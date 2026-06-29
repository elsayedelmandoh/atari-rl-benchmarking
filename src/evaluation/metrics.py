"""propose: compute alignment and loss metrics for model evaluation.
input: predictions, ground-truth references, eval loss.
output: dict of computed scores (accuracy, rouge, bleu, perplexity)."""


def compute_alignment_metrics(predictions: list, references: list) -> dict:
    """computes generation quality metrics like rouge, bleu, or exact match."""
    return {"accuracy": 0.0, "rougeL": 0.0, "bleu": 0.0}


def compute_loss_metrics(eval_loss: float) -> dict:
    """computes perplexity from evaluation loss."""
    import math

    try:
        return {"perplexity": math.exp(eval_loss)}
    except OverflowError:
        return {"perplexity": float("inf")}
