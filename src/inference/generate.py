"""propose: run batch inference and generation using trained models.
input: trained model, prompts, sampling parameters.
output: generated outputs or predictions."""


def apply_prompt_template(system_prompt: str, user_input: str, template_type: str) -> str:
    """formats raw input into the target model's chat/prompt template."""
    return f"{system_prompt}\n{user_input}"


def run_batch_generation(model, tokenized_prompts: list, sampling_params: dict) -> list:
    """executes batch inference loop over input test prompts."""
    return []
