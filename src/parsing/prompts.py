import json
from pathlib import Path

from src.models import FunctionCallingPrompt


def validate_prompt(prompt: dict) -> FunctionCallingPrompt:
    """Validate one function-calling prompt."""
    return FunctionCallingPrompt.model_validate(prompt)


def load_function_calling_prompts(
    input_path: Path,
) -> list[FunctionCallingPrompt]:
    """Load and validate function-calling prompts from a JSON file."""
    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)

        if not isinstance(data, list):
            raise ValueError("Prompt input must contain a JSON array.")

        return [validate_prompt(prompt) for prompt in data]

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in prompt input file: " f"{input_path}"
        ) from error
