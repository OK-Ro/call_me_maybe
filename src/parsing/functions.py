import json
from pathlib import Path

from src.models import FunctionDefinition


def load_function_definitions(
    functions_definition_path: Path,
) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    try:
        with functions_definition_path.open(
            "r",
            encoding="utf-8",
        ) as definition_file:
            data = json.load(definition_file)

        if not isinstance(data, list):
            raise ValueError("Function definitions must contain a JSON array.")

        return [
            FunctionDefinition.model_validate(function_definition)
            for function_definition in data
        ]

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in function definitions file: "
            f"{functions_definition_path}"
        ) from error
