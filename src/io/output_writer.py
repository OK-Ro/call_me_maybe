"""Write function-calling results to JSON."""

import json
from pathlib import Path

from src.models import FunctionCall


def write_function_calling_results(
    results: list[FunctionCall],
    output_path: Path,
) -> None:
    """Write function-calling results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            [result.model_dump() for result in results],
            file,
            indent=2,
        )
