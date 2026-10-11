import argparse
import sys
from pathlib import Path

from src.pipeline import run_pipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate structured function calls from prompts."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the Call Me Maybe application.

    Returns:
        0 when successful, 1 when an error occurs.
    """
    args = parse_arguments()

    try:
        function_count, prompt_count = run_pipeline(
            functions_definition_path=Path(args.functions_definition),
            input_path=Path(args.input),
            output_path=Path(args.output),
        )

        print(
            f"Loaded {function_count} functions and "
            f"{prompt_count} prompts; "
            f"wrote output to {args.output}"
        )

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
