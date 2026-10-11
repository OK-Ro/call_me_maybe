import json
from pathlib import Path

from src.parsing.functions import load_function_definitions
from src.parsing.prompts import load_function_calling_prompts
from src.io.output_writer import write_function_calling_results
from src.decoding.vocabulary import TokenVocabulary
from src.decoding.constraints import TokenConstraints
from src.decoding.generator import FunctionCallGenerator
from src.models import FunctionCall
from src.llm import LLMClient


def run_pipeline(
    functions_definition_path: Path,
    input_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Run the function-calling pipeline."""

    functions = load_function_definitions(functions_definition_path)
    prompts = load_function_calling_prompts(input_path)

    client = LLMClient()
    vocab_path = Path(client.get_path_to_vocab_file())
    vocabulary = TokenVocabulary(vocab_path)
    constraints = TokenConstraints(vocabulary, functions)
    generator = FunctionCallGenerator(client, constraints, functions)

    results: list[FunctionCall] = []

    for prompt in prompts:
        response = generator.generate(prompt.prompt)
        try:
            data = json.loads(response)
        except json.JSONDecodeError as error:
            raise ValueError("The generator returned invalid JSON.") from error

        function_name = data.get("name")
        parameters = data.get("parameters")

        if not isinstance(function_name, str) or not isinstance(
            parameters,
            dict,
        ):
            raise ValueError(
                "The generator returned an invalid function call."
            )

        results.append(
            FunctionCall(
                prompt=prompt.prompt,
                name=function_name,
                parameters=parameters,
            )
        )

    write_function_calling_results(results, output_path)

    return len(functions), len(prompts)
