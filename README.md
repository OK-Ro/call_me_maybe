*This project has been created as part of the 42 curriculum by rokuni.*

# Call Me Maybe

## Description

Call Me Maybe turns natural language requests into structured function calls. It reads a list of function definitions and prompts, uses Qwen3-0.6B to select a function and extract its arguments, validates argument names and JSON types against the definitions, and writes JSON results.

## Instructions

Python 3.10 or later and `uv` are required. The local `llm_sdk` package downloads/loads Qwen3-0.6B through Hugging Face, so first use requires network access and enough memory for the model.

```bash
uv sync
uv run python -m src
```

Custom paths are supported:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The default output is `data/output/function_calling_results.json`. The directory is created as needed. `make install`, `make run`, `make debug`, `make clean`, and `make lint` are also available.

### Optional terminal chat

For an interactive session, run:

```bash
uv run python -m visualpart.chat
```

Type a request and the program prints the selected function and arguments. It reuses one loaded model for the session. This optional interface is isolated in `visualpart/`; removing that folder does not affect `uv run python -m src`.

## Algorithm

For each prompt, the program serializes the available function names, descriptions, and parameter types into the model context. It tokenizes the permitted names and greedily selects the highest-scoring next token among names that remain possible. This constrained token traversal stops once only one function remains, avoiding a separate full scoring pass for every name. Function selection is made by the language model from the supplied definitions rather than by keyword rules.

The selected function's schema and description are then included in an argument-extraction prompt. The model generates token by token, up to a bounded token limit. The returned text is parsed as a JSON object. Extra keys are discarded, numeric strings are parsed only when they are valid JSON numbers, and one explicitly quoted input string is preserved verbatim for a function with one string parameter. `FunctionCallValidator` then checks the required parameter names and declared JSON types. Only validated calls are serialized to the output file. JSON serialization disallows non-finite numbers and escapes string content.

This implementation constrains the candidate function name to the provided function list and enforces a schema-valid final object. Argument token generation itself is not grammar-constrained at every step: malformed or schema-invalid model output produces a clear error instead of a result. This limitation is an important distinction from a full token-mask JSON-schema decoder.

## Design decisions

- Function definitions and prompts are parsed into Pydantic models before inference.
- The model is loaded lazily and reused across prompts.
- Function selection greedily follows the highest-scoring token path that matches an available function name.
- Output validation is independent of prompting, so invalid argument types cannot silently enter the results file.
- Model inference and parsing errors stop the run with a readable CLI error; incomplete results are not written as if the run succeeded.

## Performance and reliability

The model is loaded once per batch run or interactive session. Function selection now traverses only candidate-name tokens and stops as soon as one function remains; argument output is capped at 96 tokens per attempt. On this machine, one full run of the five supplied prompts fell from 86.6 seconds before this optimization to 33.7 seconds after it, about 61% faster. Hardware, model cache, prompt count, and output length affect runtime. Five example prompts are not an accuracy benchmark. On a parsing or schema failure the pipeline reports a clear error rather than writing an invalid call.

## Challenges

Subword tokenization makes a function name span multiple tokens, and free-form argument values cannot safely be restricted by a simple fixed token list. The implementation follows only token paths that can form an available name and validates complete argument objects against the runtime schema. Argument generation is not grammar-constrained token by token; malformed or schema-invalid output can still fail and semantically incorrect values remain possible. A full constrained decoder would apply a JSON grammar and type-aware prefix checks during argument generation.

## Testing strategy

During development, `make run` was run against the five supplied prompts and the resulting function names and arguments were checked. The exact quoted-string prompt was also checked separately. `flake8 .` and the required mypy flags pass. Broader checks should cover malformed JSON, missing fields, wrong types, custom function sets, and other hardware; these are not represented by the five example prompts.

## Example output

```json
[
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources and AI use

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)

AI assistance was used to inspect the project scaffold, implement and review the generation/validation flow, and draft this documentation. The project owner should review the generated code and verify behavior with the supplied model and peer review before submission.
