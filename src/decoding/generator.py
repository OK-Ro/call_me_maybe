"""Generate schema-checked function calls with model-scored choices."""

import json
import re
from typing import Any

from src.decoding.constraints import TokenConstraints
from src.decoding.function_call_validator import FunctionCallValidator
from src.llm import LLMClient
from src.models import FunctionDefinition


class FunctionCallGenerator:
    """Use model likelihoods to select a function and extract its arguments."""

    def __init__(
        self,
        client: LLMClient,
        constraints: TokenConstraints,
        functions: list[FunctionDefinition],
        max_new_tokens: int = 96,
    ) -> None:
        """Store model, schema and output limit for generation."""
        self.client = client
        self.constraints = constraints
        self.functions = functions
        self.max_new_tokens = max_new_tokens
        self.validator = FunctionCallValidator(functions=functions)

    def select_function_name(self, prompt: str) -> str:
        """Select the function with the highest model likelihood."""
        if not self.functions:
            raise ValueError("No function definitions were provided.")
        context = self._context(prompt)
        context_ids = self.client.encode(context)
        candidates: dict[str, list[int]] = {}
        for function in self.functions:
            candidate_ids = self.client.encode(function.name)
            if candidate_ids:
                candidates[function.name] = candidate_ids
        if not candidates:
            raise ValueError("Could not score any available function names.")

        selected_ids: list[int] = []
        while candidates:
            complete = [
                name
                for name, token_ids in candidates.items()
                if len(token_ids) == len(selected_ids)
            ]
            continuing = {
                name: token_ids
                for name, token_ids in candidates.items()
                if len(token_ids) > len(selected_ids)
            }
            if complete and not continuing:
                return complete[0]
            if len(candidates) == 1:
                return next(iter(candidates))
            if complete:
                return self._score_function_candidates(
                    context_ids,
                    candidates,
                )

            logits = self.client.get_logits(context_ids + selected_ids)
            next_token_ids = {
                token_ids[len(selected_ids)]
                for token_ids in continuing.values()
            }
            if any(token_id >= len(logits) for token_id in next_token_ids):
                raise ValueError("Model returned an incomplete logit vector.")
            next_token = max(next_token_ids, key=logits.__getitem__)
            selected_ids.append(next_token)
            candidates = {
                name: token_ids
                for name, token_ids in candidates.items()
                if len(token_ids) >= len(selected_ids)
                and token_ids[:len(selected_ids)] == selected_ids
            }
        raise ValueError("Could not select an available function name.")

    def _score_function_candidates(
        self,
        context_ids: list[int],
        candidates: dict[str, list[int]],
    ) -> str:
        """Resolve the rare case where one name prefixes another."""
        ranked: list[tuple[float, str]] = []
        cache: dict[tuple[int, ...], list[float]] = {}
        for name, candidate_ids in candidates.items():
            ids = list(context_ids)
            score = 0.0
            for token_id in candidate_ids:
                prefix = tuple(ids)
                if prefix not in cache:
                    cache[prefix] = self.client.get_logits(ids)
                logits = cache[prefix]
                if token_id >= len(logits):
                    raise ValueError(
                        "Model returned an incomplete logit vector."
                    )
                score += logits[token_id]
                ids.append(token_id)
            ranked.append((score / len(candidate_ids), name))
        return max(ranked, key=lambda item: item[0])[1]

    def generate(self, prompt: str) -> str:
        """Generate a call and verify its parameters against the schema."""
        function_name = self.select_function_name(prompt)
        function = next(
            item for item in self.functions if item.name == function_name
        )
        request = self._arguments_prompt(prompt, function)
        token_ids = self.client.encode(request)
        generated: list[int] = []
        for _ in range(self.max_new_tokens):
            logits = self.client.get_logits(token_ids + generated)
            token_id = max(range(len(logits)), key=logits.__getitem__)
            generated.append(token_id)
            text = self.client.decode(generated).strip()
            parsed = self._parse_arguments(text)
            if parsed is not None:
                parsed = self._normalize_parameters(parsed, function)
                parsed = self._preserve_quoted_argument(
                    prompt,
                    function,
                    parsed,
                )
                if not self.validator.is_valid_function_call(
                    function_name,
                    parsed,
                ):
                    missing_or_invalid = [
                        key
                        for key, definition in function.parameters.items()
                        if key not in parsed
                        or not self._matches_declared_type(
                            parsed[key],
                            definition.type,
                        )
                    ]
                    repair = self._repair_arguments(
                        prompt,
                        function,
                        missing_or_invalid,
                    )
                    parsed.update(
                        {
                            key: repair[key]
                            for key in missing_or_invalid
                            if key in repair
                        }
                    )
                    parsed = self._preserve_quoted_argument(
                        prompt,
                        function,
                        parsed,
                    )
                    if not self.validator.is_valid_function_call(
                        function_name,
                        parsed,
                    ):
                        raise ValueError(
                            "Model arguments do not match "
                            f"{function_name} for request {prompt!r}."
                        )
                return json.dumps(
                    {"name": function_name, "parameters": parsed},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
        raise ValueError(
            f"Model did not produce valid arguments for {function_name} "
            f"within {self.max_new_tokens} tokens."
        )

    def _repair_arguments(
        self,
        prompt: str,
        function: FunctionDefinition,
        keys: list[str],
    ) -> dict[str, Any]:
        """Ask the model to regenerate missing or mistyped schema fields."""
        if not keys:
            return {}
        schema = {
            key: function.parameters[key].type
            for key in keys
        }
        request = (
            "Extract only the requested argument fields from the request. "
            "Return one JSON object with exactly these keys and proper JSON "
            "types. Do not add a result or explanation.\n"
            f"User request: {prompt}\nFunction: {function.name}\n"
            f"Required fields and types: {json.dumps(schema)}\nJSON:"
        )
        input_ids = self.client.encode(request)
        generated: list[int] = []
        for _ in range(self.max_new_tokens):
            logits = self.client.get_logits(input_ids + generated)
            token_id = max(range(len(logits)), key=logits.__getitem__)
            generated.append(token_id)
            parsed = self._parse_arguments(
                self.client.decode(generated).strip()
            )
            if parsed is not None and all(key in parsed for key in keys):
                return self._normalize_parameters(parsed, function)
        return {}

    def _context(self, prompt: str) -> str:
        """Build a function-selection context from the prompt and schemas."""
        choices = [
            f"{item.name}: {item.description} Parameters: "
            + ", ".join(
                f"{key} ({value.type})"
                for key, value in item.parameters.items()
            )
            for item in self.functions
        ]
        return (
            "Choose the best available function.\n"
            + "\n".join(choices)
            + f"\nRequest: {prompt}\nFunction:"
        )

    @staticmethod
    def _arguments_prompt(prompt: str, function: FunctionDefinition) -> str:
        """Ask the model for only a JSON object of typed arguments."""
        schema = {
            key: value.type for key, value in function.parameters.items()
        }
        return (
            f"Extract JSON arguments for {function.name}. Required keys and "
            f"types: {json.dumps(schema)}. Use the exact input for string "
            f"values and do not add keys.\nRequest: {prompt}\nArguments:"
        )

    @staticmethod
    def _parse_arguments(text: str) -> dict[str, Any] | None:
        """Parse a JSON object, tolerating a surrounding code fence."""
        candidate = text
        if "```" in candidate:
            candidate = candidate.split("```", 1)[-1]
            if candidate.lstrip().startswith("json"):
                candidate = candidate.lstrip()[4:]
            candidate = candidate.split("```", 1)[0]
        start = candidate.find("{")
        if start < 0:
            return None
        try:
            value = json.JSONDecoder().raw_decode(candidate[start:])[0]
        except (json.JSONDecodeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    @classmethod
    def _normalize_parameters(
        cls,
        parameters: dict[str, Any],
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """Project output onto the schema and parse typed JSON literals."""
        normalized: dict[str, Any] = {}
        for key, definition in function.parameters.items():
            if key not in parameters:
                continue
            value = parameters[key]
            if definition.type != "string" and isinstance(value, str):
                try:
                    candidate = json.loads(value)
                except json.JSONDecodeError:
                    candidate = value
                if cls._matches_declared_type(candidate, definition.type):
                    value = candidate
            normalized[key] = value
        return normalized

    @staticmethod
    def _preserve_quoted_argument(
        prompt: str,
        function: FunctionDefinition,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep one quoted string argument verbatim from the prompt."""
        string_keys = [
            key
            for key, definition in function.parameters.items()
            if definition.type == "string"
        ]
        quoted_values = list(
            re.finditer(
                r"'((?:\\.|[^'\\])*)'|\"((?:\\.|[^\"\\])*)\"|`([^`]*)`",
                prompt,
            )
        )
        if len(string_keys) != 1 or len(quoted_values) != 1:
            return parameters
        value = next(
            part
            for part in quoted_values[0].groups()
            if part is not None
        )
        key = string_keys[0]
        parameters[key] = value.replace("\\'", "'").replace('\\"', '"')
        return parameters

    @staticmethod
    def _matches_declared_type(value: Any, expected_type: str) -> bool:
        """Check a parsed value against one supported JSON parameter type."""
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(
                value,
                bool,
            )
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "null":
            return value is None
        return False
