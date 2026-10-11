"""Determine allowed token IDs during constrained decoding."""

import json

from src.decoding.fsm import FSMState
from src.decoding.vocabulary import TokenVocabulary
from src.models import FunctionDefinition


class TokenConstraints:
    """Determine which tokens are valid for the next generation step."""

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        functions: list[FunctionDefinition],
    ) -> None:
        """Initialize the vocabulary and available function definitions."""
        self.vocabulary = vocabulary
        self.functions = functions

    def get_allowed_tokens_for_state(
        self,
        state: FSMState,
        generated_text: str,
    ) -> list[int]:
        """Return allowed token IDs for the current generation state."""
        if state in (FSMState.COMPLETE, FSMState.ERROR):
            return []

        if state == FSMState.START:
            return self.vocabulary.get_allowed_token_ids(["{"])

        if state == FSMState.GENERATING_NAME:
            return self._get_function_name_tokens(generated_text)

        if state == FSMState.GENERATING_PARAMETERS:
            return self._get_parameter_tokens(generated_text)

        return []

    def _get_function_name_tokens(
        self,
        generated_text: str,
    ) -> list[int]:
        """Return tokens that can continue an available function name."""
        if '"name"' not in generated_text:
            return self.vocabulary.get_allowed_token_ids(['"'])

        prefix = self._get_incomplete_string(generated_text, '"name"')

        if prefix is None:
            return []

        allowed_ids: set[int] = set()

        for function in self.functions:
            if not function.name.startswith(prefix):
                continue

            remaining = function.name[len(prefix):]

            allowed_ids.update(
                self.vocabulary.get_token_ids_with_prefix(remaining)
            )

        return sorted(allowed_ids)

    def _get_parameter_tokens(
        self,
        generated_text: str,
    ) -> list[int]:
        """Return tokens allowed during parameter generation."""
        if '"parameters"' not in generated_text:
            return self.vocabulary.get_allowed_token_ids(['"'])

        return []

    @staticmethod
    def _get_incomplete_string(
        generated_text: str,
        key: str,
    ) -> str | None:
        """Extract an unfinished string value following a JSON key."""
        key_position = generated_text.find(key)

        if key_position == -1:
            return None

        colon_position = generated_text.find(
            ":",
            key_position + len(key),
        )

        if colon_position == -1:
            return None

        quote_position = generated_text.find('"', colon_position + 1)

        if quote_position == -1:
            return None

        value = generated_text[quote_position + 1:]

        if '"' in value:
            return None

        return value

    @staticmethod
    def _parse_completed_call(
        generated_text: str,
    ) -> dict[str, object] | None:
        """Parse a completed JSON object when possible."""
        try:
            value = json.loads(generated_text)
        except json.JSONDecodeError:
            return None

        if not isinstance(value, dict):
            return None

        return value
