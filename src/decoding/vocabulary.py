"""Token vocabulary lookup utilities for constrained decoding."""

import json
from pathlib import Path


class TokenVocabulary:
    """Load a token vocabulary and provide token-to-ID lookups."""

    def __init__(self, vocab_path: Path) -> None:
        with vocab_path.open(encoding="utf-8") as file:
            self.token_to_id: dict[str, int] = json.load(file)

        self.id_to_token: dict[int, str] = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

    def get_token_id(self, token: str) -> int:
        """Return the ID of a vocabulary token."""
        return self.token_to_id[token]

    def get_token(self, token_id: int) -> str:
        """Return the token string for an ID."""
        return self.id_to_token[token_id]

    def get_allowed_token_ids(self, allowed_tokens: list[str]) -> list[int]:
        """Return IDs for tokens present in the vocabulary."""
        return [
            self.token_to_id[token]
            for token in allowed_tokens
            if token in self.token_to_id
        ]

    def get_token_ids_with_prefix(self, prefix: str) -> list[int]:
        """Return IDs of tokens whose strings start with a prefix."""
        return [
            token_id
            for token, token_id in self.token_to_id.items()
            if token.startswith(prefix)
        ]
