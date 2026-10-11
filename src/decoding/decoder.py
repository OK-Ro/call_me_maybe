"""Select valid next tokens during constrained decoding."""

from src.decoding.constraints import TokenConstraints
from src.decoding.fsm import FSMState
from src.llm import LLMClient


class ConstrainedTokenDecoder:
    """Choose the highest-scoring token allowed by the current FSM state."""

    def __init__(
        self,
        client: LLMClient,
        constraints: TokenConstraints,
    ) -> None:
        self.client = client
        self.constraints = constraints

    def select_next_token(
        self,
        input_ids: list[int],
        state: FSMState,
        generated_text: str,
    ) -> int:
        """Select the allowed token with the highest logit."""
        allowed_token_ids = self.constraints.get_allowed_tokens_for_state(
            state,
            generated_text,
        )

        if not allowed_token_ids:
            raise ValueError(f"No allowed tokens for FSM state {state.name}.")

        logits = self.client.get_logits(input_ids)
        return max(allowed_token_ids, key=lambda token_id: logits[token_id])
