from typing import Sequence, cast

from llm_sdk import Small_LLM_Model


class LLMClient:
    """Manage model access and provide tokenization and inference methods.

    The model is loaded only when first needed. This avoids unnecessary
    startup costs and allows the same model instance to be reused.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the client without immediately loading the model.

        Args:
            model_name: The identifier of the language model to load.
        """
        self._model_name = model_name
        self._model: Small_LLM_Model | None = None

    @property
    def loaded(self) -> bool:
        """Return whether the model has been initialized."""
        return self._model is not None

    @property
    def model(self) -> Small_LLM_Model:
        """Return the loaded model, initializing it on first access.

        Lazy loading avoids spending time and memory on model initialization
        until inference or tokenization is actually required.
        """
        if self._model is None:
            try:
                self._model = Small_LLM_Model(self._model_name)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to load model '{self._model_name}': {error}"
                ) from error

        return self._model

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs.

        The SDK returns a tensor with a batch dimension. Since this project
        processes one prompt at a time, the first batch is converted to a
        regular Python list for use by the decoding components.
        """
        return cast(list[int], self.model.encode(text)[0].tolist())

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back into readable text.

        Accepting a sequence allows lists and other integer sequences to
        be passed in. A list is supplied to match the SDK interface.
        """
        return cast(str, self.model.decode(list(token_ids)))

    def get_logits(self, input_ids: Sequence[int]) -> list[float]:
        """Get scores for the next possible token.

        The model returns one logit per vocabulary token. Constrained
        decoding will use these scores to choose the highest-scoring
        token that is allowed by the current constraints.
        """
        return cast(
            list[float],
            self.model.get_logits_from_input_ids(list(input_ids)),
        )

    def get_path_to_vocab_file(self) -> str:
        """Return the vocabulary file path.

        The vocabulary maps token representations to token IDs and helps
        the constraint system determine which tokens are valid next.
        """
        return cast(str, self.model.get_path_to_vocab_file())

    def get_path_to_merges_file(self) -> str:
        """Return the tokenizer merges file path.

        Merge rules describe how subword tokens are combined by the
        tokenizer. They may be useful when inspecting tokenization.
        """
        return cast(str, self.model.get_path_to_merges_file())

    def get_path_to_tokenizer_file(self) -> str:
        """Return the tokenizer configuration file path.

        This exposes the tokenizer data for inspection without directly
        importing or using the SDK's internal tokenizer implementation.
        """
        return cast(str, self.model.get_path_to_tokenizer_file())
