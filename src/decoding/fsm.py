"""Track the structure of a generated function call."""

import json
from enum import Enum, auto
from typing import Any


class FSMState(Enum):
    """Possible states during function-call generation."""

    START = auto()
    GENERATING_NAME = auto()
    GENERATING_PARAMETERS = auto()
    COMPLETE = auto()
    ERROR = auto()


class FunctionCallFSM:
    """Track progress while generating a JSON function call."""

    def __init__(self) -> None:
        """Initialize the FSM."""
        self.reset()

    def reset(self) -> None:
        """Reset the FSM for a new function call."""
        self.state = FSMState.START
        self.generated_text = ""

    def update(self, text: str) -> FSMState:
        """Append generated text and update the current state."""
        if self.state in (FSMState.COMPLETE, FSMState.ERROR):
            return self.state

        self.generated_text += text
        self.state = self._get_state()

        return self.state

    def _get_state(self) -> FSMState:
        """Determine the state from the generated JSON text."""
        text = self.generated_text.strip()

        if not text:
            return FSMState.START

        if not text.startswith("{"):
            return FSMState.ERROR

        try:
            value: Any = json.loads(text)
        except json.JSONDecodeError:
            return self._get_incomplete_state(text)

        if not isinstance(value, dict):
            return FSMState.ERROR

        if "name" not in value:
            return FSMState.GENERATING_NAME

        if not isinstance(value["name"], str):
            return FSMState.ERROR

        if "parameters" not in value:
            return FSMState.GENERATING_PARAMETERS

        if not isinstance(value["parameters"], dict):
            return FSMState.ERROR

        if set(value) != {"name", "parameters"}:
            return FSMState.ERROR

        return FSMState.COMPLETE

    @staticmethod
    def _get_incomplete_state(text: str) -> FSMState:
        """Classify an incomplete function-call JSON document."""
        if '"parameters"' in text:
            return FSMState.GENERATING_PARAMETERS

        if '"name"' in text:
            return FSMState.GENERATING_NAME

        if text.startswith("{"):
            return FSMState.GENERATING_NAME

        return FSMState.START
