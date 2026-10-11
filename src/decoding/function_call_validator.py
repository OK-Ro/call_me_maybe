"""Validate generated function calls against their definitions."""

from pydantic import BaseModel
from typing import Any
from src.models import FunctionDefinition


class FunctionCallValidator(BaseModel):
    """Check function names, parameter names, and argument types."""

    functions: list[FunctionDefinition]

    def is_valid_function_name(self, function_name: str) -> bool:
        """Return whether the function name is available."""
        return any(
            function.name == function_name for function in self.functions
        )

    def is_valid_function_call(
        self,
        function_name: str,
        parameters: dict[str, object],
    ) -> bool:
        """Validate the function name, parameter names, and parameter types."""
        function = next(
            (item for item in self.functions if item.name == function_name),
            None,
        )

        if function is None:
            return False

        if set(parameters) != set(function.parameters):
            return False

        for parameter_name, parameter in function.parameters.items():
            value = parameters[parameter_name]
            if not self._matches_type(value, parameter.type):
                return False

        return True

    @staticmethod
    def _matches_type(value: Any, expected_type: str) -> bool:
        """Return whether a value matches its expected type."""
        if expected_type == "string":
            return isinstance(value, str)

        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(
                value, bool
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
