"""Safe local implementations for the examples shown in the chat demo."""

from typing import Any, TypeGuard


def run_demo_tool(name: Any, arguments: Any) -> str:
    """Run one of the explicitly supported demo tools and format its reply."""
    if not isinstance(arguments, dict):
        return (
            "Tool arguments must be an object; I couldn't run this call."
        )

    if name == "fn_add_numbers":
        left, right = arguments.get("a"), arguments.get("b")
        if not _number(left) or not _number(right):
            return (
                "Both arguments must be numbers; I couldn't add these values."
            )
        total = left + right
        return f"The sum of {left:g} and {right:g} is {total:g}."
    if name == "fn_greet":
        person = arguments.get("name")
        if not isinstance(person, str):
            return "I couldn't greet you because the name wasn't text."
        return f"Hello, {person}! Nice to meet you."
    if name == "fn_reverse_string":
        value = arguments.get("s", arguments.get("value"))
        if not isinstance(value, str):
            return "I couldn't reverse that because the value wasn't text."
        return f"The reversed string is “{value[::-1]}”."

    return (
        f"I prepared the {name} tool call, but this demo doesn't include "
        "a local implementation for it."
    )


def _number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
