"""Guided terminal showcase for the function-calling project."""

import json
import shutil
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

from src.decoding.constraints import TokenConstraints
from src.decoding.generator import FunctionCallGenerator
from src.decoding.vocabulary import TokenVocabulary
from src.llm import LLMClient
from src.models import FunctionDefinition
from src.parsing.functions import load_function_definitions

from visualpart.demo_tools import run_demo_tool

GUIDED_PROMPTS = (
    "What is the sum of 2 and 6?",
    "Greet Maya",
    "Reverse the string 'function calling'",
)


class ChatUI:
    """Draw a colorful, responsive terminal interface with no UI dependency."""

    COLORS = {
        "cyan": "36",
        "blue": "34",
        "purple": "35",
        "green": "32",
        "yellow": "33",
        "red": "31",
        "bold": "1",
        "muted": "2",
    }

    def __init__(self) -> None:
        self.color = sys.stdout.isatty()
        terminal_width = shutil.get_terminal_size((88, 24)).columns
        self.width = max(48, min(terminal_width, 88))
        self.inner = self.width - 4

    def paint(self, value: str, color: str = "cyan") -> str:
        code = self.COLORS.get(color, color)
        return f"\033[{code}m{value}\033[0m" if self.color else value

    def panel(self, title: str, lines: list[str], color: str = "cyan") -> None:
        """Print a wrapped panel with a title and consistent border width."""
        label = f" {title} "
        remaining = max(1, self.width - 3 - len(label))
        print(self.paint("╭─" + label + "─" * remaining + "╮", color))
        for line in lines or [""]:
            for part in self._wrap(line):
                padding = " " * max(0, self.inner - len(part))
                content = self.paint("│ ", color) + part + padding
                print(content + self.paint(" │", color))
        print(self.paint("╰" + "─" * (self.width - 2) + "╯", color))

    def _wrap(self, value: str) -> list[str]:
        """Wrap one content line to the available inner panel width."""
        return textwrap.wrap(
            value, width=self.inner, break_long_words=True,
            break_on_hyphens=False,
        ) or [""]

    def intro(self) -> None:
        """Show the purpose, available tools and demo controls."""
        self.panel(
            "CALL ME MAYBE  /  FUNCTION-CALLING SHOWCASE",
            [
                "A small language model turns your words into a precise call.",
                "It picks a function and arguments, then checks the call.",
                "This demo then runs the sample tool and explains its result.",
                "",
                "  01  ADD NUMBERS       ‘What is the sum of 2 and 6?’",
                "  02  GREET A PERSON    ‘Greet Maya’",
                "  03  REVERSE A STRING  ‘Reverse the string hello’",
                "",
                "Try a prompt or type :demo for a guided tool showcase.",
                "Commands: :how  :tools  :demo  :clear  :quit",
            ],
            "purple",
        )
        flow = "  ● MODEL → TOOL CALL → VALIDATION → RESULT"
        print(self.paint(flow, "bold"))

    def thinking(self, done: threading.Event) -> None:
        """Show live feedback while the model selects and generates a call."""
        if not self.color:
            print("  ◌ Model is selecting a tool and preparing arguments…")
            return

        def animate() -> None:
            frames = ("◐", "◓", "◑", "◒")
            index = 0
            while not done.is_set():
                label = frames[index % len(frames)]
                print(
                    "\r" + self.paint(
                        label + "  Thinking through your request…", "cyan"
                    ),
                    end="", flush=True,
                )
                index += 1
                time.sleep(0.14)
            print("\r" + " " * (self.width - 1) + "\r", end="", flush=True)

        threading.Thread(target=animate, daemon=True).start()

    def clear(self) -> None:
        """Clear a real terminal, or add separation in a redirected session."""
        print("\033[2J\033[H" if self.color else "\n" * 3, end="")


def build_generator() -> tuple[
    FunctionCallGenerator, list[FunctionDefinition]
]:
    """Load model resources once and prepare a reusable call generator."""
    definitions = load_function_definitions(
        Path("data/input/functions_definition.json")
    )
    client = LLMClient()
    vocabulary = TokenVocabulary(Path(client.get_path_to_vocab_file()))
    constraints = TokenConstraints(vocabulary, definitions)
    return FunctionCallGenerator(client, constraints, definitions), definitions


def show_how_it_works(ui: ChatUI) -> None:
    """Explain the real generation and validation path in this project."""
    ui.panel(
        "HOW THE PROJECT WORKS",
        [
            "1  Understand the request",
            "   The prompt and function descriptions are sent to Qwen.",
            "2  Choose a function with the model",
            "   The model scores the available function-name tokens; the code",
            "   restricts the choice to the functions in the input schema.",
            "3  Generate arguments",
            "   The model generates JSON arguments; the parser normalizes",
            "   values and the validator checks required keys and types.",
            "   Missing or mistyped fields get one model repair attempt.",
            "4  Use the structured result",
            "   Batch mode writes the prompt and typed function call as JSON.",
            "   This demo runs three local tools and phrases their results.",
            "",
            "Arguments are parsed and validated after generation. This code",
            "does not constrain every argument token to a JSON grammar during",
            "decoding. The validator catches invalid calls.",
        ],
        "blue",
    )


def show_tools(ui: ChatUI, definitions: list[FunctionDefinition]) -> None:
    """Display names and typed parameters from the project's live schemas."""
    lines: list[str] = []
    for index, function in enumerate(definitions, start=1):
        signature = ", ".join(
            f"{key}: {parameter.type}"
            for key, parameter in function.parameters.items()
        )
        lines.append(f"{index:02d}  {function.name}({signature})")
        lines.append(f"    {function.description}")
    lines.extend([
        "", "Function choices and types load from data/input."
    ])
    ui.panel("AVAILABLE TOOLS  /  LIVE INPUT SCHEMA", lines, "yellow")


def show_result(
    ui: ChatUI,
    prompt: str,
    call: dict[str, Any],
    elapsed: float,
    definitions: list[FunctionDefinition],
) -> None:
    """Render the selected function, typed values and friendly tool result."""
    name = str(call.get("name", "unknown function"))
    parameters = call.get("parameters", call.get("arguments", {}))
    definition = next(
        (item for item in definitions if item.name == name), None
    )
    lines = [f"Tool selected  ·  {name}"]
    if isinstance(parameters, dict) and parameters:
        for key, value in parameters.items():
            kind = ""
            if definition and key in definition.parameters:
                kind = f"  {definition.parameters[key].type}"
            lines.append(f"  {key:<12} {value!r}{kind}")
    else:
        lines.append("  No structured parameters were returned.")
    ui.panel("THE MODEL'S STRUCTURED CALL", lines, "purple")

    answer = run_demo_tool(name, parameters)
    ui.panel("TOOL RESULT  /  ASSISTANT", [answer], "green")
    ui.panel(
        "WHAT JUST HAPPENED",
        [
            f"Your words became a call to {name} with typed parameters.",
            "The tool result came from the local demonstration function.",
            f"Model and validation time: {elapsed:.1f} seconds.",
            "The batch project would save the structured call as JSON.",
        ],
        "muted",
    )


def handle_prompt(
    ui: ChatUI,
    generator: FunctionCallGenerator,
    definitions: list[FunctionDefinition],
    prompt: str,
) -> bool:
    """Run one prompt and render its result; return false on model error."""
    ui.panel("YOU", [prompt], "blue")
    done = threading.Event()
    ui.thinking(done)
    started = time.perf_counter()
    try:
        call = json.loads(generator.generate(prompt))
    except Exception as error:
        ui.panel("MODEL COULD NOT COMPLETE THIS CALL", [str(error)], "red")
        return False
    finally:
        done.set()

    if not isinstance(call, dict):
        ui.panel(
            "MODEL ERROR", ["The response was not a function call."], "red"
        )
        return False
    show_result(ui, prompt, call, time.perf_counter() - started, definitions)
    print()
    return True


def run_guided_demo(
    ui: ChatUI,
    generator: FunctionCallGenerator,
    definitions: list[FunctionDefinition],
) -> None:
    """Run the three sample tools used to introduce the project topic."""
    ui.panel(
        "GUIDED EVALUATION DEMO",
        [
            "Watch three requests become different typed function calls.",
            "Each result is produced by a local sample tool after validation.",
        ],
        "yellow",
    )
    for prompt in GUIDED_PROMPTS:
        handle_prompt(ui, generator, definitions, prompt)


def main() -> int:
    """Start the showcase and accept chat requests and demo commands."""
    ui = ChatUI()
    print(ui.paint("\n  Loading the function-calling model…\n", "muted"))
    try:
        generator, definitions = build_generator()
    except Exception as error:
        print(f"Could not start the model: {error}", file=sys.stderr)
        return 1

    ui.intro()
    while True:
        try:
            prompt = input(ui.paint("\n  YOU  › ", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Showcase ended. Thanks for exploring function calling.")
            return 0

        command = prompt.lower()
        if command in {":quit", ":q", "exit", "quit"}:
            print("\n  Showcase ended. Thanks for exploring function calling.")
            return 0
        if command == ":help":
            ui.intro()
        elif command == ":how":
            show_how_it_works(ui)
        elif command == ":tools":
            show_tools(ui, definitions)
        elif command == ":demo":
            run_guided_demo(ui, generator, definitions)
        elif command == ":clear":
            ui.clear()
            ui.intro()
        elif prompt:
            handle_prompt(ui, generator, definitions, prompt)


if __name__ == "__main__":
    sys.exit(main())
