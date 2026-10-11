# Function-calling showcase

This optional terminal interface helps you demonstrate what the project does:
it takes a natural-language request, asks the local Qwen model to choose a
function and produce typed parameters, checks the structured call, and displays
what a tool could return.

Start it from the project root:

```bash
uv run python -m visualpart.chat
```

After the model loads, try a request or use one of these commands:

- `:demo` runs three guided examples: addition, a greeting, and string reversal.
- `:how` explains the function selection, argument generation, and validation
  stages, including the current implementation's limits.
- `:tools` displays the names and parameter types loaded from the project schema.
- `:clear` refreshes the screen; `:quit` exits.

For example, `What is the sum of 2 and 6?` displays the selected function,
typed arguments, local tool result (`The sum of 2 and 6 is 8.`), and a short
explanation of what happened. The tool implementations are intentionally small
examples, not external services.

The required `uv run python -m src` batch command remains unchanged and writes
machine-readable JSON. This showcase is isolated in `visualpart/` and can be
removed without changing the submission pipeline.
