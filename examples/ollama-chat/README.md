# ollama-chat example

This example runs a simple ADK runner using the Ollama provider.

## Prerequisites

- [Ollama](https://ollama.com/) running locally or accessible on your network
- An Ollama model installed locally (e.g. `gemma3:1b`)

## Run

```bash
make -C examples/ollama-chat run
```

Or pass a custom prompt:

```bash
make -C examples/ollama-chat run PROMPT='Summarize static typing in one sentence.'
```

You can optionally override the default model (`gemma3:1b`) by setting `OLLAMA_MODEL`:

```bash
OLLAMA_MODEL=llama3 make -C examples/ollama-chat run
```
