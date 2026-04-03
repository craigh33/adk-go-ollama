# ollama-stream example

This example demonstrates streaming text output using the Ollama provider directly by calling `GenerateContent(..., true)`.

## Prerequisites

- [Ollama](https://ollama.com/) running locally or accessible on your network
- An Ollama model installed locally (e.g. `gemma3`)

## Run

```bash
make -C examples/ollama-stream run
```

Or pass a custom prompt:

```bash
make -C examples/ollama-stream run PROMPT='Explain what streaming output is in one sentence.'
```

You can optionally override the default model (`gemma3`) by setting `OLLAMA_MODEL`:

```bash
OLLAMA_MODEL=llama3.1 make -C examples/ollama-stream run
```
