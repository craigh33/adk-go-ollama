# ollama-web-ui example

This example demonstrates how to launch the ADK Web UI locally using the Ollama provider.

## Prerequisites

- [Ollama](https://ollama.com/) running locally or accessible on your network
- An Ollama model installed locally (e.g. `gemma3:1b`)

## Run

To start the web server:

```bash
make -C examples/ollama-web-ui run
```

Then open the printed `http://localhost:<port>` url in your browser to chat with the agent using the ADK UI.

You can optionally override the default model by setting `OLLAMA_MODEL`:

```bash
OLLAMA_MODEL=mistral make -C examples/ollama-web-ui run
```
