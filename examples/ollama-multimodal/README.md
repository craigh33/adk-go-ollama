# ollama-multimodal example

This example demonstrates how to use vision capable Ollama models to analyze and compare images.

## Prerequisites

- [Ollama](https://ollama.com/) running locally or accessible on your network
- A vision capable Ollama model installed locally (e.g. `llava`, `minicpm-v`)

## Run

```bash
make -C examples/ollama-multimodal run
```

You can optionally override the default model (`llava`) by setting `OLLAMA_MODEL`, or provide a custom image via `IMAGE_URL`:

```bash
OLLAMA_MODEL=minicpm-v IMAGE_URL=https://example.com/cat.jpg make -C examples/ollama-multimodal run
```
