# ollama-tool-calling example

This example demonstrates tool calling with the Ollama provider using function declarations. It executes a weather tool locally, then sends the tool result back to the model for a final answer.

## Features

- **Tool Definition**: Shows how to define function declarations in `genai.Tool`
- **Direct LLM Flow**: Uses two model calls (tool call + final response)
- **Tool Execution**: Demonstrates handling tool calls from the model
- **Response Processing**: Shows how to extract and process tool calls from LLM responses

## Prerequisites

- [Ollama](https://ollama.com/) running locally or accessible on your network
- An Ollama model installed locally that supports tool calling (e.g. `gemma3:1b`, `llama3.1`, `mistral`)

## Run

```bash
make -C examples/ollama-tool-calling run
```

Or pass a custom question:

```bash
make -C examples/ollama-tool-calling run PROMPT='What is the weather in London?'
```

You can optionally override the default model (`gemma3:1b`) by setting `OLLAMA_MODEL`:

```bash
OLLAMA_MODEL=mistral make -C examples/ollama-tool-calling run
```
