# Ollama Image Generation Example

This example demonstrates how an ADK agent can autonomously generate images using the local Ollama image generation feature. It uses a text model for reasoning (e.g., `qwen3`) and uses an agent tool to draw images via the `x/flux2-klein:4b` imagegen model.

## Requirements

1. An Ollama instance running the imagegen feature (currently macOS-only for `x/flux2-klein:4b`).
2. The `x/flux2-klein:4b` model pulled in Ollama:
   ```bash
   ollama pull x/flux2-klein:4b
   ```
3. A text model to control the tool calling:
   ```bash
   ollama pull qwen3:4b
   ```

## Running the Example

```bash
# Uses qwen3:4b as the brain, and x/flux2-klein:4b to generate images
make run

# Or specify a different model entirely:
OLLAMA_MODEL=llama3.3 OLLAMA_IMAGE_MODEL=some-other-flux-model make run
```

## How It Works
- The agent interprets the user's request: *"Can you draw me a sunset over the mountains and save it as sunset.png?"*
- It determines the `generate_image` tool should be used, requesting `sunset.png`.
- The tool calls Ollama's local OpenAI-compatible endpoint (`POST /v1/images/generations`).
- The resulting base64 encoded image is converted and saved locally in the `~/.adk/artifacts` directory.
- The agent tells the user where to find the generated image.
