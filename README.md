<p align="center">
  <img
    src="docs/images/readme-header.jpg"
    alt="adk-go-ollama banner showing Agent Development Kit connected to Ollama"
    width="100%"
  />
</p>

# adk-go-ollama

[Ollama](https://ollama.com/) implementation of the [`model.LLM`](https://pkg.go.dev/google.golang.org/adk/model#LLM) interface for [adk-go](https://github.com/google/adk-go), so you can run agents on local models like Llama 3, Mistral, and others with the same ADK APIs you use for Gemini.

## Requirements

- **Go** 1.25+ (aligned with `google.golang.org/adk`)
- An instance of [Ollama](https://ollama.com/) running locally or accessible over the network
- **[golangci-lint](https://golangci-lint.run/welcome/install/)** if you run `make lint` (uses [.golangci.yaml](.golangci.yaml))

## Install

```bash
go get github.com/craigh33/adk-go-ollama
```

Replace the module path with your fork or published path if you rename the module in `go.mod`.

## Makefile

| Target | Description |
|--------|-------------|
| `make test` | Run unit tests |
| `make build` | Compile all packages |
| `make lint` | Run `golangci-lint run ./...` |
| `make pre-commit-install` | Install pre-commit hooks |

## Contributing / Development

### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com) to enforce code quality and commit hygiene. The following tools must be available on your `PATH` before installing the hooks:

| Tool | Purpose | Install |
|------|---------|---------|
| [pre-commit](https://pre-commit.com) | Hook framework | `brew install pre-commit` |
| [golangci-lint](https://golangci-lint.run/welcome/install/) | Go linter (runs `make lint`) | `brew install golangci-lint` |
| [gitleaks](https://github.com/gitleaks/gitleaks) | Secret / credential scanner | `brew install gitleaks` |

Once the tools are installed, wire the hooks into your local clone:

```bash
make pre-commit-install
```

This installs hooks for both the `pre-commit` stage and the `commit-msg` stage.

#### What the hooks do

| Hook | Stage | Description |
|------|-------|-------------|
| `trailing-whitespace` | pre-commit | Strips trailing whitespace |
| `end-of-file-fixer` | pre-commit | Ensures files end with a newline |
| `check-yaml` | pre-commit | Validates YAML syntax |
| `no-commit-to-branch` | pre-commit | Prevents direct commits to `main` |
| `conventional-pre-commit` | commit-msg | Enforces [Conventional Commits](https://www.conventionalcommits.org/) message format (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`) |
| `golangci-lint` | pre-commit | Runs `make lint` against all Go files |
| `gitleaks` | pre-commit | Scans staged diff for secrets/credentials |

## Usage

```go
ctx := context.Background()
m, err := ollama.New("gemma3")
if err != nil {
    log.Fatal(err)
}

agent, err := llmagent.New(llmagent.Config{
    Name:  "assistant",
    Model: m,
    Instruction: "You are helpful.",
})
// Wire agent into runner.New(...) as usual.
```

`ollama.New` accepts a **model name** as recognized by your Ollama instance. `LLMRequest.Model` can override the model name at runtime.

The [`ollama/mappers`](ollama/mappers/) package holds genai ↔ Ollama conversions (requests, responses, tools, usage). Import it if you need the same mappings outside the default [`ollama`](ollama/) package.

## Examples

Each example has its own `README.md` and `Makefile`:

- [`examples/ollama-chat`](examples/ollama-chat): runner-based chat example.
- [`examples/ollama-tool-calling`](examples/ollama-tool-calling): tool-calling agent example with function declarations.
- [`examples/ollama-imagegen`](examples/ollama-imagegen): image generation via `x/flux2-klein:4b` using ADK's tools.
- [`examples/ollama-stream`](examples/ollama-stream): direct streaming example using `GenerateContent(..., true)`.
- [`examples/ollama-multimodal`](examples/ollama-multimodal): image analysis and multi-image comparison using vision models like `llava`.
- [`examples/ollama-web-ui`](examples/ollama-web-ui): ADK local web UI launcher to interact with your Ollama agent.

To run the examples, you can override the default model using the `OLLAMA_MODEL` environment variable.

```bash
make -C examples/ollama-chat run
```

Run tool calling example:

```bash
make -C examples/ollama-tool-calling run
```

Run streaming example:

```bash
make -C examples/ollama-stream run
```

## How it maps to Ollama

- **Messages**: `genai` roles `user` and `model` map to Ollama `user` and `assistant`. System instructions are sent as `system` messages.
- **Tools**: the mapper converts `GenerateContentConfig.Tools` entries (specifically `FunctionDeclarations`).
- **Streaming**: When ADK uses SSE streaming, the provider streams the response and yields partial outputs, buffering until the final usage metadata is returned.

## Limitations

- **Unsupported features**: Tool variants not supported by Ollama cause a request-time error. Advanced features that don't map cleanly may be ignored or return an explicit ADK error.
- **Multimodal Content**: Ollama natively only supports base64-encoded images. While the ADK can handle arbitrary blobs natively (like PDFs, Documents, or Spreadsheets), the `adk-go-ollama` provider will only process `genai.InlineData` as images via a vision model. Arbitrary file attachments are ignored or rejected by the Ollama API.
- **Image Generation**: Ollama's image generation feature currently only supports macOS and `x/flux2-klein:4b` (based on Flux architectures) and requires a large memory footprint (~12GB). It only supports single image generation requests at a fixed step count natively via `/v1/images/generations`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
