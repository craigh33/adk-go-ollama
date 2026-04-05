package imagegenerator

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"path/filepath"

	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// DefaultModelID is the default Ollama image generation model name.
const DefaultModelID = "x/flux2-klein:4b"

// DefaultBaseURL is the default API URL for the local Ollama instance.
const DefaultBaseURL = "http://localhost:11434"

// Config configures the image generator tool.
type Config struct {
	BaseURL    string // e.g., "http://localhost:11434"
	ModelID    string // e.g., "z-image"
	HTTPClient *http.Client
}

type imageGenTool struct {
	baseURL    string
	modelID    string
	httpClient *http.Client
}

// New creates a new ADK-compatible image generator tool.
func New(cfg Config) (tool.Tool, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	if cfg.ModelID == "" {
		cfg.ModelID = DefaultModelID
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &imageGenTool{
		baseURL:    cfg.BaseURL,
		modelID:    cfg.ModelID,
		httpClient: cfg.HTTPClient,
	}, nil
}

func (t *imageGenTool) Name() string {
	return "generate_image"
}

func (t *imageGenTool) Description() string {
	return "Generates an image from a text prompt using Ollama's local image generation capabilities, and saves it as an artifact."
}

func (t *imageGenTool) IsLongRunning() bool {
	return false
}

func (t *imageGenTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"prompt": {
					Type:        "STRING",
					Description: "The text prompt describing the image to generate",
				},
				"file_name": {
					Type:        "STRING",
					Description: "The filename to save the generated image as (e.g. 'landscape.png')",
				},
				"size": {
					Type:        "STRING",
					Description: "Image dimensions (WxH). Defaults to 1024x1024.",
				},
			},
			Required: []string{"prompt"},
		},
	}
}

// ProcessRequest packs the tool declaration into the LLM request so the model
// can discover and invoke it.
func (t *imageGenTool) ProcessRequest(_ tool.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}
	name := t.Name()
	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = t

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	decl := t.Declaration()
	if decl == nil {
		return nil
	}

	var funcTool *genai.Tool
	for _, gt := range req.Config.Tools {
		if gt != nil && gt.FunctionDeclarations != nil {
			funcTool = gt
			break
		}
	}
	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{decl},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, decl)
	}
	return nil
}

// Run executes the image generation tool: invokes Ollama's /v1/images/generations endpoint,
// then persists the resulting image as an artifact via Artifacts().Save.
func (t *imageGenTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type: %T", args)
	}

	prompt, _ := m["prompt"].(string)
	if prompt == "" {
		return nil, errors.New("prompt is required")
	}
	fileName, _ := m["file_name"].(string)
	if fileName == "" {
		fileName = "generated_image.png"
	}
	mimeType := mime.TypeByExtension(filepath.Ext(fileName))
	if mimeType == "" {
		mimeType = "image/png"
	}
	size, _ := m["size"].(string)
	if size == "" {
		size = "1024x1024"
	}

	reqBody := imageRequest{
		Model:          t.modelID,
		Prompt:         prompt,
		Size:           size,
		ResponseFormat: "b64_json",
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	reqURL, err := url.JoinPath(t.baseURL, "/v1/images/generations")
	if err != nil {
		return nil, fmt.Errorf("invalid base url: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("create http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := t.httpClient.Do(httpReq) //nolint:gosec,nolintlint // trusted BaseURL
	if err != nil {
		return nil, fmt.Errorf("do http request: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(httpResp.Body)
		return nil, fmt.Errorf("ollama API error (status %d): %s", httpResp.StatusCode, string(b))
	}

	var resBody imageResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resBody); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if len(resBody.Data) == 0 {
		return nil, errors.New("ollama returned no image data")
	}

	b64Data := resBody.Data[0].B64JSON
	if b64Data == "" {
		return nil, errors.New("ollama returned empty b64_json image data")
	}
	imageData, err := base64.StdEncoding.DecodeString(b64Data)
	if err != nil {
		return nil, fmt.Errorf("decode b64_json: %w", err)
	}

	part := &genai.Part{
		InlineData: &genai.Blob{
			Data:     imageData,
			MIMEType: mimeType,
		},
	}
	saveResp, err := ctx.Artifacts().Save(ctx, fileName, part)
	if err != nil {
		return nil, fmt.Errorf("save artifact %q: %w", fileName, err)
	}

	return map[string]any{
		"file_name": fileName,
		"version":   saveResp.Version,
		"status":    "success",
	}, nil
}

type imageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	Size           string `json:"size,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
}

type imageResponse struct {
	Created int64 `json:"created"`
	Data    []struct {
		B64JSON string `json:"b64_json"`
	} `json:"data"`
}
