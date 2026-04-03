package ollama

import (
	"context"
	"errors"
	"iter"
	"net/http"
	"net/url"
	"strings"

	ollamaapi "github.com/ollama/ollama/api"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-ollama/ollama/mappers"
)

var _ model.LLM = (*Model)(nil)

// errYieldStopped is a sentinel returned from the stream callback when the
// iterator consumer stops early.
var errYieldStopped = errors.New("yield stopped")

// ChatAPI is the subset of Ollama Client operations used by this package.
type ChatAPI interface {
	Chat(ctx context.Context, req *ollamaapi.ChatRequest, fn ollamaapi.ChatResponseFunc) error
}

// Model implements [model.LLM] using Ollama.
type Model struct {
	modelID string
	api     ChatAPI
}

// Option configures [New].
type Option func(*options)

type options struct {
	baseURL    *url.URL
	httpClient *http.Client
}

// WithBaseURL sets the Ollama server URL. Defaults to http://localhost:11434.
func WithBaseURL(rawURL string) Option {
	return func(o *options) {
		u, err := url.Parse(rawURL)
		if err == nil {
			o.baseURL = u
		}
	}
}

// WithHTTPClient sets the HTTP client used to communicate with Ollama.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) {
		o.httpClient = c
	}
}

func defaultBaseURL() *url.URL {
	return &url.URL{Scheme: "http", Host: "localhost:11434"}
}

// New creates a [Model] backed by an Ollama server.
// ModelID is the Ollama model name (e.g. "gemma3", "llama3.3").
func New(modelID string, opts ...Option) (*Model, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("modelID is required")
	}
	o := &options{
		baseURL:    defaultBaseURL(),
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(o)
	}
	client := ollamaapi.NewClient(o.baseURL, o.httpClient)
	return NewWithAPI(modelID, client)
}

// NewWithAPI wires an Ollama [ChatAPI] implementation directly.
func NewWithAPI(modelID string, api ChatAPI) (*Model, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("modelID is required")
	}
	if api == nil {
		return nil, errors.New("nil ChatAPI")
	}
	return &Model{modelID: modelID, api: api}, nil
}

// Name returns the configured model identifier.
func (m *Model) Name() string {
	if m == nil {
		return ""
	}
	return m.modelID
}

// GenerateContent calls Ollama Chat API.
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m == nil || m.api == nil {
			yield(nil, errors.New("nil ollama Model"))
			return
		}
		modelID := m.modelID
		if req != nil && req.Model != "" {
			modelID = req.Model
		}
		if stream {
			m.generateStream(ctx, modelID, req)(yield)
			return
		}
		m.generateUnary(ctx, modelID, req)(yield)
	}
}

func (m *Model) generateUnary(
	ctx context.Context,
	modelID string,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		chatReq, err := mappers.ChatRequestFromLLMRequest(modelID, req, false)
		if err != nil {
			yield(nil, err)
			return
		}
		var finalResp ollamaapi.ChatResponse
		err = m.api.Chat(ctx, chatReq, func(resp ollamaapi.ChatResponse) error {
			finalResp = resp
			return nil
		})
		if err != nil {
			yield(nil, err)
			return
		}
		llmResp := mappers.LLMResponseFromChatResponse(&finalResp, false)
		llmResp.TurnComplete = true
		yield(llmResp, nil)
	}
}

func (m *Model) generateStream(
	ctx context.Context,
	modelID string,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		chatReq, err := mappers.ChatRequestFromLLMRequest(modelID, req, true)
		if err != nil {
			yield(nil, err)
			return
		}

		var textBuf strings.Builder
		var doneResp *ollamaapi.ChatResponse

		err = m.api.Chat(ctx, chatReq, func(resp ollamaapi.ChatResponse) error {
			if resp.Done {
				doneResp = &resp
				return nil
			}

			delta := resp.Message.Content
			if delta == "" {
				return nil
			}
			textBuf.WriteString(delta)

			partial := &model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: delta}},
				},
				Partial: true,
			}
			if !yield(partial, nil) {
				return errYieldStopped
			}
			return nil
		})

		if errors.Is(err, errYieldStopped) {
			return
		}
		if err != nil {
			yield(nil, err)
			return
		}

		final := mappers.FinalLLMResponseFromStream(doneResp, textBuf.String())
		yield(final, nil)
	}
}
