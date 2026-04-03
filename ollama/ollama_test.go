package ollama

import (
	"context"
	"errors"
	"testing"
	"time"

	ollamaapi "github.com/ollama/ollama/api"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type fakeChatAPI struct {
	responses []ollamaapi.ChatResponse
	err       error
}

func (f *fakeChatAPI) Chat(_ context.Context, _ *ollamaapi.ChatRequest, fn ollamaapi.ChatResponseFunc) error {
	if f.err != nil {
		return f.err
	}
	for _, resp := range f.responses {
		if err := fn(resp); err != nil {
			return err
		}
	}
	return nil
}

func TestGenerateContent_unary(t *testing.T) {
	t.Parallel()
	api := &fakeChatAPI{
		responses: []ollamaapi.ChatResponse{{
			Model: "gemma3",
			Message: ollamaapi.Message{
				Role:    "assistant",
				Content: "ok",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics: ollamaapi.Metrics{
				PromptEvalCount: 5,
				EvalCount:       1,
			},
		}},
	}
	m, err := NewWithAPI("gemma3", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	var got int
	for r, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatal(err)
		}
		got++
		if r.Content.Parts[0].Text != "ok" {
			t.Fatalf("text %q", r.Content.Parts[0].Text)
		}
		if !r.TurnComplete {
			t.Fatal("expected TurnComplete")
		}
	}
	if got != 1 {
		t.Fatalf("responses: %d", got)
	}
}

func TestGenerateContent_stream(t *testing.T) {
	t.Parallel()
	api := &fakeChatAPI{
		responses: []ollamaapi.ChatResponse{
			{Message: ollamaapi.Message{Role: "assistant", Content: "hel"}, Done: false},
			{Message: ollamaapi.Message{Role: "assistant", Content: "lo"}, Done: false},
			{
				Message:    ollamaapi.Message{Role: "assistant"},
				Done:       true,
				DoneReason: "stop",
				Metrics: ollamaapi.Metrics{
					PromptEvalCount: 2,
					EvalCount:       2,
				},
			},
		},
	}
	m, err := NewWithAPI("gemma3", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	var partial, final int
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if r.Partial {
			partial++
		} else {
			final++
			if !r.TurnComplete {
				t.Fatal("expected TurnComplete on final")
			}
		}
	}
	if partial < 1 || final != 1 {
		t.Fatalf("partial=%d final=%d", partial, final)
	}
}

func TestGenerateContent_streamToolCalls(t *testing.T) {
	t.Parallel()

	args := ollamaapi.NewToolCallFunctionArguments()
	args.Set("city", "Dublin")

	api := &fakeChatAPI{
		responses: []ollamaapi.ChatResponse{{
			Message: ollamaapi.Message{
				Role: "assistant",
				ToolCalls: []ollamaapi.ToolCall{{
					ID: "call-1",
					Function: ollamaapi.ToolCallFunction{
						Name:      "get_weather",
						Arguments: args,
					},
				}},
			},
			Done:       true,
			DoneReason: "stop",
		}},
	}

	m, err := NewWithAPI("gemma3", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("weather?", "user")},
		Config:   &genai.GenerateContentConfig{},
	}

	var final *model.LLMResponse
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if !r.Partial {
			final = r
		}
	}
	if final == nil {
		t.Fatal("missing final response")
	}
	if final.Content == nil || len(final.Content.Parts) != 1 {
		t.Fatalf("parts: %+v", final.Content)
	}
	fc := final.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatalf("expected function call, got %+v", final.Content.Parts[0])
	}
	if fc.Name != "get_weather" || fc.ID != "call-1" {
		t.Fatalf("function call: %+v", fc)
	}
	if fc.Args["city"] != "Dublin" {
		t.Fatalf("args: %+v", fc.Args)
	}
}

func TestGenerateContent_error(t *testing.T) {
	t.Parallel()
	api := &fakeChatAPI{err: errors.New("connection refused")}
	m, err := NewWithAPI("gemma3", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	var gotErr error
	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil || gotErr.Error() != "connection refused" {
		t.Fatalf("error: %v", gotErr)
	}
}

func TestNew_emptyModelID(t *testing.T) {
	t.Parallel()
	_, err := New("")
	if err == nil {
		t.Fatal("expected error for empty modelID")
	}
}

func TestNewWithAPI_nilAPI(t *testing.T) {
	t.Parallel()
	_, err := NewWithAPI("gemma3", nil)
	if err == nil {
		t.Fatal("expected error for nil API")
	}
}

func TestName(t *testing.T) {
	t.Parallel()
	m, err := NewWithAPI("gemma3", &fakeChatAPI{})
	if err != nil {
		t.Fatal(err)
	}
	if m.Name() != "gemma3" {
		t.Fatalf("name: %q", m.Name())
	}
}

func TestName_nil(t *testing.T) {
	t.Parallel()
	var m *Model
	if m.Name() != "" {
		t.Fatalf("expected empty name for nil model")
	}
}

func TestGenerateContent_usageMetadata(t *testing.T) {
	t.Parallel()
	api := &fakeChatAPI{
		responses: []ollamaapi.ChatResponse{{
			Message: ollamaapi.Message{Role: "assistant", Content: "ok"},
			Done:    true,
			Metrics: ollamaapi.Metrics{
				PromptEvalCount:    10,
				EvalCount:          5,
				TotalDuration:      time.Second,
				PromptEvalDuration: 500 * time.Millisecond,
				EvalDuration:       500 * time.Millisecond,
			},
		}},
	}
	m, err := NewWithAPI("gemma3", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	for r, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatal(err)
		}
		if r.UsageMetadata == nil {
			t.Fatal("expected usage metadata")
		}
		if r.UsageMetadata.PromptTokenCount != 10 {
			t.Fatalf("prompt tokens: %d", r.UsageMetadata.PromptTokenCount)
		}
		if r.UsageMetadata.CandidatesTokenCount != 5 {
			t.Fatalf("candidates tokens: %d", r.UsageMetadata.CandidatesTokenCount)
		}
		if r.UsageMetadata.TotalTokenCount != 15 {
			t.Fatalf("total tokens: %d", r.UsageMetadata.TotalTokenCount)
		}
	}
}
