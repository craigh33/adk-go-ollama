package mappers

import (
	"testing"
	"time"

	ollamaapi "github.com/ollama/ollama/api"
	"google.golang.org/genai"
)

func TestLLMResponseFromChatResponse_text(t *testing.T) {
	t.Parallel()
	resp := &ollamaapi.ChatResponse{
		Message:    ollamaapi.Message{Role: "assistant", Content: "hello"},
		Done:       true,
		DoneReason: "stop",
		Metrics: ollamaapi.Metrics{
			PromptEvalCount: 5,
			EvalCount:       3,
		},
	}
	llmResp := LLMResponseFromChatResponse(resp, false)
	if llmResp.Content == nil || len(llmResp.Content.Parts) == 0 {
		t.Fatal("expected content parts")
	}
	if llmResp.Content.Parts[0].Text != "hello" {
		t.Fatalf("text: %q", llmResp.Content.Parts[0].Text)
	}
	if llmResp.FinishReason != genai.FinishReasonStop {
		t.Fatalf("finish reason: %v", llmResp.FinishReason)
	}
	if llmResp.UsageMetadata.PromptTokenCount != 5 {
		t.Fatalf("prompt tokens: %d", llmResp.UsageMetadata.PromptTokenCount)
	}
	if llmResp.UsageMetadata.CandidatesTokenCount != 3 {
		t.Fatalf("candidates tokens: %d", llmResp.UsageMetadata.CandidatesTokenCount)
	}
}

func TestLLMResponseFromChatResponse_toolCalls(t *testing.T) {
	t.Parallel()
	args := ollamaapi.NewToolCallFunctionArguments()
	args.Set("city", "Dublin")

	resp := &ollamaapi.ChatResponse{
		Message: ollamaapi.Message{
			Role: "assistant",
			ToolCalls: []ollamaapi.ToolCall{{
				ID:       "call-1",
				Function: ollamaapi.ToolCallFunction{Name: "get_weather", Arguments: args},
			}},
		},
		Done:       true,
		DoneReason: "stop",
	}
	llmResp := LLMResponseFromChatResponse(resp, false)
	if len(llmResp.Content.Parts) != 1 {
		t.Fatalf("parts: %d", len(llmResp.Content.Parts))
	}
	fc := llmResp.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("expected function call")
	}
	if fc.Name != "get_weather" || fc.ID != "call-1" {
		t.Fatalf("function call: %+v", fc)
	}
	if fc.Args["city"] != "Dublin" {
		t.Fatalf("args: %+v", fc.Args)
	}
}

func TestLLMResponseFromChatResponse_thinking(t *testing.T) {
	t.Parallel()
	resp := &ollamaapi.ChatResponse{
		Message: ollamaapi.Message{
			Role:     "assistant",
			Content:  "answer",
			Thinking: "reasoning...",
		},
		Done:       true,
		DoneReason: "stop",
	}
	llmResp := LLMResponseFromChatResponse(resp, false)
	if len(llmResp.Content.Parts) != 2 {
		t.Fatalf("parts: %d", len(llmResp.Content.Parts))
	}
	if !llmResp.Content.Parts[0].Thought || llmResp.Content.Parts[0].Text != "reasoning..." {
		t.Fatalf("thought part: %+v", llmResp.Content.Parts[0])
	}
	if llmResp.Content.Parts[1].Text != "answer" {
		t.Fatalf("text part: %+v", llmResp.Content.Parts[1])
	}
}

func TestFinishReasonFromDoneReason(t *testing.T) {
	t.Parallel()
	tests := []struct {
		reason string
		want   genai.FinishReason
	}{
		{"stop", genai.FinishReasonStop},
		{"length", genai.FinishReasonMaxTokens},
		{"load", genai.FinishReasonOther},
		{"", genai.FinishReasonStop},
		{"unknown", genai.FinishReasonStop},
	}
	for _, tt := range tests {
		if got := FinishReasonFromDoneReason(tt.reason); got != tt.want {
			t.Errorf("FinishReasonFromDoneReason(%q) = %v, want %v", tt.reason, got, tt.want)
		}
	}
}

func TestUsageFromChatResponse(t *testing.T) {
	t.Parallel()
	resp := &ollamaapi.ChatResponse{
		Metrics: ollamaapi.Metrics{
			PromptEvalCount:    10,
			EvalCount:          5,
			TotalDuration:      time.Second,
			PromptEvalDuration: 500 * time.Millisecond,
			EvalDuration:       500 * time.Millisecond,
		},
	}
	usage := UsageFromChatResponse(resp)
	if usage.PromptTokenCount != 10 {
		t.Fatalf("prompt: %d", usage.PromptTokenCount)
	}
	if usage.CandidatesTokenCount != 5 {
		t.Fatalf("candidates: %d", usage.CandidatesTokenCount)
	}
	if usage.TotalTokenCount != 15 {
		t.Fatalf("total: %d", usage.TotalTokenCount)
	}
}

func TestFinalLLMResponseFromStream(t *testing.T) {
	t.Parallel()
	doneResp := &ollamaapi.ChatResponse{
		DoneReason: "stop",
		Done:       true,
		Metrics: ollamaapi.Metrics{
			PromptEvalCount: 5,
			EvalCount:       10,
		},
	}
	resp := FinalLLMResponseFromStream(doneResp, "hello world")
	if !resp.TurnComplete {
		t.Fatal("expected TurnComplete")
	}
	if resp.Content.Parts[0].Text != "hello world" {
		t.Fatalf("text: %q", resp.Content.Parts[0].Text)
	}
	if resp.FinishReason != genai.FinishReasonStop {
		t.Fatalf("finish reason: %v", resp.FinishReason)
	}
}

func TestLLMResponseFromChatResponse_nil(t *testing.T) {
	t.Parallel()
	resp := LLMResponseFromChatResponse(nil, false)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
}

func TestLLMResponseFromChatResponse_partial(t *testing.T) {
	t.Parallel()
	resp := &ollamaapi.ChatResponse{
		Message: ollamaapi.Message{Role: "assistant", Content: "hel"},
		Done:    false,
	}
	llmResp := LLMResponseFromChatResponse(resp, true)
	if !llmResp.Partial {
		t.Fatal("expected partial=true")
	}
	if llmResp.FinishReason != "" {
		t.Fatalf("expected no finish reason on partial, got %v", llmResp.FinishReason)
	}
}
