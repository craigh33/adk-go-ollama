package mappers

import (
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestChatRequestFromLLMRequest_basic(t *testing.T) {
	t.Parallel()
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hello", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	chatReq, err := ChatRequestFromLLMRequest("gemma3", req, false)
	if err != nil {
		t.Fatal(err)
	}
	if chatReq.Model != "gemma3" {
		t.Fatalf("model: %q", chatReq.Model)
	}
	if *chatReq.Stream {
		t.Fatal("expected stream=false")
	}
	if len(chatReq.Messages) == 0 {
		t.Fatal("expected messages")
	}
}

func TestChatRequestFromLLMRequest_nilRequest(t *testing.T) {
	t.Parallel()
	_, err := ChatRequestFromLLMRequest("gemma3", nil, false)
	if err == nil {
		t.Fatal("expected error for nil request")
	}
}

func TestChatRequestFromLLMRequest_systemInstruction(t *testing.T) {
	t.Parallel()
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("you are helpful", "system"),
		},
	}
	chatReq, err := ChatRequestFromLLMRequest("gemma3", req, false)
	if err != nil {
		t.Fatal(err)
	}
	if chatReq.Messages[0].Role != ollamaRoleSystem {
		t.Fatalf("first message role: %q", chatReq.Messages[0].Role)
	}
	if chatReq.Messages[0].Content != "you are helpful" {
		t.Fatalf("system content: %q", chatReq.Messages[0].Content)
	}
}

func TestChatRequestFromLLMRequest_tools(t *testing.T) {
	t.Parallel()
	temp := float32(0.5)
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config: &genai.GenerateContentConfig{
			Temperature: &temp,
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "gets the weather",
				}},
			}},
		},
	}
	chatReq, err := ChatRequestFromLLMRequest("gemma3", req, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(chatReq.Tools) != 1 {
		t.Fatalf("tools: %d", len(chatReq.Tools))
	}
	if chatReq.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("tool name: %q", chatReq.Tools[0].Function.Name)
	}
	if chatReq.Options["temperature"] != float32(0.5) {
		t.Fatalf("temperature: %v", chatReq.Options["temperature"])
	}
}

func TestChatRequestFromLLMRequest_functionResponse(t *testing.T) {
	t.Parallel()
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{
					ID:       "call-1",
					Name:     "get_weather",
					Response: map[string]any{"temp": "15C"},
				},
			}}},
		},
		Config: &genai.GenerateContentConfig{},
	}
	chatReq, err := ChatRequestFromLLMRequest("gemma3", req, false)
	if err != nil {
		t.Fatal(err)
	}
	// Should have: tool message + appended user message (from MaybeAppendUserContent)
	found := false
	for _, msg := range chatReq.Messages {
		if msg.Role == ollamaRoleTool {
			found = true
			if msg.ToolName != "get_weather" {
				t.Fatalf("tool name: %q", msg.ToolName)
			}
		}
	}
	if !found {
		t.Fatal("expected tool message")
	}
}

func TestChatRequestFromLLMRequest_modelContentWithToolCalls(t *testing.T) {
	t.Parallel()
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("hi", "user"),
			{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{
					ID:   "call-1",
					Name: "get_weather",
					Args: map[string]any{"city": "Dublin"},
				},
			}}},
			genai.NewContentFromText("continue", "user"),
		},
		Config: &genai.GenerateContentConfig{},
	}
	chatReq, err := ChatRequestFromLLMRequest("gemma3", req, false)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, msg := range chatReq.Messages {
		if msg.Role == ollamaRoleAssistant && len(msg.ToolCalls) > 0 {
			found = true
			if msg.ToolCalls[0].Function.Name != "get_weather" {
				t.Fatalf("tool call name: %q", msg.ToolCalls[0].Function.Name)
			}
		}
	}
	if !found {
		t.Fatal("expected assistant message with tool calls")
	}
}

func TestOptionsFromGenai(t *testing.T) {
	t.Parallel()
	temp := float32(0.7)
	topP := float32(0.9)
	topK := float32(40)
	cfg := &genai.GenerateContentConfig{
		Temperature:     &temp,
		TopP:            &topP,
		TopK:            &topK,
		MaxOutputTokens: 1024,
		StopSequences:   []string{"END"},
	}
	opts := optionsFromGenai(cfg)
	if opts["temperature"] != float32(0.7) {
		t.Fatalf("temperature: %v", opts["temperature"])
	}
	if opts["top_p"] != float32(0.9) {
		t.Fatalf("top_p: %v", opts["top_p"])
	}
	if opts["top_k"] != int(40) {
		t.Fatalf("top_k: %v", opts["top_k"])
	}
	if opts["num_predict"] != 1024 {
		t.Fatalf("num_predict: %v", opts["num_predict"])
	}
}

func TestMaybeAppendUserContent_empty(t *testing.T) {
	t.Parallel()
	out := MaybeAppendUserContent(nil)
	if len(out) != 1 || out[0].Role != "user" {
		t.Fatalf("expected user content, got %+v", out)
	}
}

func TestMaybeAppendUserContent_lastIsModel(t *testing.T) {
	t.Parallel()
	contents := []*genai.Content{genai.NewContentFromText("ok", "model")}
	out := MaybeAppendUserContent(contents)
	if len(out) != 2 || out[1].Role != "user" {
		t.Fatalf("expected appended user content, got %+v", out)
	}
}

func TestMaybeAppendUserContent_lastIsUser(t *testing.T) {
	t.Parallel()
	contents := []*genai.Content{genai.NewContentFromText("hi", "user")}
	out := MaybeAppendUserContent(contents)
	if len(out) != 1 {
		t.Fatalf("expected no append, got %d", len(out))
	}
}
