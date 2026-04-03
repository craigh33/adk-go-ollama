package mappers

import (
	ollamaapi "github.com/ollama/ollama/api"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// LLMResponseFromChatResponse maps an Ollama ChatResponse to an ADK LLMResponse.
func LLMResponseFromChatResponse(resp *ollamaapi.ChatResponse, partial bool) *model.LLMResponse {
	if resp == nil {
		return &model.LLMResponse{}
	}

	llmResp := &model.LLMResponse{
		Content: contentFromMessage(&resp.Message),
		Partial: partial,
	}

	if resp.Done {
		llmResp.FinishReason = FinishReasonFromDoneReason(resp.DoneReason)
		llmResp.UsageMetadata = UsageFromChatResponse(resp)
	}

	return llmResp
}

// FinalLLMResponseFromStream builds the final TurnComplete response from
// the accumulated stream text and the done-chunk metadata.
func FinalLLMResponseFromStream(doneResp *ollamaapi.ChatResponse, accumulatedText string) *model.LLMResponse {
	var parts []*genai.Part

	// Include thinking if present in the final chunk.
	if doneResp != nil && doneResp.Message.Thinking != "" {
		parts = append(parts, &genai.Part{
			Text:    doneResp.Message.Thinking,
			Thought: true,
		})
	}

	if accumulatedText != "" {
		parts = append(parts, &genai.Part{Text: accumulatedText})
	}

	// Tool calls arrive in the done chunk.
	if doneResp != nil {
		for i := range doneResp.Message.ToolCalls {
			parts = append(parts, &genai.Part{
				FunctionCall: FunctionCallFromToolCall(&doneResp.Message.ToolCalls[i]),
			})
		}
	}

	if len(parts) == 0 {
		parts = []*genai.Part{{Text: ""}}
	}

	resp := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: parts,
		},
		TurnComplete: true,
	}

	if doneResp != nil {
		resp.FinishReason = FinishReasonFromDoneReason(doneResp.DoneReason)
		resp.UsageMetadata = UsageFromChatResponse(doneResp)
	}

	return resp
}

func contentFromMessage(msg *ollamaapi.Message) *genai.Content {
	parts := partsFromMessage(msg)
	if len(parts) == 0 {
		parts = []*genai.Part{{Text: ""}}
	}
	return &genai.Content{Role: "model", Parts: parts}
}

func partsFromMessage(msg *ollamaapi.Message) []*genai.Part {
	var parts []*genai.Part

	if msg.Thinking != "" {
		parts = append(parts, &genai.Part{Text: msg.Thinking, Thought: true})
	}
	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
	}
	for i := range msg.ToolCalls {
		parts = append(parts, &genai.Part{
			FunctionCall: FunctionCallFromToolCall(&msg.ToolCalls[i]),
		})
	}
	return parts
}

// FunctionCallFromToolCall converts an Ollama ToolCall to a genai FunctionCall.
func FunctionCallFromToolCall(tc *ollamaapi.ToolCall) *genai.FunctionCall {
	if tc == nil {
		return nil
	}
	return &genai.FunctionCall{
		ID:   tc.ID,
		Name: tc.Function.Name,
		Args: tc.Function.Arguments.ToMap(),
	}
}

// FinishReasonFromDoneReason maps Ollama done reasons to genai finish reasons.
func FinishReasonFromDoneReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "load":
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonStop
	}
}

// UsageFromChatResponse extracts token usage metadata from an Ollama ChatResponse.
//
//nolint:gosec // safe to cast token sums to int32
func UsageFromChatResponse(resp *ollamaapi.ChatResponse) *genai.GenerateContentResponseUsageMetadata {
	if resp == nil {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(resp.PromptEvalCount),
		CandidatesTokenCount: int32(resp.EvalCount),
		TotalTokenCount:      int32(resp.PromptEvalCount + resp.EvalCount),
	}
}
