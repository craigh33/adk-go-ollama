// Package mappers converts between ADK/genai types and Ollama Chat API types.
package mappers

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	ollamaapi "github.com/ollama/ollama/api"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	genaiRoleUser   = "user"
	genaiRoleModel  = "model"
	genaiRoleSystem = "system"

	ollamaRoleUser      = "user"
	ollamaRoleAssistant = "assistant"
	ollamaRoleSystem    = "system"
	ollamaRoleTool      = "tool"
)

// MaybeAppendUserContent ensures the conversation ends with a user message,
// mirroring the Gemini provider behavior so empty histories or
// assistant-terminated turns still receive a valid user message.
func MaybeAppendUserContent(contents []*genai.Content) []*genai.Content {
	if len(contents) == 0 {
		return append(contents, genai.NewContentFromText(
			"Handle the requests as specified in the System Instruction.", genaiRoleUser))
	}
	if last := contents[len(contents)-1]; last != nil && last.Role != genaiRoleUser {
		return append(contents, genai.NewContentFromText(
			"Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.",
			genaiRoleUser,
		))
	}
	return contents
}

// ChatRequestFromLLMRequest builds an Ollama [ollamaapi.ChatRequest] from an ADK LLMRequest.
func ChatRequestFromLLMRequest(modelID string, req *model.LLMRequest, stream bool) (*ollamaapi.ChatRequest, error) {
	if req == nil {
		return nil, errors.New("nil LLMRequest")
	}
	cfg := req.Config
	if cfg == nil {
		cfg = &genai.GenerateContentConfig{}
	}

	contents := MaybeAppendUserContent(append([]*genai.Content(nil), req.Contents...))

	messages, err := messagesToOllama(cfg, contents)
	if err != nil {
		return nil, err
	}

	chatReq := &ollamaapi.ChatRequest{
		Model:    modelID,
		Messages: messages,
		Stream:   &stream,
		Options:  optionsFromGenai(cfg),
	}

	if tools := toolsFromGenai(cfg); len(tools) > 0 {
		chatReq.Tools = tools
	}

	return chatReq, nil
}

func systemPartsToMessages(parts []*genai.Part) []ollamaapi.Message {
	var msgs []ollamaapi.Message
	for _, part := range parts {
		if part != nil && part.Text != "" {
			msgs = append(msgs, ollamaapi.Message{
				Role:    ollamaRoleSystem,
				Content: part.Text,
			})
		}
	}
	return msgs
}

func messagesToOllama(cfg *genai.GenerateContentConfig, contents []*genai.Content) ([]ollamaapi.Message, error) {
	var messages []ollamaapi.Message

	// Add system instruction from config.
	if cfg != nil && cfg.SystemInstruction != nil {
		messages = append(messages, systemPartsToMessages(cfg.SystemInstruction.Parts)...)
	}

	for _, c := range contents {
		if c == nil {
			continue
		}
		switch c.Role {
		case genaiRoleSystem:
			messages = append(messages, systemPartsToMessages(c.Parts)...)
		case genaiRoleUser:
			msgs, err := userContentToMessages(c)
			if err != nil {
				return nil, err
			}
			messages = append(messages, msgs...)
		case genaiRoleModel:
			msg := modelContentToMessage(c)
			messages = append(messages, msg)
		default:
			return nil, fmt.Errorf("unsupported content role: %q", c.Role)
		}
	}

	return messages, nil
}

func userContentToMessages(c *genai.Content) ([]ollamaapi.Message, error) {
	var toolMsgs []ollamaapi.Message
	var texts []string
	var images []ollamaapi.ImageData

	for _, part := range c.Parts {
		if part == nil {
			continue
		}

		if part.FunctionResponse != nil {
			respBytes, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("marshal function response: %w", err)
			}
			toolMsgs = append(toolMsgs, ollamaapi.Message{
				Role:       ollamaRoleTool,
				Content:    string(respBytes),
				ToolName:   part.FunctionResponse.Name,
				ToolCallID: part.FunctionResponse.ID,
			})
			continue
		}

		if part.Text != "" {
			texts = append(texts, part.Text)
		}
		if part.InlineData != nil && len(part.InlineData.Data) > 0 {
			images = append(images, ollamaapi.ImageData(part.InlineData.Data))
		}
	}

	var out []ollamaapi.Message
	out = append(out, toolMsgs...)
	if len(texts) > 0 || len(images) > 0 {
		out = append(out, ollamaapi.Message{
			Role:    ollamaRoleUser,
			Content: strings.Join(texts, "\n"),
			Images:  images,
		})
	} else if len(toolMsgs) > 0 {
		out = append(out, ollamaapi.Message{
			Role:    ollamaRoleUser,
			Content: "Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.",
		})
	}
	return out, nil
}

func modelContentToMessage(c *genai.Content) ollamaapi.Message {
	msg := ollamaapi.Message{Role: ollamaRoleAssistant}
	var texts []string

	for _, part := range c.Parts {
		if part == nil {
			continue
		}

		if part.FunctionCall != nil {
			args := ollamaapi.NewToolCallFunctionArguments()
			for k, v := range part.FunctionCall.Args {
				args.Set(k, v)
			}
			msg.ToolCalls = append(msg.ToolCalls, ollamaapi.ToolCall{
				ID: part.FunctionCall.ID,
				Function: ollamaapi.ToolCallFunction{
					Name:      part.FunctionCall.Name,
					Arguments: args,
				},
			})
			continue
		}

		if part.Thought {
			msg.Thinking = part.Text
			continue
		}

		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	msg.Content = strings.Join(texts, "\n")
	return msg
}

func toolsFromGenai(cfg *genai.GenerateContentConfig) ollamaapi.Tools {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil
	}
	var tools ollamaapi.Tools
	for _, t := range cfg.Tools {
		if t == nil {
			continue
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil || fd.Name == "" {
				continue
			}
			tools = append(tools, ollamaapi.Tool{
				Type: "function",
				Function: ollamaapi.ToolFunction{
					Name:        fd.Name,
					Description: fd.Description,
					Parameters:  functionParametersToOllama(fd),
				},
			})
		}
	}
	return tools
}

func functionParametersToOllama(fd *genai.FunctionDeclaration) ollamaapi.ToolFunctionParameters {
	// Prefer JSON schema if provided.
	if fd.ParametersJsonSchema != nil {
		if out, err := jsonRoundTrip[ollamaapi.ToolFunctionParameters](fd.ParametersJsonSchema); err == nil {
			return out
		}
	}
	if fd.Parameters != nil {
		if out, err := parseMapParameters(fd.Parameters); err == nil {
			return out
		}
	}
	return ollamaapi.ToolFunctionParameters{Type: "object"}
}

func parseMapParameters(v any) (ollamaapi.ToolFunctionParameters, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return ollamaapi.ToolFunctionParameters{}, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return ollamaapi.ToolFunctionParameters{}, err
	}
	normalizeSchemaTypes(m)
	return jsonRoundTrip[ollamaapi.ToolFunctionParameters](m)
}

func jsonRoundTrip[T any](v any) (T, error) {
	var out T
	b, err := json.Marshal(v)
	if err != nil {
		return out, err
	}
	err = json.Unmarshal(b, &out)
	return out, err
}

// normalizeSchemaTypes recursively lowercases every "type" field value in a
// JSON Schema map since genai marshals uppercase type names (e.g. "STRING").
func normalizeSchemaTypes(v any) {
	switch m := v.(type) {
	case map[string]any:
		for k, val := range m {
			if k == "type" {
				if s, ok := val.(string); ok {
					m[k] = strings.ToLower(s)
				}
			} else {
				normalizeSchemaTypes(val)
			}
		}
	case []any:
		for _, item := range m {
			normalizeSchemaTypes(item)
		}
	}
}

func optionsFromGenai(cfg *genai.GenerateContentConfig) map[string]any {
	if cfg == nil {
		return nil
	}
	opts := map[string]any{}
	if cfg.Temperature != nil {
		opts["temperature"] = *cfg.Temperature
	}
	if cfg.TopP != nil {
		opts["top_p"] = *cfg.TopP
	}
	if cfg.TopK != nil {
		opts["top_k"] = int(*cfg.TopK)
	}
	if cfg.MaxOutputTokens > 0 {
		opts["num_predict"] = int(cfg.MaxOutputTokens)
	}
	if len(cfg.StopSequences) > 0 {
		opts["stop"] = cfg.StopSequences
	}
	if len(opts) == 0 {
		return nil
	}
	return opts
}
