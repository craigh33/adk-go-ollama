package ollama

import (
	"context"
	"iter"

	"google.golang.org/adk/model"
)

// Model implements [model.LLM] using Ollama.
type Model struct {
	modelID string
}

// GenerateContent implements [model.LLM].
func (m *Model) GenerateContent(
	_ context.Context,
	_ *model.LLMRequest,
	_ bool,
) iter.Seq2[*model.LLMResponse, error] {
	panic("unimplemented")
}

// Name implements [model.LLM].
func (m *Model) Name() string {
	return m.modelID
}

var _ model.LLM = (*Model)(nil)
