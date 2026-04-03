package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-ollama/ollama"
)

func main() {
	ctx := context.Background()

	modelName := os.Getenv("OLLAMA_MODEL")
	if modelName == "" {
		log.Println("OLLAMA_MODEL is not set, defaulting to gemma3:1b")
		modelName = "gemma3:1b"
	}

	prompt := "Explain what streaming model output means in one short paragraph."
	if len(os.Args) > 1 {
		prompt = strings.Join(os.Args[1:], " ")
	}

	llm, err := ollama.New(modelName)
	if err != nil {
		log.Panicf("ollama model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText(prompt, genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	}

	fmt.Printf("Prompt: %s\n\n", prompt)
	var accumulatedText strings.Builder
	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			log.Panicf("stream: %v", err)
		}
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			continue
		}
		if resp.Partial {
			if txt := resp.Content.Parts[0].Text; txt != "" {
				accumulatedText.WriteString(txt)
				fmt.Printf("[partial] %s\n", txt)
			}
			continue
		}

		fmt.Println("\n[final]")
		for _, p := range resp.Content.Parts {
			if p.Text != "" {
				fmt.Println(p.Text)
			}
			if p.FunctionCall != nil {
				fmt.Printf("function_call id=%s name=%s args=%v\n",
					p.FunctionCall.ID, p.FunctionCall.Name, p.FunctionCall.Args)
			}
		}
		fmt.Printf("finish_reason=%s turn_complete=%t\n",
			resp.FinishReason, resp.TurnComplete)
	}
}
