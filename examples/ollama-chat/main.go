// Ollama chat example for adk-go: run:
//
//	go run ./examples/ollama-chat
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
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

	llm, err := ollama.New(modelName)
	if err != nil {
		log.Panicf("ollama model: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Description: "A helpful assistant",
		Model:       llm,
		Instruction: "You reply briefly and clearly.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	})
	if err != nil {
		log.Panicf("agent: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "ollama-chat-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Panicf("runner: %v", err)
	}

	userMsg := "What is 2+2? Reply with just the number."
	if len(os.Args) > 1 {
		userMsg = os.Args[1]
	}

	for ev, err := range r.Run(ctx, "local-user", "demo-session", genai.NewContentFromText(userMsg, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Panicf("run: %v", err)
		}
		if ev.Author != a.Name() {
			continue
		}
		if ev.LLMResponse.Partial {
			continue
		}
		if ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text != "" {
					fmt.Print(p.Text)
				}
			}
			fmt.Println()
		}
	}
}
