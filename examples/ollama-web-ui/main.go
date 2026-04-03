package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-ollama/ollama"
)

func main() {
	ctx := context.Background()

	modelName := os.Getenv("OLLAMA_MODEL")
	if modelName == "" {
		log.Println("OLLAMA_MODEL is not set, using default 'gemma3'")
		modelName = "gemma3"
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

	launcherCfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, launcherCfg, append([]string{"run"}, os.Args[1:]...)); err != nil {
		log.Panicf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
