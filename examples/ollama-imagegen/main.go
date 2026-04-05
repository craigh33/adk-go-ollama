// Ollama imagegen tool example for adk-go: demonstrates how to use the local
// image generation feature in Ollama.
//
//	go run ./examples/ollama-imagegen
//
// Note: Requires an Ollama version built with image generation support
// and the 'x/flux2-klein:4b' model pulled.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-ollama/ollama"
	"github.com/craigh33/adk-go-ollama/tools/imagegenerator"
)

func main() {
	ctx := context.Background()

	// 1. Text model for the agent's brain (planning and determining when to invoke the tool)
	chatModelName := os.Getenv("OLLAMA_MODEL")
	if chatModelName == "" {
		log.Println("OLLAMA_MODEL is not set, defaulting to qwen3:4b")
		chatModelName = "qwen3:4b"
	}

	llm, err := ollama.New(chatModelName)
	if err != nil {
		log.Panicf("ollama model: %v", err)
	}

	// 2. Instantiate the image generation tool
	imgModelName := os.Getenv("OLLAMA_IMAGE_MODEL")
	if imgModelName == "" {
		log.Println("OLLAMA_IMAGE_MODEL is not set, defaulting to x/flux2-klein:4b")
		imgModelName = "x/flux2-klein:4b"
	}

	imgTool, err := imagegenerator.New(imagegenerator.Config{
		ModelID: imgModelName,
	})
	if err != nil {
		log.Panicf("image tool: %v", err)
	}

	// 3. Create the agent with the tool available
	a, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Description: "A helpful creative assistant",
		Model:       llm,
		Tools:       []tool.Tool{imgTool},
		Instruction: "You are a helpful assistant. If the user asks for a picture, use the generate_image tool. Be brief.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	})
	if err != nil {
		log.Panicf("agent: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "ollama-imagegen-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Panicf("runner: %v", err)
	}

	userMsg := "Can you draw me a sunset over the mountains and save it as sunset.png?"
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("User: %s\n\n", userMsg)

	for ev, err := range r.Run(ctx, "local-user", "demo-session", genai.NewContentFromText(userMsg, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Panicf("run: %v", err)
		}

		if ev.Author != a.Name() {
			continue
		}

		llmResp := &ev.LLMResponse
		if llmResp.Partial || llmResp.Content == nil {
			continue
		}

		for _, p := range llmResp.Content.Parts {
			if p.FunctionCall != nil {
				fmt.Printf("=> Tool Calling [%s] with args: %v\n", p.FunctionCall.Name, p.FunctionCall.Args)
			}
			if p.FunctionResponse != nil {
				fmt.Printf("=> Tool Result [%s]: %v\n", p.FunctionResponse.Name, p.FunctionResponse.Response)
				file, _ := p.FunctionResponse.Response["file_name"].(string)
				if file != "" {
					fmt.Printf("=> Image saved! Check artifact: %s\n\n", file)
				}
			}
			if p.Text != "" {
				fmt.Print(p.Text)
			}
		}
	}
	fmt.Println()
}
