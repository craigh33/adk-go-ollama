// Ollama tool-calling agent example for adk-go: demonstrates how to define
// and use tools with the Ollama model. The agent can call a weather tool to answer
// questions about weather.
//
//	go run ./examples/ollama-tool-calling
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-ollama/ollama"
)

// getWeather simulates a weather API call and returns fake weather data.
func getWeather(city string) string {
	weathers := []string{"sunny", "rainy", "cloudy", "snowy"}
	temps := []int{55, 65, 72, 45, 80}
	hash := 0
	for _, r := range strings.ToLower(city) {
		hash += int(r)
	}
	weather := weathers[hash%len(weathers)]
	temp := temps[hash%len(temps)]
	return fmt.Sprintf("Weather in %s: %s, %d°F", city, weather, temp)
}

// weatherToolHandler processes tool calls for the weather tool.
func weatherToolHandler(toolCall *genai.FunctionCall) (map[string]any, error) {
	city, ok := toolCall.Args["city"].(string)
	if !ok {
		return nil, errors.New("city parameter is required and must be a string")
	}

	result := getWeather(city)
	return map[string]any{"result": result}, nil
}

func firstResponse(ctx context.Context, llm model.LLM, req *model.LLMRequest) (*model.LLMResponse, error) {
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, err
		}
		if resp != nil {
			return resp, nil
		}
	}
	return nil, errors.New("empty model response")
}

//nolint:funlen // Example keeps the end-to-end tool-calling flow in one function for readability.
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

	// Define the weather tool
	weatherTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "get_weather",
			Description: "Gets the current weather for a specified city",
			Parameters: &genai.Schema{
				Type: "object",
				Properties: map[string]*genai.Schema{
					"city": {
						Type:        "string",
						Description: "The city name to get weather for",
					},
				},
				Required: []string{"city"},
			},
		}},
	}

	userMsg := "What's the weather like in Seattle?"
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("User: %s\n\n", userMsg)

	userContent := genai.NewContentFromText(userMsg, genai.RoleUser)
	initialReq := &model.LLMRequest{
		Contents: []*genai.Content{userContent},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
			Tools:           []*genai.Tool{weatherTool},
		},
	}

	resp, err := firstResponse(ctx, llm, initialReq)
	if err != nil {
		log.Panicf("initial model call: %v", err)
	}

	if resp.Content == nil || len(resp.Content.Parts) == 0 {
		log.Panic("model returned no content")
	}

	var modelToolParts []*genai.Part
	var toolResultParts []*genai.Part

	for _, part := range resp.Content.Parts {
		if part.FunctionCall == nil {
			continue
		}
		fmt.Printf("Tool call: %s args=%v\n", part.FunctionCall.Name, part.FunctionCall.Args)
		result, werr := weatherToolHandler(part.FunctionCall)
		if werr != nil {
			log.Panicf("weather tool: %v", werr)
		}
		fmt.Printf("Tool result: %v\n\n", result["result"])
		modelToolParts = append(modelToolParts, part)
		toolResultParts = append(toolResultParts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID:       part.FunctionCall.ID,
			Name:     part.FunctionCall.Name,
			Response: result,
		}})
	}

	if len(toolResultParts) == 0 {
		fmt.Println("Assistant response:")
		for _, p := range resp.Content.Parts {
			if p.Text != "" {
				fmt.Println(p.Text)
			}
		}
		fmt.Printf("\nFinish reason: %s\n", resp.FinishReason)
		return
	}

	followupReq := &model.LLMRequest{
		Contents: []*genai.Content{
			userContent,
			{Role: genai.RoleModel, Parts: modelToolParts},
			{Role: genai.RoleUser, Parts: toolResultParts},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
			Tools:           []*genai.Tool{weatherTool},
		},
	}

	finalResp, err := firstResponse(ctx, llm, followupReq)
	if err != nil {
		log.Panicf("final model call: %v", err)
	}

	fmt.Println("Assistant's final response:")
	if finalResp.Content != nil {
		for _, p := range finalResp.Content.Parts {
			if p.Text != "" {
				fmt.Println(p.Text)
			}
		}
	}
	fmt.Printf("\nFinish reason: %s\n", finalResp.FinishReason)
}
