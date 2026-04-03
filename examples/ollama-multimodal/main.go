package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-ollama/ollama"
)

// downloadImage fetches an image from a URL and returns its raw binary content.
func downloadImage(url string) ([]byte, error) {
	resp, err := http.Get(url) //nolint:gosec,noctx // demonstration utility accepts arbitrary URLs
	if err != nil {
		return nil, fmt.Errorf("download image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download image: status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read image data: %w", err)
	}

	return data, nil
}

// mimeTypeFromURL infers the image MIME type from the URL file extension, defaulting to JPEG.
func mimeTypeFromURL(rawURL string) string {
	lower := strings.ToLower(rawURL)
	if i := strings.IndexByte(lower, '?'); i >= 0 {
		lower = lower[:i]
	}
	switch {
	case strings.HasSuffix(lower, ".png"):
		return "image/png"
	case strings.HasSuffix(lower, ".gif"):
		return "image/gif"
	case strings.HasSuffix(lower, ".webp"):
		return "image/webp"
	default:
		return "image/jpeg"
	}
}

// analyzeImageWithVision demonstrates image analysis with the Ollama model.
func analyzeImageWithVision(ctx context.Context, llm model.LLM, imageURL string) error {
	fmt.Println("\n=== Image Analysis Example ===")
	fmt.Printf("Downloading image from: %s\n", imageURL)

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "What do you see in this image? Please describe it in detail."},
					{InlineData: &genai.Blob{MIMEType: mimeTypeFromURL(imageURL), Data: imageData}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful vision assistant. Provide detailed, accurate descriptions of images."},
				},
			},
		},
	}

	// Generate response
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
		if resp.UsageMetadata != nil {
			fmt.Printf("Usage - Prompt: %d, Candidates: %d, Total: %d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount,
				resp.UsageMetadata.TotalTokenCount)
		}
	}

	return nil
}

// demonstrateMultipleImages shows how to analyze multiple images in sequence.
func demonstrateMultipleImages(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Comparing Multiple Images ===")

	// Use two different public images
	imageURLs := []string{
		"https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
		"https://www.google.com/images/branding/googleg/1x/googleg_standard_color_128dp.png",
	}

	type imageBlob struct {
		data     []byte
		mimeType string
	}
	blobs := make([]imageBlob, 0, len(imageURLs))
	for _, url := range imageURLs {
		data, err := downloadImage(url)
		if err != nil {
			fmt.Printf("Note: Could not download image from %s: %v\n", url, err)
			continue
		}
		blobs = append(blobs, imageBlob{data: data, mimeType: mimeTypeFromURL(url)})
	}

	if len(blobs) < 2 {
		fmt.Println("Note: Could not download enough images for comparison")
		return nil
	}

	// Create comparison request
	parts := []*genai.Part{
		{Text: "Compare these two images. What are the similarities and differences?"},
	}
	for _, b := range blobs {
		parts = append(parts, &genai.Part{
			InlineData: &genai.Blob{MIMEType: b.mimeType, Data: b.data},
		})
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role:  genai.RoleUser,
				Parts: parts,
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Comparison: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

func main() {
	ctx := context.Background()

	modelName := os.Getenv("OLLAMA_MODEL")
	if modelName == "" {
		log.Println("OLLAMA_MODEL is not set, using default 'llava'")
		modelName = "llava"
	}

	imageURL := os.Getenv("IMAGE_URL")
	if imageURL == "" {
		// Use a sample public image for testing
		imageURL = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
	}

	llm, err := ollama.New(modelName)
	if err != nil {
		log.Panicf("create ollama model: %v", err)
	}

	if err := analyzeImageWithVision(ctx, llm, imageURL); err != nil {
		log.Printf("ERROR in Image Analysis with Vision: %v\n", err)
	}

	if err := demonstrateMultipleImages(ctx, llm); err != nil {
		log.Printf("ERROR in Comparing Multiple Images: %v\n", err)
	}

	fmt.Println("\n=== Multimodal Examples Complete ===")
}
