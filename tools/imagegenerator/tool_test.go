package imagegenerator

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/genai"
)

var errNotImplemented = errors.New("not implemented")

// mockArtifacts implements agent.Artifacts.
type mockArtifacts struct {
	savedCount int
}

func (m *mockArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	m.savedCount++
	return &artifact.SaveResponse{Version: 1}, nil
}
func (m *mockArtifacts) List(context.Context) (*artifact.ListResponse, error) {
	return nil, errNotImplemented
}
func (m *mockArtifacts) Load(ctx context.Context, name string) (*artifact.LoadResponse, error) {
	return nil, errNotImplemented
}
func (m *mockArtifacts) LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error) {
	return nil, errNotImplemented
}

// mockContext implements tool.Context.
type mockContext struct {
	ctx       context.Context
	artifacts agent.Artifacts
}

// Context methods.
func (m *mockContext) Deadline() (time.Time, bool) { return m.ctx.Deadline() }
func (m *mockContext) Done() <-chan struct{}       { return m.ctx.Done() }
func (m *mockContext) Err() error                  { return m.ctx.Err() }
func (m *mockContext) Value(key any) any           { return m.ctx.Value(key) }

// tool.Context methods.
func (m *mockContext) Artifacts() agent.Artifacts     { return m.artifacts }
func (m *mockContext) FunctionCallID() string         { return "" }
func (m *mockContext) Actions() *session.EventActions { return nil }
func (m *mockContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return nil, errNotImplemented
}
func (m *mockContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }
func (m *mockContext) RequestConfirmation(hint string, payload any) error   { return nil }
func (m *mockContext) UserContent() *genai.Content                          { return nil }
func (m *mockContext) InvocationID() string                                 { return "" }
func (m *mockContext) AgentName() string                                    { return "" }
func (m *mockContext) ReadonlyState() session.ReadonlyState                 { return nil }
func (m *mockContext) UserID() string                                       { return "" }
func (m *mockContext) AppName() string                                      { return "" }
func (m *mockContext) SessionID() string                                    { return "" }
func (m *mockContext) Branch() string                                       { return "" }
func (m *mockContext) State() session.State                                 { return nil }

func mockServerHandler(w http.ResponseWriter, r *http.Request) {
	var req imageRequest
	_ = json.NewDecoder(r.Body).Decode(&req)

	if req.Prompt == "bad-json" {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{ bad json`))
		return
	}
	if req.Prompt == "server-error" {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	if req.Prompt == "empty-data" {
		_ = json.NewEncoder(w).Encode(imageResponse{
			Data: []struct {
				B64JSON string `json:"b64_json"`
			}{},
		})
		return
	}
	if req.Prompt == "empty-b64" {
		_ = json.NewEncoder(w).Encode(imageResponse{
			Data: []struct {
				B64JSON string `json:"b64_json"`
			}{{B64JSON: ""}},
		})
		return
	}
	if req.Prompt == "bad-b64" {
		_ = json.NewEncoder(w).Encode(imageResponse{
			Data: []struct {
				B64JSON string `json:"b64_json"`
			}{{B64JSON: "not-base-64!!++"}},
		})
		return
	}

	validBase64 := base64.StdEncoding.EncodeToString([]byte("fake-image"))
	_ = json.NewEncoder(w).Encode(imageResponse{
		Data: []struct {
			B64JSON string `json:"b64_json"`
		}{{B64JSON: validBase64}},
	})
}

func verifyError(t *testing.T, err error, wantErr bool, errContains string) {
	t.Helper()
	if wantErr {
		if err == nil {
			t.Fatalf("expected error containing %q, got nil", errContains)
		}
		if !strings.Contains(err.Error(), errContains) {
			t.Errorf("expected error containing %q, got %q", errContains, err.Error())
		}
		return
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func verifySuccess(t *testing.T, resp map[string]any, mArt *mockArtifacts) {
	t.Helper()
	if resp["status"] != "success" {
		t.Errorf("expected status success, got %v", resp["status"])
	}
	if mArt.savedCount != 1 {
		t.Errorf("expected Artifacts.Save to be called 1 time, got %d", mArt.savedCount)
	}
}

func TestNew(t *testing.T) {
	_, err := New(Config{BaseURL: "ftp://localhost"})
	if err == nil {
		t.Error("expected error for ftp scheme, got nil")
	}

	_, err = New(Config{BaseURL: "::broken:url://"})
	if err == nil {
		t.Error("expected error for parser failure, got nil")
	}

	toolImpl, err := New(Config{BaseURL: "http://localhost:11434"})
	if err != nil {
		t.Errorf("unexpected error for valid URL: %v", err)
	}
	if toolImpl == nil {
		t.Error("expected tool to be non-nil")
	}
}

func TestRun(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(mockServerHandler))
	defer ts.Close()

	toolImpl, _ := New(Config{
		BaseURL:    ts.URL,
		HTTPClient: ts.Client(),
	})

	tests := []struct {
		name        string
		args        any
		expectErr   bool
		errContains string
	}{
		{name: "success", args: map[string]any{"prompt": "a nice cat"}, expectErr: false},
		{
			name:      "success with filename",
			args:      map[string]any{"prompt": "a nice cat", "file_name": "cat.png"},
			expectErr: false,
		},
		{
			name:        "invalid file name",
			args:        map[string]any{"prompt": "cat", "file_name": "../cat.png"},
			expectErr:   true,
			errContains: "invalid file_name provided",
		},
		{name: "missing prompt", args: map[string]any{}, expectErr: true, errContains: "prompt is required"},
		{
			name:        "server err",
			args:        map[string]any{"prompt": "server-error"},
			expectErr:   true,
			errContains: "ollama API error",
		},
		{name: "bad json", args: map[string]any{"prompt": "bad-json"}, expectErr: true, errContains: "decode response"},
		{
			name:        "empty data",
			args:        map[string]any{"prompt": "empty-data"},
			expectErr:   true,
			errContains: "no image data",
		},
		{
			name:        "empty b64",
			args:        map[string]any{"prompt": "empty-b64"},
			expectErr:   true,
			errContains: "empty b64_json",
		},
		{name: "bad b64", args: map[string]any{"prompt": "bad-b64"}, expectErr: true, errContains: "decode b64_json"},
		{name: "invalid args", args: "not-a-map", expectErr: true, errContains: "unexpected args type"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mArt := &mockArtifacts{}
			mCtx := &mockContext{ctx: context.Background(), artifacts: mArt}

			resp, err := toolImpl.(interface {
				Run(tool.Context, any) (map[string]any, error)
			}).Run(mCtx, tt.args)

			verifyError(t, err, tt.expectErr, tt.errContains)
			if !tt.expectErr {
				verifySuccess(t, resp, mArt)
			}
		})
	}
}
