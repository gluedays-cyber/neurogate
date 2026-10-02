package neurogate

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestRouterDispatchAndFallback(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.1)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	var refundCalled, deliveryCalled, fallbackCalled bool

	router.
		Bind("Refund", func(ctx context.Context, payload any) error {
			refundCalled = true
			return nil
		}).
		Bind("Delivery", func(ctx context.Context, payload any) error {
			deliveryCalled = true
			return nil
		}).
		Fallback(func(ctx context.Context, payload any) error {
			fallbackCalled = true
			return nil
		})

	// Dispatch with matching label tokens
	err = router.Dispatch(context.Background(), "refund", nil)
	if err != nil {
		t.Fatalf("Dispatch failed: %v", err)
	}

	if !refundCalled && !fallbackCalled && !deliveryCalled {
		t.Fatal("No route action was executed")
	}

	// Dispatch with high threshold causing deterministic fallback
	strictRouter, err := NewRouter(modelPath, 0.999999)
	if err != nil {
		t.Fatalf("Failed to create strict router: %v", err)
	}

	fallbackTriggered := false
	strictRouter.
		Bind("Refund", func(ctx context.Context, payload any) error {
			return nil
		}).
		Fallback(func(ctx context.Context, payload any) error {
			fallbackTriggered = true
			return nil
		})

	if err := strictRouter.Dispatch(context.Background(), "refund", nil); err != nil {
		t.Fatalf("Strict dispatch failed: %v", err)
	}

	if !fallbackTriggered {
		t.Error("Expected fallback route to trigger when threshold exceeds confidence")
	}
}

func TestRouterInspectAndTrace(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_trace_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.5)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	router.Bind("Refund", func(ctx context.Context, payload any) error {
		return nil
	})

	trace := router.Inspect("refund")
	if len(trace.TokenIDs) == 0 {
		t.Error("Expected non-empty token IDs in trace")
	}
	if len(trace.ClassProbabilities) == 0 {
		t.Error("Expected class probability breakdown in trace")
	}
	if trace.Confidence <= 0.0 {
		t.Errorf("Invalid confidence in trace: %f", trace.Confidence)
	}
}

func TestRouterContextCancellation(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_ctx_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.1)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	router.Bind("Refund", func(ctx context.Context, payload any) error {
		return nil
	})

	// Pre-canceled context should abort immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = router.Dispatch(ctx, "refund", nil)
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
}

func TestRouter3TierDispatch(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_3tier_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.5)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Configure strict policy: High=0.80, Low=0.30, Margin=0.20
	policy := DispatchPolicy{
		HighThreshold: 0.80,
		LowThreshold:  0.30,
		MarginCutoff:  0.20,
		MaxEntropy:    2.0,
	}
	router.SetPolicy(policy)

	var definiteCalled, ambiguousCalled, fallbackCalled bool

	router.
		Bind("Refund", func(ctx context.Context, payload any) error {
			definiteCalled = true
			return nil
		}).
		Ambiguous(func(ctx context.Context, primary string, secondary string, payload any) error {
			ambiguousCalled = true
			return nil
		}).
		Fallback(func(ctx context.Context, payload any) error {
			fallbackCalled = true
			return nil
		})

	// 1. Trace inspect to check confidence
	trace := router.Inspect("refund")
	_ = router.Dispatch(context.Background(), "refund", nil)

	if trace.IsAmbiguous {
		if !ambiguousCalled {
			t.Error("Expected Ambiguous handler to trigger for borderline query")
		}
	} else if trace.IsFallback {
		if !fallbackCalled {
			t.Error("Expected Fallback handler to trigger for isolated query")
		}
	} else {
		if !definiteCalled {
			t.Error("Expected Definite handler to trigger for confident query")
		}
	}
}

func TestRouterOODEntropyIsolation(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_ood_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.5)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Set very strict MaxEntropy: 0.001 to force OOD isolation
	policy := DispatchPolicy{
		HighThreshold: 0.70,
		LowThreshold:  0.20,
		MarginCutoff:  0.10,
		MaxEntropy:    0.001, // Anything with uncertainty is treated as OOD
	}
	router.SetPolicy(policy)

	fallbackTriggered := false
	router.
		Bind("Refund", func(ctx context.Context, payload any) error {
			return nil
		}).
		Fallback(func(ctx context.Context, payload any) error {
			fallbackTriggered = true
			return nil
		})

	err = router.Dispatch(context.Background(), "refund", nil)
	if err != nil {
		t.Fatalf("Dispatch failed: %v", err)
	}

	if !fallbackTriggered {
		t.Error("Expected Fallback route to trigger when entropy exceeds MaxEntropy (OOD)")
	}
}

func TestRouterMultiIntentPipeline(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_pipeline_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.5)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Policy allowing secondary intent with confidence >= 0.20 to enter pipeline
	policy := DispatchPolicy{
		HighThreshold:     0.80,
		LowThreshold:      0.20,
		MarginCutoff:      0.15,
		MaxEntropy:        3.0,
		PipelineThreshold: 0.20,
	}
	router.SetPolicy(policy)

	var specificPipelineCalled bool
	var defaultPipelineCalled bool

	router.
		BindPipeline("Refund", "Delivery", func(ctx context.Context, primary string, secondary string, payload any) error {
			specificPipelineCalled = true
			if primary != "Refund" || secondary != "Delivery" {
				t.Errorf("Unexpected pipeline labels: %s -> %s", primary, secondary)
			}
			return nil
		}).
		DefaultPipeline(func(ctx context.Context, primary string, secondary string, payload any) error {
			defaultPipelineCalled = true
			return nil
		})

	// Dispatch with pipeline support
	err = router.DispatchPipeline(context.Background(), "refund delivery", nil)
	if err != nil {
		t.Fatalf("DispatchPipeline failed: %v", err)
	}

	if !specificPipelineCalled && !defaultPipelineCalled {
		t.Error("Expected at least one pipeline handler to execute for multi-intent input")
	}
}

func TestRouterAtomicReloadConcurrently(t *testing.T) {
	tempDir := t.TempDir()
	modelPathA := filepath.Join(tempDir, "model_a.bin")
	modelPathB := filepath.Join(tempDir, "model_b.bin")

	modelA := createSampleModel()
	if err := SaveBinaryModel(modelPathA, modelA); err != nil {
		t.Fatalf("Failed to save model A: %v", err)
	}

	modelB := createSampleModel()
	modelB.Temperature = 0.8
	if err := SaveBinaryModel(modelPathB, modelB); err != nil {
		t.Fatalf("Failed to save model B: %v", err)
	}

	router, err := NewRouter(modelPathA, 0.5)
	if err != nil {
		t.Fatalf("Failed to construct router: %v", err)
	}

	router.Bind("Refund", func(ctx context.Context, payload any) error {
		return nil
	}).Fallback(func(ctx context.Context, payload any) error {
		return nil
	})

	stopChan := make(chan struct{})
	var wg sync.WaitGroup

	// 8 Reader Goroutines hammering Dispatch concurrently
	numReaders := 8
	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx := context.Background()
			for {
				select {
				case <-stopChan:
					return
				default:
					_ = router.Dispatch(ctx, "refund my payment immediately", nil)
					_ = router.Inspect("check status")
				}
			}
		}(i)
	}

	// 1 Reload Goroutine repeatedly swapping models
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			select {
			case <-stopChan:
				return
			default:
				target := modelPathB
				if i%2 == 0 {
					target = modelPathA
				}
				if err := router.Reload(target); err != nil {
					t.Errorf("Reload failed during concurrent traffic: %v", err)
				}
				time.Sleep(2 * time.Millisecond)
			}
		}
	}()

	time.Sleep(60 * time.Millisecond)
	close(stopChan)
	wg.Wait()
}

func TestRouterTelemetryDrain(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "telemetry_test.bin")

	sampleModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, sampleModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.5)
	if err != nil {
		t.Fatalf("Failed to initialize router: %v", err)
	}

	// Configure small telemetry ring buffer (capacity 4)
	router.EnableTelemetry(4)

	// Dispatch queries that trigger fallback/ambiguous
	ctx := context.Background()
	_ = router.Dispatch(ctx, "unknown gibberish query 12345", nil)
	_ = router.Dispatch(ctx, "another isolated query 67890", nil)

	events := router.DrainTelemetry()
	if len(events) < 2 {
		t.Fatalf("Expected at least 2 telemetry events, got %d", len(events))
	}

	for _, ev := range events {
		if !ev.IsFallback && !ev.IsAmbiguous && !ev.IsPipeline {
			t.Errorf("Expected event to flag fallback, ambiguous, or pipeline: %+v", ev)
		}
	}

	// Ensure Drain emptied the ring buffer
	drainedAgain := router.DrainTelemetry()
	if len(drainedAgain) != 0 {
		t.Errorf("Expected ring buffer to be empty after drain, got %d events", len(drainedAgain))
	}
}


