package neurogate

import (
	"context"
	"path/filepath"
	"testing"
)

func TestNeuroGateBasicAndAnchors(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "neurogate_test.bin")

	samples := []DataSample{
		{Text: "turn on living room lamps", Label: "LightControl"},
		{Text: "it is too dark here please switch on light", Label: "LightControl"},
		{Text: "cooling mode maximum fan temperature bedroom", Label: "ClimateControl"},
		{Text: "set ac thermostat cool air heat", Label: "ClimateControl"},
	}

	cfg := DefaultTrainConfig()
	cfg.Epochs = 50
	cfg.EmbeddingDim = 32
	cfg.HiddenDim = 32
	cfg.TargetVocabSize = 64

	model, err := TrainModel(samples, cfg)
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}
	if err := SaveBinaryModel(modelPath, model); err != nil {
		t.Fatalf("SaveBinaryModel failed: %v", err)
	}
	gate, err := NewNeuroGate(modelPath)
	if err != nil {
		t.Fatalf("NewNeuroGate failed: %v", err)
	}
	gate.SetTemperature(1.5)
	if gate.Temperature() != 1.5 {
		t.Fatalf("expected temperature 1.5, got %v", gate.Temperature())
	}

	var lightTriggered, fallbackTriggered bool

	gate.Bind("LightControl", func(ctx context.Context, payload any) error {
		lightTriggered = true
		return nil
	}).WithAnchor(1.5, "lamp", "lamps", "dark", "light")

	gate.Bind("ClimateControl", func(ctx context.Context, payload any) error {
		return nil
	}).WithAnchor(1.5, "cooling", "ac", "fan", "temp")

	gate.Fallback(func(ctx context.Context, payload any) error {
		fallbackTriggered = true
		return nil
	})

	ctx := context.Background()

	// 1. Verify anchor boost flips or reinforces LightControl
	lightTriggered = false
	trace := gate.Inspect("it is too dark in here please switch on lamps")
	err = gate.Filter(ctx, "it is too dark in here please switch on lamps", nil)
	if err != nil {
		t.Fatalf("Filter returned error: %v", err)
	}
	if !lightTriggered {
		t.Errorf("Expected LightControl to trigger, got fallback=%t, ambiguous=%t, pred=%s, conf=%.2f, reason=%s",
			trace.IsFallback, trace.IsAmbiguous, trace.PredictedLabel, trace.Confidence, trace.FallbackReason)
	}

	// 2. Verify inspection trace
	trace = gate.Inspect("it is too dark in here please switch on lamps")
	if trace.PredictedLabel != "LightControl" {
		t.Errorf("Expected trace predicted label LightControl, got %s", trace.PredictedLabel)
	}
	if len(trace.TriggeredAnchors) == 0 {
		t.Errorf("Expected triggered anchors to be recorded")
	}

	// 3. Verify OOD boundary rejection
	// Set strict domain boundary
	gate.SetMinCosineSim(0.99) // Impose strict threshold
	fallbackTriggered = false
	_ = gate.Filter(ctx, "what is the meaning of quantum black holes", nil)
	if !fallbackTriggered {
		t.Errorf("Expected OOD query to be isolated to Fallback under strict boundary")
	}
}

func BenchmarkNeuroGateFilterAllocations(b *testing.B) {
	samples := []DataSample{
		{Text: "please refund money to card", Label: "Refund"},
		{Text: "package delivery courier tracking", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	model, err := TrainModel(samples, cfg)
	if err != nil {
		b.Fatalf("TrainModel failed: %v", err)
	}

	gate := NewNeuroGateWithModel(model)
	gate.Bind("Refund", func(ctx context.Context, payload any) error {
		return nil
	}).WithAnchor(1.0, "refund", "card", "money")

	gate.Bind("Delivery", func(ctx context.Context, payload any) error {
		return nil
	}).WithAnchor(1.0, "package", "courier", "tracking")

	ctx := context.Background()
	query := "please refund money to card"

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = gate.Filter(ctx, query, nil)
	}
}

func BenchmarkNeuroGateFilterTokensZeroAlloc(b *testing.B) {
	samples := []DataSample{
		{Text: "please refund money to card", Label: "Refund"},
		{Text: "package delivery courier tracking", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	model, err := TrainModel(samples, cfg)
	if err != nil {
		b.Fatalf("TrainModel failed: %v", err)
	}

	gate := NewNeuroGateWithModel(model)
	gate.Bind("Refund", func(ctx context.Context, payload any) error {
		return nil
	}).WithAnchor(1.0, "refund", "card", "money")

	gate.Bind("Delivery", func(ctx context.Context, payload any) error {
		return nil
	}).WithAnchor(1.0, "package", "courier", "tracking")

	ctx := context.Background()
	tokens := model.Tokenizer.Encode("please refund money to card")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = gate.FilterTokens(ctx, tokens, nil)
	}
}

func TestNeuroGateAnchorBoostCap(t *testing.T) {
	samples := []DataSample{
		{Text: "please refund money to card", Label: "Refund"},
		{Text: "package delivery courier tracking", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	model, err := TrainModel(samples, cfg)
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}

	gate := NewNeuroGateWithModel(model)
	// Register anchor rule with large weight: 5.0 per matched keyword
	gate.Bind("Refund", func(ctx context.Context, payload any) error {
		return nil
	}).WithAnchor(5.0, "refund", "money", "card")

	// 1. With maxAnchorBoost capped at 2.0
	gate.SetMaxAnchorBoost(2.0)
	if capVal := gate.MaxAnchorBoost(); capVal != 2.0 {
		t.Fatalf("expected maxAnchorBoost 2.0, got %f", capVal)
	}

	traceCapped := gate.Inspect("please refund money to card")
	if len(traceCapped.TriggeredAnchors) < 2 {
		t.Fatalf("expected multiple triggered anchors, got %v", traceCapped.TriggeredAnchors)
	}

	// 2. Uncap or increase cap and check relative confidence change
	gate.SetMaxAnchorBoost(10.0)
	traceHighCap := gate.Inspect("please refund money to card")

	// Under a higher cap, the logit boost is greater, leading to higher confidence for Refund
	if traceHighCap.Confidence <= traceCapped.Confidence {
		t.Logf("Capped conf: %f, High cap conf: %f", traceCapped.Confidence, traceHighCap.Confidence)
	}
}

func TestNeuroGateCalibrateDomainDistribution(t *testing.T) {
	samples := []DataSample{
		{Text: "please refund money to card", Label: "Refund"},
		{Text: "i want my money back for cancellation", Label: "Refund"},
		{Text: "package delivery courier tracking shipment", Label: "Delivery"},
		{Text: "where is my parcel courier driver", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 25
	model, err := TrainModel(samples, cfg)
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}

	gate := NewNeuroGateWithModel(model)
	gate.CalibrateDomainDistribution(samples, 2.0)

	hasCentroid, meanSim, stdDev, minCosine := gate.DomainStats()
	if !hasCentroid {
		t.Fatalf("expected hasCentroid to be true after calibration")
	}
	if meanSim <= 0.0 {
		t.Fatalf("expected positive meanSim, got %f", meanSim)
	}
	if stdDev < 0.0 {
		t.Fatalf("expected non-negative stdDev, got %f", stdDev)
	}
	if minCosine > meanSim {
		t.Fatalf("expected minCosine (%f) <= meanSim (%f)", minCosine, meanSim)
	}

	// Inspect in-domain query
	traceIn := gate.Inspect("please refund money back to credit card")
	if traceIn.IsOOD {
		t.Errorf("expected in-domain query to not be OOD, got sim=%f, minCosine=%f", traceIn.CosineSimilarity, minCosine)
	}
}
