package neurogate

import (
	"context"
	"path/filepath"
	"testing"
)

// TestArchitectureFeedback_Case1_PatternGuard validates 0-Alloc Hex and random sequence blocking.
func TestArchitectureFeedback_Case1_PatternGuard(t *testing.T) {
	// 1. Direct Pattern Scanner Unit Verification
	if !ScanUnlearnedPatterns("0xDEADBEEF") {
		t.Errorf("Expected 0xDEADBEEF to be detected as unlearned pattern")
	}
	if !ScanUnlearnedPatterns("request with 0xCAFEBABE1234 token") {
		t.Errorf("Expected embedded hex sequence to be detected")
	}
	if !ScanUnlearnedPatterns("DEADBEEF0123") {
		t.Errorf("Expected uppercase hex block to be detected")
	}
	if ScanUnlearnedPatterns("normal english command turn on lights") {
		t.Errorf("Normal text should not trigger pattern scanner")
	}
	if ScanUnlearnedPatterns("xzqjwpkvmcty1837") {
		t.Errorf("Lowercase non-hex string should bypass pattern scanner and proceed to Layer 1 BPE guard")
	}

	// 2. Integration with Router Fail-Safe
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "feedback_test.bin")
	samples := []DataSample{
		{Text: "turn on living room lights", Label: "LightControl"},
		{Text: "set air conditioner temperature", Label: "ClimateControl"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	model, err := TrainModel(samples, cfg)
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}
	if err := SaveBinaryModel(modelPath, model); err != nil {
		t.Fatalf("SaveBinaryModel failed: %v", err)
	}

	router, err := NewRouter(modelPath, 0.70)
	if err != nil {
		t.Fatalf("NewRouter failed: %v", err)
	}

	ctx := context.Background()
	_, err = router.RouteQuery(ctx, "0xDEADBEEF1234")
	if err != ErrUnlearnedPattern {
		t.Fatalf("Expected ErrUnlearnedPattern from Router.RouteQuery, got %v", err)
	}

	// 3. Integration with NeuroGate Inspect
	gate, err := NewNeuroGate(modelPath)
	if err != nil {
		t.Fatalf("NewNeuroGate failed: %v", err)
	}
	trace := gate.Inspect("0xDEADBEEF1234")
	if !trace.IsFallback || trace.FallbackReason != "unlearned pattern detected" {
		t.Errorf("Expected NeuroGate Inspect to fallback with unlearned pattern, got fallback=%t, reason=%s",
			trace.IsFallback, trace.FallbackReason)
	}
}

// TestArchitectureFeedback_Case2_LogitMarginAmbiguity validates raw logit margin guard against softmax overconfidence.
func TestArchitectureFeedback_Case2_LogitMarginAmbiguity(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "margin_test.bin")
	samples := []DataSample{
		{Text: "refund my money from account", Label: "Refund"},
		{Text: "track my shipping package courier", Label: "Tracking"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 30
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
	gate.SetMinCosineSim(0.0)

	// Set an aggressive RawLogitMargin requirement
	gate.SetPolicy(DispatchPolicy{
		HighThreshold:      0.50,
		LowThreshold:       0.30,
		MarginCutoff:       0.05,
		MaxEntropy:         2.5,
		RawLogitMargin:     5.0, // Requires at least 5.0 logit gap to avoid ambiguity
		EnablePatternGuard: true,
	})

	trace := gate.Inspect("refund package money tracking")
	if !trace.IsAmbiguous {
		t.Errorf("Expected ambiguous routing when logit margin (%.4f) < 5.0, fallback=%t, reason=%s",
			trace.LogitMargin, trace.IsFallback, trace.FallbackReason)
	}
}

// TestArchitectureFeedback_Case3_AsymmetricInhibition validates negative biasing to suppress competing classes.
func TestArchitectureFeedback_Case3_AsymmetricInhibition(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "inhibit_test.bin")
	samples := []DataSample{
		{Text: "safe query general question info", Label: "SafeOps"},
		{Text: "dangerous exploit attack payload", Label: "SecurityThreat"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 30
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
	gate.SetMinCosineSim(0.0)
	gate.SetPolicy(DispatchPolicy{
		HighThreshold:      0.60,
		LowThreshold:       0.30,
		MarginCutoff:       0.10,
		MaxEntropy:         2.5,
		RawLogitMargin:     0.35,
		MaxSingleCharRatio: 0.85,
		EnablePatternGuard: true,
	})

	var securityTriggered, safeTriggered bool
	gate.Bind("SecurityThreat", func(ctx context.Context, payload any) error {
		securityTriggered = true
		return nil
	}).WithAnchor(2.0, "exploit", "attack").Inhibit(5.0, "SafeOps")

	gate.Bind("SafeOps", func(ctx context.Context, payload any) error {
		safeTriggered = true
		return nil
	})

	ctx := context.Background()
	// Input with a dominant safe framing but containing a critical exploit anchor
	query := "safe query general question with dangerous exploit"
	err = gate.Filter(ctx, query, nil)
	if err != nil {
		t.Fatalf("Filter failed: %v", err)
	}

	if !securityTriggered || safeTriggered {
		trace := gate.Inspect(query)
		t.Errorf("Expected SecurityThreat to trigger via asymmetric anchor inhibition. Security=%t, Safe=%t, Pred=%s, Conf=%.2f, IsFallback=%t, FallbackReason=%s, IsAmbiguous=%t, Logits=%v",
			securityTriggered, safeTriggered, trace.PredictedLabel, trace.Confidence, trace.IsFallback, trace.FallbackReason, trace.IsAmbiguous, trace.ClassProbabilities)
	}
}

// TestTemperatureScaling_Effects validates that temperature adjusts softmax entropy and confidence as expected.
func TestTemperatureScaling_Effects(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "temp_test.bin")
	samples := []DataSample{
		{Text: "request refund order charge", Label: "Refund"},
		{Text: "track my parcel delivery courier", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 40
	model, err := TrainModel(samples, cfg)
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}
	if err := SaveBinaryModel(modelPath, model); err != nil {
		t.Fatalf("SaveBinaryModel failed: %v", err)
	}

	router, err := NewRouter(modelPath, 0.50)
	if err != nil {
		t.Fatalf("NewRouter failed: %v", err)
	}

	// 1. Baseline: T = 1.0
	router.SetTemperature(1.0)
	if router.Temperature() != 1.0 {
		t.Fatalf("expected router temperature 1.0, got %v", router.Temperature())
	}
	traceBase := router.Inspect("request refund on recent order")

	// 2. High Temperature: T = 2.0 -> Softens logits, reduces confidence, increases entropy
	router.SetTemperature(2.0)
	if router.Temperature() != 2.0 {
		t.Fatalf("expected router temperature 2.0, got %v", router.Temperature())
	}
	traceWarm := router.Inspect("request refund on recent order")

	if traceWarm.Confidence >= traceBase.Confidence {
		t.Errorf("Higher temperature (T=2.0) should reduce peak confidence: base=%.4f, warm=%.4f",
			traceBase.Confidence, traceWarm.Confidence)
	}
	if traceWarm.Entropy <= traceBase.Entropy {
		t.Errorf("Higher temperature (T=2.0) should increase entropy: base=%.4f, warm=%.4f",
			traceBase.Entropy, traceWarm.Entropy)
	}

	// 3. Low Temperature: T = 0.5 -> Sharpens logits, increases confidence, reduces entropy
	router.SetTemperature(0.5)
	traceCool := router.Inspect("request refund on recent order")

	if traceCool.Confidence <= traceBase.Confidence {
		t.Errorf("Lower temperature (T=0.5) should increase peak confidence: base=%.4f, cool=%.4f",
			traceBase.Confidence, traceCool.Confidence)
	}
	if traceCool.Entropy >= traceBase.Entropy {
		t.Errorf("Lower temperature (T=0.5) should decrease entropy: base=%.4f, cool=%.4f",
			traceBase.Entropy, traceCool.Entropy)
	}
}

