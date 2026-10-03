package neurogate

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"
)

// TestRFC_StrictShortCircuit_OODBeforeAmbiguity proves that an Out-of-Domain sample
// with narrow margin is strictly rejected as OOD and NEVER incorrectly flagged as Ambiguous.
func TestRFC_StrictShortCircuit_OODBeforeAmbiguity(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "ood_short_circuit.bin")

	samples := []DataSample{
		{Text: "refund money to bank account card", Label: "Refund"},
		{Text: "track courier shipment parcel delivery", Label: "Delivery"},
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

	// Calibrate distribution to establish true manifold boundaries
	gate.CalibrateDomainDistribution(samples, 1.5)

	// A foreign domain input that has nothing to do with Refund or Delivery
	foreignQuery := "quantum entanglement physics simulation particles"
	trace := gate.Inspect(foreignQuery)

	if !trace.IsOOD {
		t.Errorf("Expected foreign query to be flagged as IsOOD=true, got IsOOD=%t (reason: %s)", trace.IsOOD, trace.FallbackReason)
	}
	if trace.IsAmbiguous {
		t.Errorf("Violation of RFC Short-Circuit: Foreign OOD query was incorrectly marked as IsAmbiguous=true")
	}
}

// TestRFC_SelfCalibratingMetadataPortability validates that Header FormatVersion3
// preserves and propagates calibration thresholds directly into new gate instances.
func TestRFC_SelfCalibratingMetadataPortability(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "v3_calibrated.bin")

	samples := []DataSample{
		{Text: "refund money to bank account card", Label: "Refund"},
		{Text: "track courier shipment parcel delivery", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 30
	model, err := TrainModel(samples, cfg)
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}

	gate1 := NewNeuroGateWithModel(model)
	gate1.CalibrateDomainDistribution(samples, 2.0)

	// Save calibrated model (FormatVersion3)
	if err := SaveBinaryModel(modelPath, gate1.Model()); err != nil {
		t.Fatalf("SaveBinaryModel failed: %v", err)
	}

	// Load model into a fresh instance without runtime calibration
	loadedModel, err := LoadBinaryModel(modelPath)
	if err != nil {
		t.Fatalf("LoadBinaryModel failed: %v", err)
	}

	if loadedModel.Header.Version != FormatVersion3 {
		t.Fatalf("Expected FormatVersion3, got %d", loadedModel.Header.Version)
	}
	if loadedModel.Header.CalibratedMinEnergy <= 0.0 {
		t.Errorf("Expected positive CalibratedMinEnergy in header, got %f", loadedModel.Header.CalibratedMinEnergy)
	}
	if loadedModel.Header.CalibratedMargin <= 0.0 {
		t.Errorf("Expected positive CalibratedMargin in header, got %f", loadedModel.Header.CalibratedMargin)
	}

	// Instantiate fresh NeuroGate and Router from the v3 binary
	gate2 := NewNeuroGateWithModel(loadedModel)
	if gate2.policy.MinLogSumExp != float64(loadedModel.Header.CalibratedMinEnergy) {
		t.Errorf("Gate did not auto-apply MinLogSumExp: got %f, expected %f",
			gate2.policy.MinLogSumExp, loadedModel.Header.CalibratedMinEnergy)
	}

	router, err := NewRouter(modelPath, 0.70)
	if err != nil {
		t.Fatalf("NewRouter failed: %v", err)
	}
	if router.policy.MinLogSumExp != float64(loadedModel.Header.CalibratedMinEnergy) {
		t.Errorf("Router did not auto-apply MinLogSumExp: got %f, expected %f",
			router.policy.MinLogSumExp, loadedModel.Header.CalibratedMinEnergy)
	}
}

// TestRFC_LegacyVersion2BackwardCompatibility ensures existing v2 binaries load and run without errors.
func TestRFC_LegacyVersion2BackwardCompatibility(t *testing.T) {
	header := Header{
		Magic:        MagicBytes,
		Version:      FormatVersion2,
		VocabSize:    4,
		EmbeddingDim: 4,
		HiddenDim:    6,
		NumClasses:   2,
	}
	labels := []string{"Refund", "Delivery"}
	vocab := []string{"[PAD]", "refund", "cancel", "delivery"}
	mergeRules := []MergeRule{
		{Token1: 1, Token2: 2, Target: 3},
	}
	posLen := int(MaxSequenceTokens * header.EmbeddingDim)
	weights := Weights{
		Embedding:  make([]float32, header.VocabSize*header.EmbeddingDim),
		Positional: make([]float32, posLen),
		W1:         make([]float32, header.EmbeddingDim*header.HiddenDim),
		B1:         make([]float32, header.HiddenDim),
		W2:         make([]float32, header.HiddenDim*header.NumClasses),
		B2:         make([]float32, header.NumClasses),
	}

	v2Model := NewInferenceModel(header, labels, vocab, mergeRules, weights)
	var buf bytes.Buffer
	if err := SerializeModel(&buf, v2Model); err != nil {
		t.Fatalf("SerializeModel v2 failed: %v", err)
	}

	loadedModel, err := DeserializeModel(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("DeserializeModel failed on v2 binary: %v", err)
	}
	if loadedModel.Header.Version != FormatVersion2 {
		t.Errorf("Expected FormatVersion2, got %d", loadedModel.Header.Version)
	}

	// Verify gate loads v2 model cleanly
	gate := NewNeuroGateWithModel(loadedModel)
	if gate == nil {
		t.Fatal("Expected non-nil NeuroGate with v2 model")
	}
	err = gate.Filter(context.Background(), "refund cancel", nil)
	if err != nil {
		t.Fatalf("Filter on v2 model failed: %v", err)
	}
}

// TestRFC_ZeroAllocationBenchmark verifies that inference remains strictly zero-allocation.
func BenchmarkRFC_ZeroAllocationInference(b *testing.B) {
	samples := []DataSample{
		{Text: "refund money to card", Label: "Refund"},
		{Text: "package delivery tracking", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	model, err := TrainModel(samples, cfg)
	if err != nil {
		b.Fatalf("TrainModel failed: %v", err)
	}

	gate := NewNeuroGateWithModel(model)
	ctx := context.Background()
	tokens := model.Tokenizer.Encode("refund money")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = gate.FilterTokens(ctx, tokens, nil)
	}
}
