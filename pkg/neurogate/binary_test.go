package neurogate

import (
	"bytes"
	"path/filepath"
	"testing"
)

func createSampleModel() *InferenceModel {
	header := Header{
		Magic:        MagicBytes,
		Version:      1,
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

	weights := Weights{
		Embedding: make([]float32, header.VocabSize*header.EmbeddingDim),
		W1:        make([]float32, header.EmbeddingDim*header.HiddenDim),
		B1:        make([]float32, header.HiddenDim),
		W2:        make([]float32, header.HiddenDim*header.NumClasses),
		B2:        make([]float32, header.NumClasses),
	}

	// Fill with deterministic dummy values
	for i := range weights.Embedding {
		weights.Embedding[i] = float32(i) * 0.05
	}
	for i := range weights.W1 {
		weights.W1[i] = float32(i) * 0.02
	}
	for i := range weights.B1 {
		weights.B1[i] = 0.1
	}
	for i := range weights.W2 {
		weights.W2[i] = float32(i) * 0.03
	}
	for i := range weights.B2 {
		weights.B2[i] = 0.2
	}

	return NewInferenceModel(header, labels, vocab, mergeRules, weights)
}

func TestSerializationRoundTrip(t *testing.T) {
	origModel := createSampleModel()

	var buf bytes.Buffer
	if err := SerializeModel(&buf, origModel); err != nil {
		t.Fatalf("SerializeModel failed: %v", err)
	}

	rawBytes := buf.Bytes()
	if len(rawBytes) == 0 {
		t.Fatal("Serialized buffer is empty")
	}

	parsedModel, err := DeserializeModel(bytes.NewReader(rawBytes))
	if err != nil {
		t.Fatalf("DeserializeModel failed: %v", err)
	}

	// Verify Header
	if parsedModel.Header != origModel.Header {
		t.Errorf("Header mismatch: %+v vs %+v", parsedModel.Header, origModel.Header)
	}

	// Verify Labels
	if len(parsedModel.Labels) != len(origModel.Labels) {
		t.Fatalf("Labels length mismatch: %d vs %d", len(parsedModel.Labels), len(origModel.Labels))
	}
	for i := range origModel.Labels {
		if parsedModel.Labels[i] != origModel.Labels[i] {
			t.Errorf("Label mismatch at %d: %s vs %s", i, parsedModel.Labels[i], origModel.Labels[i])
		}
	}

	// Verify Vocab
	if len(parsedModel.Vocab) != len(origModel.Vocab) {
		t.Fatalf("Vocab length mismatch: %d vs %d", len(parsedModel.Vocab), len(origModel.Vocab))
	}
	for i := range origModel.Vocab {
		if parsedModel.Vocab[i] != origModel.Vocab[i] {
			t.Errorf("Vocab mismatch at %d: %s vs %s", i, parsedModel.Vocab[i], origModel.Vocab[i])
		}
	}

	// Verify MergeRules
	if len(parsedModel.MergeRules) != len(origModel.MergeRules) {
		t.Fatalf("MergeRules length mismatch: %d vs %d", len(parsedModel.MergeRules), len(origModel.MergeRules))
	}
	for i := range origModel.MergeRules {
		if parsedModel.MergeRules[i] != origModel.MergeRules[i] {
			t.Errorf("MergeRule mismatch at %d", i)
		}
	}

	// Verify Weights
	for i := range origModel.Weights.Embedding {
		if parsedModel.Weights.Embedding[i] != origModel.Weights.Embedding[i] {
			t.Errorf("Embedding weight mismatch at %d", i)
		}
	}
	for i := range origModel.Weights.W1 {
		if parsedModel.Weights.W1[i] != origModel.Weights.W1[i] {
			t.Errorf("W1 weight mismatch at %d", i)
		}
	}
	for i := range origModel.Weights.B1 {
		if parsedModel.Weights.B1[i] != origModel.Weights.B1[i] {
			t.Errorf("B1 weight mismatch at %d", i)
		}
	}
	for i := range origModel.Weights.W2 {
		if parsedModel.Weights.W2[i] != origModel.Weights.W2[i] {
			t.Errorf("W2 weight mismatch at %d", i)
		}
	}
	for i := range origModel.Weights.B2 {
		if parsedModel.Weights.B2[i] != origModel.Weights.B2[i] {
			t.Errorf("B2 weight mismatch at %d", i)
		}
	}
}

func TestInvalidMagicRejection(t *testing.T) {
	origModel := createSampleModel()
	var buf bytes.Buffer
	if err := SerializeModel(&buf, origModel); err != nil {
		t.Fatalf("SerializeModel failed: %v", err)
	}

	data := buf.Bytes()
	// Corrupt magic bytes
	data[0] = 'X'

	_, err := DeserializeModel(bytes.NewReader(data))
	if err == nil {
		t.Fatal("Expected error for corrupted magic bytes, got nil")
	}
}

func TestChecksumVerificationFailure(t *testing.T) {
	origModel := createSampleModel()
	var buf bytes.Buffer
	if err := SerializeModel(&buf, origModel); err != nil {
		t.Fatalf("SerializeModel failed: %v", err)
	}

	data := buf.Bytes()
	// Corrupt a byte in the tensor section (before checksum at end)
	data[len(data)-40] ^= 0xFF

	_, err := DeserializeModel(bytes.NewReader(data))
	if err == nil {
		t.Fatal("Expected checksum verification error, got nil")
	}
	if err != ErrChecksumFailed {
		t.Logf("Received expected checksum failure error: %v", err)
	}
}

func TestFileIO(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "test_model.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("SaveBinaryModel failed: %v", err)
	}

	loadedModel, err := LoadBinaryModel(modelPath)
	if err != nil {
		t.Fatalf("LoadBinaryModel failed: %v", err)
	}

	if loadedModel.Header.VocabSize != origModel.Header.VocabSize {
		t.Errorf("VocabSize mismatch: %d vs %d", loadedModel.Header.VocabSize, origModel.Header.VocabSize)
	}
}

func TestFormatVersion2RoundTrip(t *testing.T) {
	header := Header{
		Magic:        MagicBytes,
		Version:      CurrentFormatVersion, // Version 2
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

	// Set distinct positional values
	for i := range weights.Positional {
		weights.Positional[i] = float32(i+1) * 0.01
	}

	v2Model := NewInferenceModel(header, labels, vocab, mergeRules, weights)

	var buf bytes.Buffer
	if err := SerializeModel(&buf, v2Model); err != nil {
		t.Fatalf("SerializeModel v2 failed: %v", err)
	}

	loaded, err := DeserializeModel(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("DeserializeModel v2 failed: %v", err)
	}

	if loaded.Header.Version != 2 {
		t.Errorf("Expected version 2, got %d", loaded.Header.Version)
	}

	// Verify positional weights integrity
	for i := 0; i < posLen; i++ {
		if loaded.Weights.Positional[i] != weights.Positional[i] {
			t.Fatalf("Positional weight mismatch at index %d: expected %f, got %f", i, weights.Positional[i], loaded.Weights.Positional[i])
		}
	}
}
