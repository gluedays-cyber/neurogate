package neurogate

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

var (
	// MagicBytes is the 4-byte signature at the beginning of an IntelliBranch binary model ("IBRN").
	MagicBytes = [4]byte{'I', 'B', 'R', 'N'}

	// CurrentFormatVersion defines the supported serialization format version (v2 supports Positional Embeddings).
	CurrentFormatVersion uint32 = 2

	ErrInvalidMagic     = errors.New("invalid binary format: missing IBRN magic header")
	ErrUnsupportedVer   = errors.New("unsupported model format version")
	ErrChecksumFailed   = errors.New("checksum verification failed: model file corrupted")
	ErrInvalidTensorDim = errors.New("tensor dimension does not match header configuration")
	ErrEmptyInput       = errors.New("input token slice cannot be empty")
)

// Header contains the structural hyperparameters of the embedded neural model.
type Header struct {
	Magic        [4]byte
	Version      uint32
	VocabSize    uint32
	EmbeddingDim uint32
	HiddenDim    uint32
	NumClasses   uint32
}

// MergeRule represents a pair-to-target token merge rule for BPE.
type MergeRule struct {
	Token1 uint32
	Token2 uint32
	Target uint32
}

// Weights holds the linear algebra parameters for the embedding, positional encoding, and 2-layer MLP.
type Weights struct {
	Embedding  []float32 // [VocabSize * EmbeddingDim]
	Positional []float32 // [MaxSequenceTokens * EmbeddingDim]
	W1         []float32 // [EmbeddingDim * HiddenDim]
	B1         []float32 // [HiddenDim]
	W2         []float32 // [HiddenDim * NumClasses]
	B2         []float32 // [NumClasses]
}

// SaveBinaryModel serializes an InferenceModel into a Little-Endian binary file with SHA-256 validation.
func SaveBinaryModel(filePath string, model *InferenceModel) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer file.Close()

	if err := SerializeModel(file, model); err != nil {
		return fmt.Errorf("failed to serialize model: %w", err)
	}
	return nil
}

// LoadBinaryModel reads and validates an IntelliBranch binary model from the filesystem.
func LoadBinaryModel(filePath string) (*InferenceModel, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	return DeserializeModel(file)
}

// SerializeModel serializes the model structure into Little-Endian bytes and appends a SHA-256 checksum.
func SerializeModel(w io.Writer, model *InferenceModel) error {
	hasher := sha256.New()
	mw := io.MultiWriter(w, hasher)

	// 1. Write Header Block
	if _, err := mw.Write(model.Header.Magic[:]); err != nil {
		return err
	}
	if err := binary.Write(mw, binary.LittleEndian, model.Header.Version); err != nil {
		return err
	}
	if err := binary.Write(mw, binary.LittleEndian, model.Header.VocabSize); err != nil {
		return err
	}
	if err := binary.Write(mw, binary.LittleEndian, model.Header.EmbeddingDim); err != nil {
		return err
	}
	if err := binary.Write(mw, binary.LittleEndian, model.Header.HiddenDim); err != nil {
		return err
	}
	if err := binary.Write(mw, binary.LittleEndian, model.Header.NumClasses); err != nil {
		return err
	}

	// 2. Write Labels Block
	if err := binary.Write(mw, binary.LittleEndian, uint32(len(model.Labels))); err != nil {
		return err
	}
	for _, label := range model.Labels {
		strBytes := []byte(label)
		if err := binary.Write(mw, binary.LittleEndian, uint32(len(strBytes))); err != nil {
			return err
		}
		if _, err := mw.Write(strBytes); err != nil {
			return err
		}
	}

	// 3. Write Vocabulary Block
	if err := binary.Write(mw, binary.LittleEndian, uint32(len(model.Vocab))); err != nil {
		return err
	}
	for _, token := range model.Vocab {
		strBytes := []byte(token)
		if err := binary.Write(mw, binary.LittleEndian, uint32(len(strBytes))); err != nil {
			return err
		}
		if _, err := mw.Write(strBytes); err != nil {
			return err
		}
	}

	// 4. Write BPE Merge Rules Block
	if err := binary.Write(mw, binary.LittleEndian, uint32(len(model.MergeRules))); err != nil {
		return err
	}
	for _, rule := range model.MergeRules {
		if err := binary.Write(mw, binary.LittleEndian, rule.Token1); err != nil {
			return err
		}
		if err := binary.Write(mw, binary.LittleEndian, rule.Token2); err != nil {
			return err
		}
		if err := binary.Write(mw, binary.LittleEndian, rule.Target); err != nil {
			return err
		}
	}

	// 5. Write Tensor Blocks (IEEE 754 float32 Little-Endian)
	if err := binary.Write(mw, binary.LittleEndian, model.Weights.Embedding); err != nil {
		return err
	}
	// Write Positional Embeddings only for format version >= 2
	if model.Header.Version >= 2 {
		posLen := int(MaxSequenceTokens * model.Header.EmbeddingDim)
		posWeights := model.Weights.Positional
		if len(posWeights) < posLen {
			paddedPos := make([]float32, posLen)
			copy(paddedPos, posWeights)
			posWeights = paddedPos
		}
		if err := binary.Write(mw, binary.LittleEndian, posWeights[:posLen]); err != nil {
			return err
		}
	}

	if err := binary.Write(mw, binary.LittleEndian, model.Weights.W1); err != nil {
		return err
	}
	if err := binary.Write(mw, binary.LittleEndian, model.Weights.B1); err != nil {
		return err
	}
	if err := binary.Write(mw, binary.LittleEndian, model.Weights.W2); err != nil {
		return err
	}
	if err := binary.Write(mw, binary.LittleEndian, model.Weights.B2); err != nil {
		return err
	}

	// 6. Write Integrity Checksum (SHA-256 32 bytes)
	computedChecksum := hasher.Sum(nil)
	if _, err := w.Write(computedChecksum); err != nil {
		return err
	}

	return nil
}

// DeserializeModel parses Little-Endian bytes, validates header & SHA-256 checksum, and instantiates an InferenceModel.
func DeserializeModel(r io.Reader) (*InferenceModel, error) {
	hasher := sha256.New()
	tr := io.TeeReader(r, hasher)

	// 1. Read Header Block
	var header Header
	if _, err := io.ReadFull(tr, header.Magic[:]); err != nil {
		return nil, fmt.Errorf("failed to read magic header: %w", err)
	}
	if header.Magic != MagicBytes {
		return nil, ErrInvalidMagic
	}

	if err := binary.Read(tr, binary.LittleEndian, &header.Version); err != nil {
		return nil, err
	}
	if header.Version != 1 && header.Version != 2 {
		return nil, fmt.Errorf("%w: got %d, expected version 1 or 2", ErrUnsupportedVer, header.Version)
	}

	if err := binary.Read(tr, binary.LittleEndian, &header.VocabSize); err != nil {
		return nil, err
	}
	if err := binary.Read(tr, binary.LittleEndian, &header.EmbeddingDim); err != nil {
		return nil, err
	}
	if err := binary.Read(tr, binary.LittleEndian, &header.HiddenDim); err != nil {
		return nil, err
	}
	if err := binary.Read(tr, binary.LittleEndian, &header.NumClasses); err != nil {
		return nil, err
	}

	// 2. Read Labels Block
	var numLabels uint32
	if err := binary.Read(tr, binary.LittleEndian, &numLabels); err != nil {
		return nil, err
	}
	if numLabels != header.NumClasses {
		return nil, fmt.Errorf("num labels (%d) does not match NumClasses (%d)", numLabels, header.NumClasses)
	}

	labels := make([]string, numLabels)
	for i := uint32(0); i < numLabels; i++ {
		var strLen uint32
		if err := binary.Read(tr, binary.LittleEndian, &strLen); err != nil {
			return nil, err
		}
		strBuf := make([]byte, strLen)
		if _, err := io.ReadFull(tr, strBuf); err != nil {
			return nil, err
		}
		labels[i] = string(strBuf)
	}

	// 3. Read Vocabulary Block
	var vocabCount uint32
	if err := binary.Read(tr, binary.LittleEndian, &vocabCount); err != nil {
		return nil, err
	}
	if vocabCount != header.VocabSize {
		return nil, fmt.Errorf("vocab count (%d) does not match VocabSize (%d)", vocabCount, header.VocabSize)
	}

	vocab := make([]string, vocabCount)
	for i := uint32(0); i < vocabCount; i++ {
		var strLen uint32
		if err := binary.Read(tr, binary.LittleEndian, &strLen); err != nil {
			return nil, err
		}
		strBuf := make([]byte, strLen)
		if _, err := io.ReadFull(tr, strBuf); err != nil {
			return nil, err
		}
		vocab[i] = string(strBuf)
	}

	// 4. Read Merge Rules Block
	var numRules uint32
	if err := binary.Read(tr, binary.LittleEndian, &numRules); err != nil {
		return nil, err
	}
	mergeRules := make([]MergeRule, numRules)
	for i := uint32(0); i < numRules; i++ {
		if err := binary.Read(tr, binary.LittleEndian, &mergeRules[i].Token1); err != nil {
			return nil, err
		}
		if err := binary.Read(tr, binary.LittleEndian, &mergeRules[i].Token2); err != nil {
			return nil, err
		}
		if err := binary.Read(tr, binary.LittleEndian, &mergeRules[i].Target); err != nil {
			return nil, err
		}
	}

	// 5. Read Tensor Blocks
	posLen := int(MaxSequenceTokens * header.EmbeddingDim)
	weights := Weights{
		Embedding:  make([]float32, header.VocabSize*header.EmbeddingDim),
		Positional: make([]float32, posLen),
		W1:         make([]float32, header.EmbeddingDim*header.HiddenDim),
		B1:         make([]float32, header.HiddenDim),
		W2:         make([]float32, header.HiddenDim*header.NumClasses),
		B2:         make([]float32, header.NumClasses),
	}

	if err := binary.Read(tr, binary.LittleEndian, weights.Embedding); err != nil {
		return nil, fmt.Errorf("failed to read embedding table: %w", err)
	}
	if header.Version >= 2 {
		if err := binary.Read(tr, binary.LittleEndian, weights.Positional); err != nil {
			return nil, fmt.Errorf("failed to read positional table: %w", err)
		}
	}
	if err := binary.Read(tr, binary.LittleEndian, weights.W1); err != nil {
		return nil, fmt.Errorf("failed to read W1: %w", err)
	}
	if err := binary.Read(tr, binary.LittleEndian, weights.B1); err != nil {
		return nil, fmt.Errorf("failed to read B1: %w", err)
	}
	if err := binary.Read(tr, binary.LittleEndian, weights.W2); err != nil {
		return nil, fmt.Errorf("failed to read W2: %w", err)
	}
	if err := binary.Read(tr, binary.LittleEndian, weights.B2); err != nil {
		return nil, fmt.Errorf("failed to read B2: %w", err)
	}

	// 6. Verify Checksum Block
	var storedChecksum [32]byte
	if _, err := io.ReadFull(r, storedChecksum[:]); err != nil {
		return nil, fmt.Errorf("failed to read checksum: %w", err)
	}

	computedChecksum := hasher.Sum(nil)
	for i := 0; i < 32; i++ {
		if storedChecksum[i] != computedChecksum[i] {
			return nil, ErrChecksumFailed
		}
	}

	return NewInferenceModel(header, labels, vocab, mergeRules, weights), nil
}
