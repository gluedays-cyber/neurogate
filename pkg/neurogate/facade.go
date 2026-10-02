package neurogate

import (
	"fmt"
	"os"
	"path/filepath"
)

// Train executes the complete pipeline: dataset loading, BPE tokenization,
// AdamW neural training, directory creation, and Little-Endian binary model serialization.
func Train(csvPath string, outputPath string, configs ...TrainConfig) error {
	samples, err := LoadCSVDataset(csvPath)
	if err != nil {
		return fmt.Errorf("failed to load dataset from %s: %w", csvPath, err)
	}

	cfg := DefaultTrainConfig()
	if len(configs) > 0 {
		cfg = configs[0]
	}

	model, err := TrainModel(samples, cfg)
	if err != nil {
		return fmt.Errorf("model training failed: %w", err)
	}

	outDir := filepath.Dir(outputPath)
	if outDir != "" && outDir != "." {
		if err := os.MkdirAll(outDir, 0755); err != nil {
			return fmt.Errorf("failed to create output directory %s: %w", outDir, err)
		}
	}

	if err := SaveBinaryModel(outputPath, model); err != nil {
		return fmt.Errorf("failed to save binary model to %s: %w", outputPath, err)
	}

	return nil
}

// EnsureModel guarantees the binary model exists at modelPath.
// If it does not exist, it trains a new model from csvPath and serializes it.
func EnsureModel(csvPath string, modelPath string, configs ...TrainConfig) error {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return Train(csvPath, modelPath, configs...)
	}
	return nil
}

// Open loads an existing compiled binary model file into memory and creates an in-memory Router.
// If threshold is omitted, the default calibrated threshold of 0.60 is applied.
func Open(modelPath string, defaultThreshold ...float64) (*Router, error) {
	threshold := 0.60
	if len(defaultThreshold) > 0 && defaultThreshold[0] > 0.0 {
		threshold = defaultThreshold[0]
	}
	return NewRouter(modelPath, threshold)
}

// OpenOrTrain ensures the binary model is trained from csvPath if missing,
// then loads it directly into an active in-memory Router for intelligent branching.
func OpenOrTrain(csvPath string, modelPath string, defaultThreshold ...float64) (*Router, error) {
	if err := EnsureModel(csvPath, modelPath); err != nil {
		return nil, fmt.Errorf("failed to ensure model: %w", err)
	}
	return Open(modelPath, defaultThreshold...)
}
