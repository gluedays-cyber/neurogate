package neurogate

import (
	core "github.com/gluedays-cyber/neurogate/pkg/neurogate"
)

// Re-export core types for top-level root import
type (
	Router              = core.Router
	RouteAction         = core.RouteAction
	AmbiguousAction     = core.AmbiguousAction
	PipelineAction      = core.PipelineAction
	DispatchPolicy      = core.DispatchPolicy
	RouteDecision       = core.RouteDecision
	RouteTrace          = core.RouteTrace
	TrainConfig         = core.TrainConfig
	DataSample          = core.DataSample
	InferenceModel      = core.InferenceModel
	NeuroGate           = core.NeuroGate
	TelemetryEvent      = core.TelemetryEvent
	TelemetryRingBuffer = core.TelemetryRingBuffer
)

// Sentinel Errors
var (
	ErrModelNotInitialized = core.ErrModelNotInitialized
	ErrClassIndexOutOfRange = core.ErrClassIndexOutOfRange
	ErrUnlearnedVocabulary = core.ErrUnlearnedVocabulary
	ErrLowConfidence       = core.ErrLowConfidence
	ErrHighEntropy         = core.ErrHighEntropy
	ErrOutOfDomain         = core.ErrOutOfDomain
	ErrAmbiguousIntent     = core.ErrAmbiguousIntent
)

// High-level functions exported at the package root
var (
	// High-Level Facade APIs (Zero-boilerplate Train & Route)
	Train       = core.Train
	EnsureModel = core.EnsureModel
	Open        = core.Open
	OpenOrTrain = core.OpenOrTrain

	// Core Training & Dataset Helpers
	DefaultTrainConfig = core.DefaultTrainConfig
	LoadCSVDataset     = core.LoadCSVDataset
	TrainModel         = core.TrainModel

	// Binary Serialization
	SaveBinaryModel = core.SaveBinaryModel
	LoadBinaryModel = core.LoadBinaryModel

	// In-Memory Routing Engines
	NewRouter             = core.NewRouter
	NewNeuroGate          = core.NewNeuroGate
	DefaultDispatchPolicy = core.DefaultDispatchPolicy

	// Math & Ops
	LogSumExp        = core.LogSumExp
	CosineSimilarity = core.CosineSimilarity
)

