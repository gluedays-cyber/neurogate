package neurogate

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
)

// Router coordinates in-memory inference routing with atomic hot-swap, 3-tier safety, and telemetry feedback.
type Router struct {
	mu              sync.RWMutex
	model           atomic.Pointer[InferenceModel]
	threshold       float64
	policy          DispatchPolicy
	routes          map[string]RouteAction
	pipelines       map[string]PipelineAction
	defaultPipeline PipelineAction
	ambiguous       AmbiguousAction
	fallback        RouteAction
	telemetry       *TelemetryRingBuffer
}

// NewRouter loads a binary model file into memory once and constructs an immutable routing core.
func NewRouter(modelPath string, defaultThreshold float64) (*Router, error) {
	model, err := LoadBinaryModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize inference model: %w", err)
	}

	policy := DefaultDispatchPolicy()
	if defaultThreshold > 0.0 {
		policy.HighThreshold = defaultThreshold
		policy.LowThreshold = defaultThreshold * 0.6
	}

	// Auto-apply self-calibrated thresholds from model header if present
	if model.Header.CalibratedMinEnergy > 0 {
		policy.MinLogSumExp = float64(model.Header.CalibratedMinEnergy)
	}
	if model.Header.CalibratedMargin > 0 {
		policy.RawLogitMargin = model.Header.CalibratedMargin
	}

	router := &Router{
		threshold: defaultThreshold,
		policy:    policy,
		routes:    make(map[string]RouteAction),
		pipelines: make(map[string]PipelineAction),
		fallback: func(ctx context.Context, payload any) error {
			return nil
		},
		telemetry: NewTelemetryRingBuffer(1024),
	}
	router.model.Store(model)
	return router, nil
}

// Reload parses, validates, and atomically swaps model weights without interrupting active traffic.
func (r *Router) Reload(modelPath string) error {
	newModel, err := LoadBinaryModel(modelPath)
	if err != nil {
		return fmt.Errorf("failed to reload model: %w", err)
	}

	r.model.Store(newModel)
	return nil
}

// SwapModel replaces the active inference model atomically.
func (r *Router) SwapModel(newModel *InferenceModel) {
	r.model.Store(newModel)
}

// Model returns the currently active InferenceModel snapshot.
func (r *Router) Model() *InferenceModel {
	return r.model.Load()
}

// EnableTelemetry configures or resizes the telemetry ring buffer for active learning feedback.
func (r *Router) EnableTelemetry(capacity int) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.telemetry = NewTelemetryRingBuffer(capacity)
	return r
}

// DrainTelemetry extracts all recorded routing events for drift monitoring and active learning retraining.
func (r *Router) DrainTelemetry() []TelemetryEvent {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.telemetry == nil {
		return nil
	}
	return r.telemetry.Drain()
}

func (r *Router) recordTelemetry(text, primary, secondary string, conf, entropy float64, isAmbiguous, isPipeline, isFallback bool) {
	if r.telemetry != nil && (isAmbiguous || isPipeline || isFallback) {
		r.telemetry.Push(TelemetryEvent{
			InputText:      text,
			PredictedLabel: primary,
			SecondaryLabel: secondary,
			Confidence:     conf,
			Entropy:        entropy,
			IsAmbiguous:    isAmbiguous,
			IsPipeline:     isPipeline,
			IsFallback:     isFallback,
		})
	}
}

// SetPolicy updates the 3-tier routing thresholds and OOD entropy boundary.
func (r *Router) SetPolicy(policy DispatchPolicy) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.policy = policy
	return r
}

// Bind registers an action handler for a target class label.
func (r *Router) Bind(label string, action RouteAction) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.routes[label] = action
	return r
}

// Branch is an alias for Bind, providing a semantic and intuitive syntax for registering intelligent branches.
func (r *Router) Branch(label string, action RouteAction) *Router {
	return r.Bind(label, action)
}

// BindPipeline registers a specific multi-intent pipeline handler for a primary and secondary label pair.
func (r *Router) BindPipeline(primary string, secondary string, action PipelineAction) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pipelines[pipelineKey(primary, secondary)] = action
	return r
}

// DefaultPipeline registers a fallback multi-intent handler executed when no specific pair is bound.
func (r *Router) DefaultPipeline(action PipelineAction) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.defaultPipeline = action
	return r
}

// Ambiguous registers the handler triggered when confidence is between thresholds or top-1/top-2 margin is narrow.
func (r *Router) Ambiguous(action AmbiguousAction) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ambiguous = action
	return r
}

// Fallback registers the default handler triggered when confidence is below threshold, OOD, or label is unmatched.
func (r *Router) Fallback(action RouteAction) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fallback = action
	return r
}

// SetSingleCharRatioCutoff configures the Layer 1 unlearned single-character token ratio threshold.
func (r *Router) SetSingleCharRatioCutoff(cutoff float64) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.policy.MaxSingleCharRatio = cutoff
	return r
}

// SetTemperature configures the temperature scaling factor used in softmax calculations.
func (r *Router) SetTemperature(t float32) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	if m := r.model.Load(); m != nil {
		m.Temperature = t
	}
	return r
}

// Temperature returns the current temperature scaling factor.
func (r *Router) Temperature() float32 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if m := r.model.Load(); m != nil {
		return m.Temperature
	}
	return 0
}
