package neurogate

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// RouteAction defines the execution handler signature for a matched branch.
type RouteAction func(ctx context.Context, payload any) error

// AmbiguousAction defines the handler signature for ambiguous requests with competing top-2 predictions.
type AmbiguousAction func(ctx context.Context, primary string, secondary string, payload any) error

// PipelineAction defines the execution signature when both primary and secondary intents are eligible for multi-intent handling.
type PipelineAction func(ctx context.Context, primary string, secondary string, payload any) error

// DispatchPolicy defines the 3-tier confidence criteria, multi-intent threshold, and OOD entropy boundary.
type DispatchPolicy struct {
	HighThreshold        float64 `json:"high_threshold"`            // Minimum confidence for definite execution (default: 0.75)
	LowThreshold         float64 `json:"low_threshold"`             // Minimum confidence below which request is isolated to Fallback (default: 0.40)
	MarginCutoff         float64 `json:"margin_cutoff"`             // Minimum required gap between Top-1 and Top-2 (default: 0.15)
	MaxEntropy           float64 `json:"max_entropy"`               // Maximum allowable prediction entropy before triggering OOD Fallback (default: 2.0)
	PipelineThreshold    float64 `json:"pipeline_threshold"`        // Minimum secondary confidence to qualify for multi-intent pipeline (default: 0.30)
	MinLogSumExp         float64 `json:"min_log_sum_exp,omitempty"` // Minimum log-sum-exp energy boundary before OOD isolation (0 disables)
	MaxSingleCharRatio   float64 `json:"max_single_char_ratio"`     // Layer 1: Max ratio of single-char fallback tokens (default: 0.70)
	MaxUnknownTokenRatio float64 `json:"max_unknown_token_ratio"`   // Layer 1: Max ratio of UNK tokens (default: 0.30)
}

// DefaultDispatchPolicy creates standard production-ready 3-tier routing criteria.
func DefaultDispatchPolicy() DispatchPolicy {
	return DispatchPolicy{
		HighThreshold:        0.75,
		LowThreshold:         0.40,
		MarginCutoff:         0.15,
		MaxEntropy:           2.0,
		PipelineThreshold:    0.30,
		MinLogSumExp:         0.0,
		MaxSingleCharRatio:   0.70,
		MaxUnknownTokenRatio: 0.30,
	}
}

// RouteDecision encapsulates the verified intent result and diagnostic signals for fail-safe error handling.
type RouteDecision struct {
	Intent              string             `json:"intent"`
	Confidence          float64            `json:"confidence"`
	Entropy             float64            `json:"entropy"`
	Energy              float64            `json:"energy"`
	Margin              float64            `json:"margin"`
	SingleCharRatio     float64            `json:"single_char_ratio"`
	UnknownTokenRatio   float64            `json:"unknown_token_ratio"`
	SecondaryIntent     string             `json:"secondary_intent,omitempty"`
	SecondaryConfidence float64            `json:"secondary_confidence,omitempty"`
}

// RouteTrace encapsulates comprehensive diagnostic metadata explaining a routing decision.
type RouteTrace struct {
	InputText          string             `json:"input_text"`
	TokenIDs           []uint32           `json:"token_ids"`
	Subwords           []string           `json:"subwords"`
	SingleCharRatio    float64            `json:"single_char_ratio"`
	UnknownTokenRatio  float64            `json:"unknown_token_ratio"`
	ClassProbabilities map[string]float32 `json:"class_probabilities"`
	PredictedLabel     string             `json:"predicted_label"`
	SecondaryLabel     string             `json:"secondary_label,omitempty"`
	Confidence         float64            `json:"confidence"`
	Margin             float64            `json:"margin"`
	Entropy            float64            `json:"entropy"`
	Energy             float64            `json:"energy"`
	Threshold          float64            `json:"threshold"`
	IsAmbiguous        bool               `json:"is_ambiguous"`
	IsPipeline         bool               `json:"is_pipeline"`
	IsFallback         bool               `json:"is_fallback"`
	FallbackReason     string             `json:"fallback_reason,omitempty"`
	LatencyMicros      int64              `json:"latency_micros"`
}


func pipelineKey(primary, secondary string) string {
	return primary + "->" + secondary
}

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

// RouteQuery executes 2-layer fail-safe evaluation and returns standard Go sentinel errors on failure.
func (r *Router) RouteQuery(ctx context.Context, text string) (RouteDecision, error) {
	select {
	case <-ctx.Done():
		return RouteDecision{}, ctx.Err()
	default:
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	if err := ctx.Err(); err != nil {
		return RouteDecision{}, err
	}

	model := r.model.Load()
	if model == nil {
		return RouteDecision{}, ErrModelNotInitialized
	}

	if !utf8.ValidString(text) {
		return RouteDecision{}, ErrEmptyInput
	}
	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	tokenIDs := model.Tokenizer.Encode(text)
	if len(tokenIDs) == 0 {
		return RouteDecision{}, ErrEmptyInput
	}
	if len(tokenIDs) > MaxSequenceTokens {
		tokenIDs = tokenIDs[:MaxSequenceTokens]
	}

	// -------------------------------------------------------------
	// [Layer 1 Guard] Tokenizer-level unlearned vocabulary cutoff (< 1 μs)
	// -------------------------------------------------------------
	singleRatio, unkRatio := model.Tokenizer.AnalyzeUnlearnedRatio(tokenIDs)
	if len(tokenIDs) >= 2 && r.policy.MaxSingleCharRatio > 0.0 && singleRatio >= r.policy.MaxSingleCharRatio {
		return RouteDecision{
			SingleCharRatio:   singleRatio,
			UnknownTokenRatio: unkRatio,
		}, ErrUnlearnedVocabulary
	}
	if r.policy.MaxUnknownTokenRatio > 0.0 && unkRatio >= r.policy.MaxUnknownTokenRatio {
		return RouteDecision{
			SingleCharRatio:   singleRatio,
			UnknownTokenRatio: unkRatio,
		}, ErrUnlearnedVocabulary
	}

	// -------------------------------------------------------------
	// [Layer 2 Guard] Neural forward pass & metric-based cutoff (~29 μs)
	// -------------------------------------------------------------
	res, err := model.PredictSlots(tokenIDs, model.Temperature)
	if err != nil {
		return RouteDecision{}, err
	}
	if res.Total == 0 {
		return RouteDecision{}, ErrOutOfDomain
	}

	primaryIdx := int(res.Primary.Index)
	if primaryIdx < 0 || primaryIdx >= len(model.Labels) {
		return RouteDecision{}, ErrClassIndexOutOfRange
	}

	primaryLabel := model.Labels[primaryIdx]
	primaryConf := float64(res.Primary.Confidence)
	entropy := float64(res.Entropy)
	energy := float64(res.Energy)

	var secondaryLabel string
	var secondaryConf float64
	if res.Total >= 2 {
		secIdx := int(res.Secondary.Index)
		if secIdx >= 0 && secIdx < len(model.Labels) {
			secondaryLabel = model.Labels[secIdx]
			secondaryConf = float64(res.Secondary.Confidence)
		}
	}
	margin := primaryConf - secondaryConf

	decision := RouteDecision{
		Intent:              primaryLabel,
		Confidence:          primaryConf,
		Entropy:             entropy,
		Energy:              energy,
		Margin:              margin,
		SingleCharRatio:     singleRatio,
		UnknownTokenRatio:   unkRatio,
		SecondaryIntent:     secondaryLabel,
		SecondaryConfidence: secondaryConf,
	}

	// 1. Energy boundary (LogSumExp)
	if r.policy.MinLogSumExp != 0.0 && energy < r.policy.MinLogSumExp {
		return decision, ErrOutOfDomain
	}

	// 2. High entropy boundary
	if entropy > r.policy.MaxEntropy {
		return decision, ErrHighEntropy
	}

	// 3. Low confidence boundary
	if primaryConf < r.policy.LowThreshold {
		return decision, ErrLowConfidence
	}

	// 4. Ambiguity margin boundary
	if primaryConf < r.policy.HighThreshold || margin < r.policy.MarginCutoff {
		return decision, ErrAmbiguousIntent
	}

	return decision, nil
}

// Dispatch executes microsecond inference and routes through a 3-tier decision pipeline (Definite / Ambiguous / Fallback).
func (r *Router) Dispatch(ctx context.Context, text string, payload any) error {
	// Guard 0: Check context cancellation immediately before acquiring lock
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	// Check context cancellation before inference
	if err := ctx.Err(); err != nil {
		return err
	}

	model := r.model.Load()
	if model == nil {
		r.recordTelemetry(text, "", "", 0, 0, false, false, true)
		return r.fallback(ctx, payload)
	}

	// Zero-allocation detailed inference
	res, unkRatio, err := model.PredictDetailed(text)
	if err != nil || res.Total == 0 {
		r.recordTelemetry(text, "", "", 0, 0, false, false, true)
		return r.fallback(ctx, payload)
	}

	tokenIDs := model.Tokenizer.Encode(text)
	singleRatio, _ := model.Tokenizer.AnalyzeUnlearnedRatio(tokenIDs)

	// Layer 1 Fallback Guard: check unlearned vocabulary single-character fragmenting
	if len(tokenIDs) >= 2 && r.policy.MaxSingleCharRatio > 0.0 && singleRatio >= r.policy.MaxSingleCharRatio {
		r.recordTelemetry(text, "", "", 0, 0, false, false, true)
		return r.fallback(ctx, payload)
	}

	primaryIdx := int(res.Primary.Index)
	if primaryIdx < 0 || primaryIdx >= len(model.Labels) {
		r.recordTelemetry(text, "", "", 0, 0, false, false, true)
		return r.fallback(ctx, payload)
	}
	primaryLabel := model.Labels[primaryIdx]
	primaryConf := float64(res.Primary.Confidence)

	var secondaryLabel string
	var secondaryConf float64
	if res.Total >= 2 {
		secIdx := int(res.Secondary.Index)
		if secIdx >= 0 && secIdx < len(model.Labels) {
			secondaryLabel = model.Labels[secIdx]
			secondaryConf = float64(res.Secondary.Confidence)
		}
	}
	margin := primaryConf - secondaryConf
	entropy := float64(res.Entropy)
	energy := float64(res.Energy)

	// Layer 2 Fallback Guard: Low confidence, excessive UNKs, high entropy (OOD), or low energy
	if primaryConf < r.policy.LowThreshold || unkRatio >= 0.5 || entropy > r.policy.MaxEntropy || (r.policy.MinLogSumExp != 0.0 && energy < r.policy.MinLogSumExp) {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, false, false, true)
		return r.fallback(ctx, payload)
	}

	// 2. Ambiguous Route: Borderline confidence or competitive margin gap
	isAmbiguous := primaryConf < r.policy.HighThreshold || margin < r.policy.MarginCutoff
	if isAmbiguous {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, true, false, false)
		if r.ambiguous != nil {
			return r.ambiguous(ctx, primaryLabel, secondaryLabel, payload)
		}
		return r.fallback(ctx, payload)
	}

	// 3. Definite Route: Confident, decisive margin, low entropy
	action, exists := r.routes[primaryLabel]
	if !exists {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, false, false, true)
		return r.fallback(ctx, payload)
	}

	// Final check before executing bound business handler
	if err := ctx.Err(); err != nil {
		return err
	}

	return action(ctx, payload)
}


// Inspect evaluates input text and generates a full diagnostic RouteTrace including 3-tier and entropy metrics.
func (r *Router) Inspect(text string) RouteTrace {
	start := time.Now()
	r.mu.RLock()
	defer r.mu.RUnlock()

	model := r.model.Load()
	if model == nil {
		return RouteTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "model not loaded",
			Threshold:      r.threshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	// Guard 1: Truncate oversized input strings respecting UTF-8 rune boundaries
	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	tokens := model.Tokenizer.Encode(text)
	if len(tokens) == 0 {
		return RouteTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "empty input tokens",
			Threshold:      r.threshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	// Guard 2: Clamp sequence length
	if len(tokens) > MaxSequenceTokens {
		tokens = tokens[:MaxSequenceTokens]
	}

	subwords := make([]string, len(tokens))
	unkCount := 0
	unkID, hasUnk := model.Tokenizer.VocabMap["[UNK]"]
	for i, id := range tokens {
		if hasUnk && id == unkID {
			unkCount++
		}
		if int(id) < len(model.Vocab) {
			subwords[i] = model.Vocab[id]
		}
	}

	singleRatio, unkRatio := model.Tokenizer.AnalyzeUnlearnedRatio(tokens)

	probs, err := model.Forward(tokens, model.Temperature)
	probMap := make(map[string]float32, len(model.Labels))

	var bestLabel, secondLabel string
	var bestScore, secondScore float32 = -1.0, -1.0
	for i, p := range probs {
		if i < len(model.Labels) {
			lbl := model.Labels[i]
			probMap[lbl] = p
			if p > bestScore {
				secondScore = bestScore
				secondLabel = bestLabel
				bestScore = p
				bestLabel = lbl
			} else if p > secondScore {
				secondScore = p
				secondLabel = lbl
			}
		}
	}

	// Guard 3: Apply calibrated confidence penalty
	effectivePenalty := unkRatio
	if singleRatio > 0.5 {
		effectivePenalty = math.Max(unkRatio, (singleRatio-0.5)*2.0)
	}
	calibratedConfidence := float64(bestScore) * (1.0 - effectivePenalty)
	calibratedSecond := float64(secondScore) * (1.0 - effectivePenalty)
	margin := calibratedConfidence - calibratedSecond
	entropy := float64(computeEntropy(probs))
	energy := float64(LogSumExp(probs))

	trace := RouteTrace{
		InputText:          text,
		TokenIDs:           tokens,
		Subwords:           subwords,
		SingleCharRatio:    singleRatio,
		UnknownTokenRatio:  unkRatio,
		ClassProbabilities: probMap,
		PredictedLabel:     bestLabel,
		SecondaryLabel:     secondLabel,
		Confidence:         calibratedConfidence,
		Margin:             margin,
		Entropy:            entropy,
		Energy:             energy,
		Threshold:          r.policy.HighThreshold,
		LatencyMicros:      time.Since(start).Microseconds(),
	}

	if err != nil {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("inference error: %v", err)
	} else if len(tokens) >= 2 && r.policy.MaxSingleCharRatio > 0.0 && singleRatio >= r.policy.MaxSingleCharRatio {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("unlearned vocabulary (single-char ratio %.2f >= %.2f)", singleRatio, r.policy.MaxSingleCharRatio)
	} else if trace.Confidence < r.policy.LowThreshold {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("confidence %.4f below low threshold %.4f", trace.Confidence, r.policy.LowThreshold)
	} else if unkRatio >= 0.5 {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("excessive unknown tokens (%.2f >= 0.50)", unkRatio)
	} else if entropy > r.policy.MaxEntropy {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("prediction entropy %.4f exceeds limit %.4f (OOD)", entropy, r.policy.MaxEntropy)
	} else if r.policy.MinLogSumExp != 0.0 && energy < r.policy.MinLogSumExp {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("energy %.4f below minimum threshold %.4f (OOD)", energy, r.policy.MinLogSumExp)
	} else {
		if secondLabel != "" && calibratedSecond >= r.policy.PipelineThreshold {
			trace.IsPipeline = true
		}
		if trace.Confidence < r.policy.HighThreshold || margin < r.policy.MarginCutoff {
			trace.IsAmbiguous = true
		}
		if _, exists := r.routes[bestLabel]; !exists {
			trace.IsFallback = true
			trace.FallbackReason = fmt.Sprintf("label '%s' has no bound route handler", bestLabel)
		}
	}

	return trace
}

// DispatchPipeline routes requests with multi-intent support, executing pipeline handlers when both top-1 and top-2 are eligible.
func (r *Router) DispatchPipeline(ctx context.Context, text string, payload any) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	if err := ctx.Err(); err != nil {
		return err
	}

	model := r.model.Load()
	if model == nil {
		r.recordTelemetry(text, "", "", 0, 0, false, false, true)
		return r.fallback(ctx, payload)
	}

	res, unkRatio, err := model.PredictDetailed(text)
	if err != nil || res.Total == 0 {
		r.recordTelemetry(text, "", "", 0, 0, false, false, true)
		return r.fallback(ctx, payload)
	}

	primaryIdx := int(res.Primary.Index)
	if primaryIdx < 0 || primaryIdx >= len(model.Labels) {
		r.recordTelemetry(text, "", "", 0, 0, false, false, true)
		return r.fallback(ctx, payload)
	}
	primaryLabel := model.Labels[primaryIdx]
	primaryConf := float64(res.Primary.Confidence)

	var secondaryLabel string
	var secondaryConf float64
	if res.Total >= 2 {
		secIdx := int(res.Secondary.Index)
		if secIdx >= 0 && secIdx < len(model.Labels) {
			secondaryLabel = model.Labels[secIdx]
			secondaryConf = float64(res.Secondary.Confidence)
		}
	}
	entropy := float64(res.Entropy)

	// 1. Fallback Isolation
	if primaryConf < r.policy.LowThreshold || unkRatio >= 0.5 || entropy > r.policy.MaxEntropy {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, false, false, true)
		return r.fallback(ctx, payload)
	}

	// 2. Multi-Intent Pipeline Dispatch
	if secondaryLabel != "" && secondaryConf >= r.policy.PipelineThreshold {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, false, true, false)
		key := pipelineKey(primaryLabel, secondaryLabel)
		if pipeAction, exists := r.pipelines[key]; exists {
			return pipeAction(ctx, primaryLabel, secondaryLabel, payload)
		}
		if r.defaultPipeline != nil {
			return r.defaultPipeline(ctx, primaryLabel, secondaryLabel, payload)
		}
	}

	// 3. Fallback to standard 3-tier routing if no pipeline applies
	margin := primaryConf - secondaryConf
	isAmbiguous := primaryConf < r.policy.HighThreshold || margin < r.policy.MarginCutoff
	if isAmbiguous {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, true, false, false)
		if r.ambiguous != nil {
			return r.ambiguous(ctx, primaryLabel, secondaryLabel, payload)
		}
		return r.fallback(ctx, payload)
	}

	action, exists := r.routes[primaryLabel]
	if !exists {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, false, false, true)
		return r.fallback(ctx, payload)
	}

	return action(ctx, payload)
}

// DispatchWithTrace runs inference, collects full diagnostic telemetry, and dispatches to handler.
func (r *Router) DispatchWithTrace(ctx context.Context, text string, payload any) (RouteTrace, error) {
	trace := r.Inspect(text)

	r.mu.RLock()
	defer r.mu.RUnlock()

	r.recordTelemetry(text, trace.PredictedLabel, trace.SecondaryLabel, trace.Confidence, trace.Entropy, trace.IsAmbiguous, trace.IsPipeline, trace.IsFallback)

	if trace.IsFallback {
		return trace, r.fallback(ctx, payload)
	}
	if trace.IsPipeline {
		key := pipelineKey(trace.PredictedLabel, trace.SecondaryLabel)
		if pipeAction, exists := r.pipelines[key]; exists {
			return trace, pipeAction(ctx, trace.PredictedLabel, trace.SecondaryLabel, payload)
		}
		if r.defaultPipeline != nil {
			return trace, r.defaultPipeline(ctx, trace.PredictedLabel, trace.SecondaryLabel, payload)
		}
	}
	if trace.IsAmbiguous && r.ambiguous != nil {
		return trace, r.ambiguous(ctx, trace.PredictedLabel, trace.SecondaryLabel, payload)
	}

	action, exists := r.routes[trace.PredictedLabel]
	if !exists {
		return trace, r.fallback(ctx, payload)
	}
	return trace, action(ctx, payload)
}
