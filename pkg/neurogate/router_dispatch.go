package neurogate

import (
	"context"
	"fmt"
	"math"
	"time"
	"unicode/utf8"
)

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

	// Guard 0: Pre-inference pattern detection (< 1 μs)
	if r.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
		return RouteDecision{}, ErrUnlearnedPattern
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
	uniqueRatio := CalculateUniqueTokenRatio(tokenIDs)

	if len(tokenIDs) >= 4 && r.policy.MinUniqueTokenRatio > 0.0 && uniqueRatio < r.policy.MinUniqueTokenRatio {
		return RouteDecision{
			SingleCharRatio:   singleRatio,
			UnknownTokenRatio: unkRatio,
			UniqueTokenRatio:  uniqueRatio,
		}, ErrDegeneratedInput
	}
	if len(tokenIDs) >= 2 && r.policy.MaxSingleCharRatio > 0.0 && singleRatio >= r.policy.MaxSingleCharRatio {
		return RouteDecision{
			SingleCharRatio:   singleRatio,
			UnknownTokenRatio: unkRatio,
			UniqueTokenRatio:  uniqueRatio,
		}, ErrUnlearnedVocabulary
	}
	if r.policy.MaxUnknownTokenRatio > 0.0 && unkRatio >= r.policy.MaxUnknownTokenRatio {
		return RouteDecision{
			SingleCharRatio:   singleRatio,
			UnknownTokenRatio: unkRatio,
			UniqueTokenRatio:  uniqueRatio,
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
		LogitMargin:         res.LogitMargin,
		CoActiveCount:       res.CoActiveCount,
		SingleCharRatio:     singleRatio,
		UnknownTokenRatio:   unkRatio,
		UniqueTokenRatio:    uniqueRatio,
		SecondaryIntent:     secondaryLabel,
		SecondaryConfidence: secondaryConf,
	}

	effectiveMinEnergy := r.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model != nil && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}
	effectiveMargin := r.policy.RawLogitMargin
	if effectiveMargin == 0 && model != nil && model.Header.CalibratedMargin > 0 {
		effectiveMargin = model.Header.CalibratedMargin
	}

	// 1. Energy boundary (LogSumExp)
	if effectiveMinEnergy != 0.0 && energy < effectiveMinEnergy {
		return decision, ErrOutOfDomain
	}

	// 2. High entropy boundary
	if entropy > r.policy.MaxEntropy {
		return decision, ErrHighEntropy
	}

	// 3. Ambiguity margin boundary (both probability margin, raw logit gap, and co-activation conflict)
	isAmbiguous := primaryConf < r.policy.HighThreshold || margin < r.policy.MarginCutoff || (effectiveMargin > 0.0 && res.LogitMargin < effectiveMargin) || (res.CoActiveCount >= 2 && effectiveMargin > 0.0 && res.LogitMargin < effectiveMargin*1.5)
	if isAmbiguous {
		return decision, ErrAmbiguousIntent
	}

	// 4. Low confidence boundary
	if primaryConf < r.policy.LowThreshold {
		return decision, ErrLowConfidence
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

	// Guard 0: Pre-inference pattern validation
	if r.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
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

	// Layer 1 Fallback Guard: check repetitive token flooding or unlearned vocabulary
	if len(tokenIDs) >= 4 && r.policy.MinUniqueTokenRatio > 0.0 && CalculateUniqueTokenRatio(tokenIDs) < r.policy.MinUniqueTokenRatio {
		r.recordTelemetry(text, "", "", 0, 0, false, false, true)
		return r.fallback(ctx, payload)
	}
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

	effectiveMinEnergy := r.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}
	effectiveMargin := r.policy.RawLogitMargin
	if effectiveMargin == 0 && model.Header.CalibratedMargin > 0 {
		effectiveMargin = model.Header.CalibratedMargin
	}

	// Layer 2 Fallback Guard: Low energy (OOD), excessive UNKs, high entropy (OOD)
	if (effectiveMinEnergy != 0.0 && energy < effectiveMinEnergy) || unkRatio >= 0.5 || entropy > r.policy.MaxEntropy {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, false, false, true)
		return r.fallback(ctx, payload)
	}

	// 2. Ambiguous Route: Borderline confidence, competitive margin gap, narrow logit margin, or co-activation conflict
	isAmbiguous := primaryConf < r.policy.HighThreshold || margin < r.policy.MarginCutoff || (effectiveMargin > 0.0 && res.LogitMargin < effectiveMargin) || (res.CoActiveCount >= 2 && effectiveMargin > 0.0 && res.LogitMargin < effectiveMargin*1.5)
	if isAmbiguous {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, true, false, false)
		if r.ambiguous != nil {
			return r.ambiguous(ctx, primaryLabel, secondaryLabel, payload)
		}
		return r.fallback(ctx, payload)
	}

	// 2.1 Low confidence boundary for non-ambiguous samples
	if primaryConf < r.policy.LowThreshold {
		r.recordTelemetry(text, primaryLabel, secondaryLabel, primaryConf, entropy, false, false, true)
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

	// Guard 0: Pre-inference pattern check
	if r.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
		return RouteTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "unlearned pattern detected",
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
	uniqueRatio := CalculateUniqueTokenRatio(tokens)

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

	slotRes, _ := model.PredictSlots(tokens, model.Temperature)
	energy := float64(slotRes.Energy)

	effectiveMinEnergy := r.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}
	effectiveMargin := r.policy.RawLogitMargin
	if effectiveMargin == 0 && model.Header.CalibratedMargin > 0 {
		effectiveMargin = model.Header.CalibratedMargin
	}

	trace := RouteTrace{
		InputText:          text,
		TokenIDs:           tokens,
		Subwords:           subwords,
		SingleCharRatio:    singleRatio,
		UnknownTokenRatio:  unkRatio,
		UniqueTokenRatio:   uniqueRatio,
		ClassProbabilities: probMap,
		PredictedLabel:     bestLabel,
		SecondaryLabel:     secondLabel,
		Confidence:         calibratedConfidence,
		Margin:             margin,
		LogitMargin:        slotRes.LogitMargin,
		CoActiveCount:      slotRes.CoActiveCount,
		Entropy:            entropy,
		Energy:             energy,
		Threshold:          r.policy.HighThreshold,
		LatencyMicros:      time.Since(start).Microseconds(),
	}

	if err != nil {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("inference error: %v", err)
	} else if len(tokens) >= 4 && r.policy.MinUniqueTokenRatio > 0.0 && uniqueRatio < r.policy.MinUniqueTokenRatio {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("repetitive pattern detected (unique token ratio %.2f < %.2f)", uniqueRatio, r.policy.MinUniqueTokenRatio)
	} else if len(tokens) >= 2 && r.policy.MaxSingleCharRatio > 0.0 && singleRatio >= r.policy.MaxSingleCharRatio {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("unlearned vocabulary (single-char ratio %.2f >= %.2f)", singleRatio, r.policy.MaxSingleCharRatio)
	} else if unkRatio >= 0.5 {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("excessive unknown tokens (%.2f >= 0.50)", unkRatio)
	} else if effectiveMinEnergy != 0.0 && energy < effectiveMinEnergy {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("energy %.4f below minimum threshold %.4f (OOD)", energy, effectiveMinEnergy)
	} else if entropy > r.policy.MaxEntropy {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("prediction entropy %.4f exceeds limit %.4f (OOD)", entropy, r.policy.MaxEntropy)
	} else if trace.Confidence < r.policy.LowThreshold {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("confidence %.4f below low threshold %.4f", trace.Confidence, r.policy.LowThreshold)
	} else {
		if secondLabel != "" && calibratedSecond >= r.policy.PipelineThreshold {
			trace.IsPipeline = true
		}
		if trace.Confidence < r.policy.HighThreshold || margin < r.policy.MarginCutoff || (effectiveMargin > 0.0 && trace.LogitMargin < effectiveMargin) || (trace.CoActiveCount >= 2 && effectiveMargin > 0.0 && trace.LogitMargin < effectiveMargin*1.5) {
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

	// Guard 0: Pre-inference pattern validation
	if r.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
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
	isAmbiguous := primaryConf < r.policy.HighThreshold || margin < r.policy.MarginCutoff || (r.policy.RawLogitMargin > 0.0 && res.LogitMargin < r.policy.RawLogitMargin) || (res.CoActiveCount >= 2 && r.policy.RawLogitMargin > 0.0 && res.LogitMargin < r.policy.RawLogitMargin*1.5)
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
