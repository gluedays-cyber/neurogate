package neurogate

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// MaxGateClasses defines the maximum supported classes for zero-allocation stack buffers.
	MaxGateClasses = 16

	// MaxGateEmbDim defines the maximum supported embedding dimension for zero-allocation stack buffers.
	MaxGateEmbDim = 64
)

var (
	ErrNeuroGateModelNil  = errors.New("neurogate: underlying model is nil")
	ErrClassLimitExceeded = errors.New("neurogate: registered classes exceed MaxGateClasses")
)

// NeuroGate coordinates a single shared neural backbone with a geometric 3-head zero-allocation gate.
type NeuroGate struct {
	model atomic.Pointer[InferenceModel]

	mu           sync.RWMutex
	labels       [MaxGateClasses]string
	routes       [MaxGateClasses]RouteAction
	classCount   int
	labelToIndex map[string]int

	anchorDict     map[string]uint64
	anchorTokenMap map[uint32]uint64
	anchorRules    []AnchorRule
	maxAnchorBoost float32

	domainCentroid [MaxGateEmbDim]float32
	hasCentroid    bool
	minCosineSim   float32
	domainMeanSim  float32
	domainStdDev   float32

	policy    DispatchPolicy
	pipelines map[string]PipelineAction
	ambiguous AmbiguousAction
	fallback  RouteAction
}

// NewNeuroGate loads an InferenceModel from disk and prepares a high-performance NeuroGate.
func NewNeuroGate(modelPath string) (*NeuroGate, error) {
	model, err := LoadBinaryModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("neurogate: failed to load model %s: %w", modelPath, err)
	}
	return NewNeuroGateWithModel(model), nil
}

// NewNeuroGateWithModel wraps an existing InferenceModel into a NeuroGate instance.
func NewNeuroGateWithModel(model *InferenceModel) *NeuroGate {
	gate := &NeuroGate{
		labelToIndex:   make(map[string]int),
		anchorDict:     make(map[string]uint64),
		anchorTokenMap: make(map[uint32]uint64),
		pipelines:      make(map[string]PipelineAction),
		policy:         DefaultDispatchPolicy(),
		minCosineSim:   0.25,
		maxAnchorBoost: DefaultMaxAnchorBoost,
	}
	gate.policy.MaxSingleCharRatio = 0.85
	gate.model.Store(model)

	// Pre-register model labels up to MaxGateClasses
	for i, lbl := range model.Labels {
		if i >= MaxGateClasses {
			break
		}
		gate.labels[i] = lbl
		gate.labelToIndex[lbl] = i
		gate.classCount++
	}

	// Apply self-calibrated thresholds from binary header if present
	if model.Header.CalibratedMinEnergy > 0 {
		gate.policy.MinLogSumExp = float64(model.Header.CalibratedMinEnergy)
	}
	if model.Header.CalibratedMargin > 0 {
		gate.policy.RawLogitMargin = model.Header.CalibratedMargin
	}
	if model.Header.CalibratedMinCosine != 0 {
		gate.minCosineSim = model.Header.CalibratedMinCosine
	}

	// Auto-compute baseline domain centroid from valid vocabulary embeddings
	gate.computeBaselineCentroid(model)

	// Default fallback
	gate.fallback = func(ctx context.Context, payload any) error {
		return nil
	}

	return gate
}

// SetMaxAnchorBoost sets the maximum cumulative soft-bias logit boost allowed per class (prevents logit explosion).
func (g *NeuroGate) SetMaxAnchorBoost(cap float32) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.maxAnchorBoost = cap
	return g
}

// MaxAnchorBoost returns the current cap for cumulative anchor logit boost.
func (g *NeuroGate) MaxAnchorBoost() float32 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.maxAnchorBoost
}

// SetDomainBoundary configures the reference L2 centroid and minimum cosine similarity for OOD rejection.
func (g *NeuroGate) SetDomainBoundary(centroid []float32, minCosine float32) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()

	dim := len(centroid)
	if dim > MaxGateEmbDim {
		dim = MaxGateEmbDim
	}
	copy(g.domainCentroid[:dim], centroid[:dim])
	L2Normalize(g.domainCentroid[:dim], g.domainCentroid[:dim])
	g.hasCentroid = true
	g.minCosineSim = minCosine
	return g
}

// SetMinCosineSim configures the cosine boundary threshold for the auto-computed centroid.
func (g *NeuroGate) SetMinCosineSim(threshold float32) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.minCosineSim = threshold
	if threshold <= 0 {
		g.hasCentroid = false
	} else {
		g.hasCentroid = true
	}
	return g
}

// SetPolicy updates the dispatch thresholds, margin gap, and entropy cutoff.
func (g *NeuroGate) SetPolicy(policy DispatchPolicy) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.policy = policy
	return g
}

// SetSingleCharRatioCutoff configures the Layer 1 unlearned single-character token ratio threshold.
func (g *NeuroGate) SetSingleCharRatioCutoff(cutoff float64) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.policy.MaxSingleCharRatio = cutoff
	return g
}

// SetTemperature configures the temperature scaling factor used in softmax calculations.
func (g *NeuroGate) SetTemperature(t float32) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	if m := g.model.Load(); m != nil {
		m.Temperature = t
	}
	return g
}

// Temperature returns the current temperature scaling factor.
func (g *NeuroGate) Temperature() float32 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if m := g.model.Load(); m != nil {
		return m.Temperature
	}
	return 0
}

// Bind registers an action handler for a target class label, returning a GateRouteBuilder for anchor chaining.
func (g *NeuroGate) Bind(label string, handler RouteAction) *GateRouteBuilder {
	g.mu.Lock()
	defer g.mu.Unlock()

	idx, exists := g.labelToIndex[label]
	if !exists {
		if g.classCount >= MaxGateClasses {
			panic(ErrClassLimitExceeded)
		}
		idx = g.classCount
		g.labels[idx] = label
		g.labelToIndex[label] = idx
		g.classCount++
	}
	g.routes[idx] = handler

	return &GateRouteBuilder{
		gate:       g,
		classIndex: idx,
		label:      label,
	}
}

// BindPipeline registers a multi-intent handler triggered when both primary and secondary meet thresholds.
func (g *NeuroGate) BindPipeline(primary, secondary string, handler PipelineAction) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.pipelines[pipelineKey(primary, secondary)] = handler
	return g
}

// Ambiguous registers the handler for borderline confidence or competitive margin gaps.
func (g *NeuroGate) Ambiguous(handler AmbiguousAction) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.ambiguous = handler
	return g
}

// Fallback registers the default safety action for OOD or uncertain inputs.
func (g *NeuroGate) Fallback(handler RouteAction) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.fallback = handler
	return g
}

// SwapModel atomically updates the underlying inference model without downtime.
func (g *NeuroGate) SwapModel(newModel *InferenceModel) {
	g.model.Store(newModel)
}

// Model returns the current atomic inference model pointer.
func (g *NeuroGate) Model() *InferenceModel {
	return g.model.Load()
}

// Inspect evaluates input text across all 3 heads and returns detailed diagnostics without mutations.
func (g *NeuroGate) Inspect(text string) GateTrace {
	start := time.Now()
	model := g.model.Load()
	if model == nil {
		return GateTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "model not loaded",
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	if g.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
		return GateTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "unlearned pattern detected",
			Threshold:      g.policy.HighThreshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	tokens := model.Tokenizer.Encode(text)
	if len(tokens) == 0 {
		return GateTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "empty input tokens",
			Threshold:      g.policy.HighThreshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}
	if len(tokens) > MaxSequenceTokens {
		tokens = tokens[:MaxSequenceTokens]
	}

	singleRatio, _ := model.Tokenizer.AnalyzeUnlearnedRatio(tokens)
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
	unkRatio := float64(unkCount) / float64(len(tokens))

	g.mu.RLock()
	defer g.mu.RUnlock()

	embDim := int(model.Header.EmbeddingDim)
	if embDim > MaxGateEmbDim {
		embDim = MaxGateEmbDim
	}
	numClasses := g.classCount

	// Stack memory allocation for strictly 0 B/op feature extraction
	var pooledStack [MaxGateEmbDim]float32
	var rawLogitsStack [MaxGateClasses]float32
	var normPooled [MaxGateEmbDim]float32

	_ = model.PredictFeatures(tokens, pooledStack[:embDim], rawLogitsStack[:numClasses])

	// -------------------------------------------------------------
	// [Head 1]: Geometric L2 Cosine Out-of-Domain (OOD) Guard
	// -------------------------------------------------------------
	L2Normalize(pooledStack[:embDim], normPooled[:embDim])
	var cosineSim float32 = 1.0
	isOOD := false
	if g.hasCentroid {
		cosineSim = DotProduct(normPooled[:embDim], g.domainCentroid[:embDim])
		if cosineSim < g.minCosineSim {
			isOOD = true
		}
	}

	// -------------------------------------------------------------
	// [Head 2]: Tokenizer Anchor Bitmask & Symbolic Soft-Bias
	// -------------------------------------------------------------
	var textBitmask uint64 = 0
	var triggeredAnchors []string
	lowerText := strings.ToLower(text)
	for kw, mask := range g.anchorDict {
		if strings.Contains(lowerText, kw) {
			textBitmask |= mask
			triggeredAnchors = append(triggeredAnchors, kw)
		}
	}

	var adjustedLogits [MaxGateClasses]float32
	copy(adjustedLogits[:numClasses], rawLogitsStack[:numClasses])
	var classDeltas [MaxGateClasses]float32
	for _, rule := range g.anchorRules {
		matchedBits := textBitmask & rule.Mask
		if matchedBits != 0 {
			count := float32(bits.OnesCount64(matchedBits))
			classDeltas[rule.ClassIndex] += rule.Weight * count
			for _, inhClass := range rule.InhibitClasses {
				if inhClass >= 0 && inhClass < numClasses {
					classDeltas[inhClass] -= rule.Penalty * count
				}
			}
		}
	}
	for c := 0; c < numClasses; c++ {
		delta := classDeltas[c]
		if delta > 0 && g.maxAnchorBoost > 0 && delta > g.maxAnchorBoost {
			delta = g.maxAnchorBoost
		}
		adjustedLogits[c] += delta
	}

	// -------------------------------------------------------------
	// [Head 3]: Stack Softmax, Margin, and Shannon Entropy Gating
	// -------------------------------------------------------------
	var probs [MaxGateClasses]float32
	_ = Softmax(adjustedLogits[:numClasses], model.Temperature, probs[:numClasses])

	var bestIdx, secondIdx int = 0, 1
	var bestScore, secondScore float32 = -1.0, -1.0
	probMap := make(map[string]float32, numClasses)

	for i := 0; i < numClasses; i++ {
		lbl := g.labels[i]
		p := probs[i]
		probMap[lbl] = p
		if p > bestScore {
			secondScore = bestScore
			secondIdx = bestIdx
			bestScore = p
			bestIdx = i
		} else if p > secondScore {
			secondScore = p
			secondIdx = i
		}
	}

	// Calibrate confidence by unknown token ratio penalty
	calibratedConfidence := float64(bestScore) * (1.0 - unkRatio)
	calibratedSecond := float64(secondScore) * (1.0 - unkRatio)
	margin := calibratedConfidence - calibratedSecond
	entropy := float64(computeEntropy(probs[:numClasses]))

	bestLabel := g.labels[bestIdx]
	secondLabel := ""
	if numClasses >= 2 {
		secondLabel = g.labels[secondIdx]
	}

	// Compute LogSumExp and Free Energy across logits
	var maxLogit float32 = adjustedLogits[0]
	for i := 1; i < numClasses; i++ {
		if adjustedLogits[i] > maxLogit {
			maxLogit = adjustedLogits[i]
		}
	}
	var sumExp float64
	for i := 0; i < numClasses; i++ {
		sumExp += math.Exp(float64(adjustedLogits[i] - maxLogit))
	}
	logSumExp := float64(maxLogit) + math.Log(sumExp)
	freeEnergy := -logSumExp

	// Compute raw top-1 and top-2 logit margin and co-activation count
	var maxLogit1, maxLogit2 float32 = -math.MaxFloat32, -math.MaxFloat32
	var coActive uint8 = 0
	for i := 0; i < numClasses; i++ {
		l := adjustedLogits[i]
		if l >= DefaultActivationThreshold {
			coActive++
		}
		if l > maxLogit1 {
			maxLogit2 = maxLogit1
			maxLogit1 = l
		} else if l > maxLogit2 {
			maxLogit2 = l
		}
	}
	var logitMargin float32 = 0.0
	if numClasses >= 2 {
		logitMargin = maxLogit1 - maxLogit2
	}

	// -------------------------------------------------------------
	// [Head 1 & Head 3]: Unified OOD & Energy Guard
	// -------------------------------------------------------------
	var oodReason string

	effectiveMinEnergy := g.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model != nil && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}
	effectiveMinCosine := g.minCosineSim
	if effectiveMinCosine == 0 && model != nil && model.Header.CalibratedMinCosine != 0 {
		effectiveMinCosine = model.Header.CalibratedMinCosine
	}

	if g.hasCentroid && effectiveMinCosine > 0 && cosineSim < effectiveMinCosine {
		isOOD = true
		oodReason = fmt.Sprintf("cosine similarity %.4f below domain threshold %.4f (OOD)", cosineSim, effectiveMinCosine)
	} else if len(tokens) >= 2 && g.policy.MaxSingleCharRatio > 0 && singleRatio >= g.policy.MaxSingleCharRatio {
		isOOD = true
		oodReason = fmt.Sprintf("unlearned vocabulary (single-char ratio %.2f >= %.2f)", singleRatio, g.policy.MaxSingleCharRatio)
	} else if effectiveMinEnergy > 0 && logSumExp < effectiveMinEnergy {
		isOOD = true
		oodReason = fmt.Sprintf("free energy %.4f (logSumExp %.4f) below in-distribution threshold %.4f (OOD)", freeEnergy, logSumExp, effectiveMinEnergy)
	} else if entropy > g.policy.MaxEntropy {
		isOOD = true
		oodReason = fmt.Sprintf("prediction entropy %.4f exceeds limit %.4f (OOD)", entropy, g.policy.MaxEntropy)
	} else if unkRatio >= 0.5 {
		isOOD = true
		oodReason = fmt.Sprintf("excessive unknown tokens (%.2f >= 0.50)", unkRatio)
	}

	trace := GateTrace{
		InputText:          text,
		TokenIDs:           tokens,
		Subwords:           subwords,
		UnknownTokenRatio:  unkRatio,
		CosineSimilarity:   cosineSim,
		IsOOD:              isOOD,
		AnchorBitmask:      textBitmask,
		TriggeredAnchors:   triggeredAnchors,
		ClassProbabilities: probMap,
		PredictedLabel:     bestLabel,
		SecondaryLabel:     secondLabel,
		Confidence:         calibratedConfidence,
		Margin:             margin,
		LogitMargin:        logitMargin,
		CoActiveCount:      coActive,
		Entropy:            entropy,
		LogSumExp:          logSumExp,
		FreeEnergy:         freeEnergy,
		Threshold:          g.policy.HighThreshold,
		LatencyMicros:      time.Since(start).Microseconds(),
	}

	// Routing tier evaluation
	if isOOD {
		trace.IsOOD = true
		trace.IsFallback = true
		trace.FallbackReason = oodReason
	} else {
		// Multi-intent pipeline evaluation (Softmax probability, dual-anchor co-activation, or pre-softmax co-activation)
		if secondLabel != "" {
			pipeKey := pipelineKey(bestLabel, secondLabel)
			_, hasPipeline := g.pipelines[pipeKey]
			var primaryAnchors, secondaryAnchors bool
			for _, rule := range g.anchorRules {
				if (textBitmask & rule.Mask) != 0 {
					if rule.ClassIndex == bestIdx {
						primaryAnchors = true
					}
					if rule.ClassIndex == secondIdx {
						secondaryAnchors = true
					}
				}
			}
			hasBothAnchors := primaryAnchors && secondaryAnchors
			isDualActivated := coActive >= 2 && adjustedLogits[secondIdx] >= DefaultActivationThreshold
			if calibratedSecond >= g.policy.PipelineThreshold || (hasBothAnchors && hasPipeline) || (isDualActivated && hasPipeline) {
				trace.IsPipeline = true
			}
		}

		effectiveMargin := g.policy.RawLogitMargin
		if effectiveMargin == 0 && model != nil && model.Header.CalibratedMargin > 0 {
			effectiveMargin = model.Header.CalibratedMargin
		}

		isAmbiguous := trace.Confidence < g.policy.HighThreshold || margin < g.policy.MarginCutoff || (effectiveMargin > 0.0 && logitMargin < effectiveMargin) || (coActive >= 2 && effectiveMargin > 0.0 && logitMargin < effectiveMargin*1.5)
		if isAmbiguous {
			trace.IsAmbiguous = true
		} else if trace.Confidence < g.policy.LowThreshold {
			trace.IsFallback = true
			trace.FallbackReason = fmt.Sprintf("confidence %.4f below low threshold %.4f", trace.Confidence, g.policy.LowThreshold)
		}

		if g.routes[bestIdx] == nil {
			trace.IsFallback = true
			if trace.FallbackReason == "" {
				trace.FallbackReason = fmt.Sprintf("label '%s' has no bound route handler", bestLabel)
			}
		}
	}

	return trace
}
