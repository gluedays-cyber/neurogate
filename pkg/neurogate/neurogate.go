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

	// DefaultMaxAnchorBoost defines the default maximum cumulative logit boost per class (prevents logit explosion).
	DefaultMaxAnchorBoost float32 = 3.0
)

var (
	ErrNeuroGateModelNil = errors.New("neurogate: underlying model is nil")
	ErrClassLimitExceeded = errors.New("neurogate: registered classes exceed MaxGateClasses")
)

// AnchorRule defines a symbolic soft-bias injected into a specific class logit upon bitmask match.
type AnchorRule struct {
	ClassIndex     int
	Mask           uint64
	Weight         float32
	InhibitClasses []int
	Penalty        float32
	Keywords       []string
}

// GateTrace captures comprehensive runtime metrics across all three geometric heads.
type GateTrace struct {
	InputText          string             `json:"input_text"`
	TokenIDs           []uint32           `json:"token_ids"`
	Subwords           []string           `json:"subwords"`
	UnknownTokenRatio  float64            `json:"unknown_token_ratio"`
	CosineSimilarity   float32            `json:"cosine_similarity"`
	IsOOD              bool               `json:"is_ood"`
	AnchorBitmask      uint64             `json:"anchor_bitmask"`
	TriggeredAnchors   []string           `json:"triggered_anchors"`
	ClassProbabilities map[string]float32 `json:"class_probabilities"`
	PredictedLabel     string             `json:"predicted_label"`
	SecondaryLabel     string             `json:"secondary_label,omitempty"`
	Confidence         float64            `json:"confidence"`
	Margin             float64            `json:"margin"`
	LogitMargin        float32            `json:"logit_margin"`
	Entropy            float64            `json:"entropy"`
	LogSumExp          float64            `json:"log_sum_exp"`
	FreeEnergy         float64            `json:"free_energy"`
	Threshold          float64            `json:"threshold"`
	IsAmbiguous        bool               `json:"is_ambiguous"`
	IsPipeline         bool               `json:"is_pipeline"`
	IsFallback         bool               `json:"is_fallback"`
	FallbackReason     string             `json:"fallback_reason,omitempty"`
	LatencyMicros      int64              `json:"latency_micros"`
}

// GateRouteBuilder provides fluent API chaining for binding routes and anchor soft biases.
type GateRouteBuilder struct {
	gate       *NeuroGate
	classIndex int
	label      string
}

// WithAnchor registers anchor keywords that inject a soft additive bias into this class's logit.
func (b *GateRouteBuilder) WithAnchor(weight float32, keywords ...string) *GateRouteBuilder {
	b.gate.mu.Lock()
	defer b.gate.mu.Unlock()

	var mask uint64 = 0
	var cleanKeywords []string
	for _, kw := range keywords {
		kw = strings.TrimSpace(strings.ToLower(kw))
		if kw == "" {
			continue
		}
		cleanKeywords = append(cleanKeywords, kw)
		m, exists := b.gate.anchorDict[kw]
		if !exists {
			if len(b.gate.anchorDict) < 64 {
				m = 1 << uint64(len(b.gate.anchorDict))
				b.gate.anchorDict[kw] = m
			}
		}
		mask |= m
	}

	model := b.gate.model.Load()
	if model != nil && model.Tokenizer != nil {
		for _, kw := range cleanKeywords {
			toks := model.Tokenizer.Encode(kw)
			for _, tid := range toks {
				b.gate.anchorTokenMap[tid] |= mask
			}
		}
	}

	b.gate.anchorRules = append(b.gate.anchorRules, AnchorRule{
		ClassIndex: b.classIndex,
		Mask:       mask,
		Weight:     weight,
		Keywords:   cleanKeywords,
	})
	return b
}

// Inhibit registers competing class labels to penalize when this anchor triggers.
func (b *GateRouteBuilder) Inhibit(penalty float32, competingLabels ...string) *GateRouteBuilder {
	b.gate.mu.Lock()
	defer b.gate.mu.Unlock()

	if len(b.gate.anchorRules) == 0 {
		return b
	}
	lastIdx := len(b.gate.anchorRules) - 1
	rule := &b.gate.anchorRules[lastIdx]

	for _, lbl := range competingLabels {
		if cIdx, exists := b.gate.labelToIndex[lbl]; exists {
			rule.InhibitClasses = append(rule.InhibitClasses, cIdx)
		}
	}
	rule.Penalty = penalty
	return b
}

// Bind allows continuing chaining for additional routes.
func (b *GateRouteBuilder) Bind(label string, handler RouteAction) *GateRouteBuilder {
	return b.gate.Bind(label, handler)
}

// NeuroGate coordinates a single shared neural backbone with a geometric 3-head zero-allocation gate.
type NeuroGate struct {
	model atomic.Pointer[InferenceModel]

	mu             sync.RWMutex
	labels         [MaxGateClasses]string
	routes         [MaxGateClasses]RouteAction
	classCount     int
	labelToIndex   map[string]int

	anchorDict     map[string]uint64
	anchorTokenMap map[uint32]uint64
	anchorRules    []AnchorRule
	maxAnchorBoost float32

	domainCentroid [MaxGateEmbDim]float32
	hasCentroid    bool
	minCosineSim   float32
	domainMeanSim  float32
	domainStdDev   float32

	policy         DispatchPolicy
	pipelines      map[string]PipelineAction
	ambiguous      AmbiguousAction
	fallback       RouteAction
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

	// Auto-compute baseline domain centroid from valid vocabulary embeddings
	gate.computeBaselineCentroid(model)

	// Default fallback
	gate.fallback = func(ctx context.Context, payload any) error {
		return nil
	}

	return gate
}

// computeBaselineCentroid derives an initial normalized centroid from the model's vocabulary embeddings.
func (g *NeuroGate) computeBaselineCentroid(model *InferenceModel) {
	embDim := int(model.Header.EmbeddingDim)
	if embDim > MaxGateEmbDim || embDim == 0 || len(model.Weights.Embedding) == 0 {
		return
	}

	var sum [MaxGateEmbDim]float64
	validTokens := 0
	for tokID, word := range model.Vocab {
		if word == "[PAD]" || word == "[UNK]" {
			continue
		}
		offset := tokID * embDim
		if offset+embDim <= len(model.Weights.Embedding) {
			for d := 0; d < embDim; d++ {
				sum[d] += float64(model.Weights.Embedding[offset+d])
			}
			validTokens++
		}
	}

	if validTokens > 0 {
		inv := 1.0 / float64(validTokens)
		var raw [MaxGateEmbDim]float32
		for d := 0; d < embDim; d++ {
			raw[d] = float32(sum[d] * inv)
		}
		L2Normalize(raw[:embDim], g.domainCentroid[:embDim])
		g.hasCentroid = true
	}
}

// CalibrateDomainCentroid calculates the true manifold center from sample dataset sentences.
func (g *NeuroGate) CalibrateDomainCentroid(samples []DataSample) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()

	model := g.model.Load()
	if model == nil || len(samples) == 0 {
		return g
	}

	embDim := int(model.Header.EmbeddingDim)
	if embDim > MaxGateEmbDim {
		embDim = MaxGateEmbDim
	}

	var sum [MaxGateEmbDim]float64
	validCount := 0
	var pooled [MaxGateEmbDim]float32

	var dummyLogits [MaxGateClasses]float32
	for _, s := range samples {
		tokens := model.Tokenizer.Encode(s.Text)
		if len(tokens) == 0 {
			continue
		}
		if err := model.PredictFeatures(tokens, pooled[:embDim], dummyLogits[:g.classCount]); err == nil {
			for d := 0; d < embDim; d++ {
				sum[d] += float64(pooled[d])
			}
			validCount++
		}
	}

	if validCount > 0 {
		inv := 1.0 / float64(validCount)
		var raw [MaxGateEmbDim]float32
		for d := 0; d < embDim; d++ {
			raw[d] = float32(sum[d] * inv)
		}
		L2Normalize(raw[:embDim], g.domainCentroid[:embDim])
		g.hasCentroid = true
	}
	return g
}

// CalibrateDomainDistribution calculates the manifold center and dynamically computes
// the standard deviation of cosine similarities across sample embeddings to configure an adaptive OOD threshold:
// minCosine = mean - (k * stdDev).
func (g *NeuroGate) CalibrateDomainDistribution(samples []DataSample, k float32) *NeuroGate {
	g.mu.Lock()
	defer g.mu.Unlock()

	model := g.model.Load()
	if model == nil || len(samples) == 0 {
		return g
	}

	embDim := int(model.Header.EmbeddingDim)
	if embDim > MaxGateEmbDim {
		embDim = MaxGateEmbDim
	}

	var sum [MaxGateEmbDim]float64
	validCount := 0
	var pooled [MaxGateEmbDim]float32
	var dummyLogits [MaxGateClasses]float32

	for _, s := range samples {
		tokens := model.Tokenizer.Encode(s.Text)
		if len(tokens) == 0 {
			continue
		}
		if err := model.PredictFeatures(tokens, pooled[:embDim], dummyLogits[:g.classCount]); err == nil {
			for d := 0; d < embDim; d++ {
				sum[d] += float64(pooled[d])
			}
			validCount++
		}
	}

	if validCount == 0 {
		return g
	}

	inv := 1.0 / float64(validCount)
	var raw [MaxGateEmbDim]float32
	for d := 0; d < embDim; d++ {
		raw[d] = float32(sum[d] * inv)
	}
	L2Normalize(raw[:embDim], g.domainCentroid[:embDim])
	g.hasCentroid = true

	// Pass 2: Calculate distribution variance and std dev of cosine similarities
	var sumSim, sumSqSim float64
	evalCount := 0
	var normPooled [MaxGateEmbDim]float32
	for _, s := range samples {
		tokens := model.Tokenizer.Encode(s.Text)
		if len(tokens) == 0 {
			continue
		}
		if err := model.PredictFeatures(tokens, pooled[:embDim], dummyLogits[:g.classCount]); err == nil {
			L2Normalize(pooled[:embDim], normPooled[:embDim])
			sim := DotProduct(normPooled[:embDim], g.domainCentroid[:embDim])
			sumSim += float64(sim)
			sumSqSim += float64(sim * sim)
			evalCount++
		}
	}

	if evalCount > 0 {
		n := float64(evalCount)
		mean := float32(sumSim / n)
		variance := float32((sumSqSim / n) - float64(mean*mean))
		if variance < 0 {
			variance = 0
		}
		stdDev := float32(math.Sqrt(float64(variance)))
		g.domainMeanSim = mean
		g.domainStdDev = stdDev

		adaptiveMin := mean - (k * stdDev)
		if adaptiveMin < -1.0 {
			adaptiveMin = -1.0
		}
		g.minCosineSim = adaptiveMin
	}

	return g
}

// DomainStats returns the calibrated manifold distribution metrics.
func (g *NeuroGate) DomainStats() (hasCentroid bool, meanSim float32, stdDev float32, minCosine float32) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.hasCentroid, g.domainMeanSim, g.domainStdDev, g.minCosineSim
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

	// Compute raw top-1 and top-2 logit margin
	var maxLogit1, maxLogit2 float32 = -math.MaxFloat32, -math.MaxFloat32
	for i := 0; i < numClasses; i++ {
		l := adjustedLogits[i]
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

	if g.hasCentroid && cosineSim < g.minCosineSim {
		isOOD = true
		oodReason = fmt.Sprintf("cosine similarity %.4f below domain threshold %.4f (OOD)", cosineSim, g.minCosineSim)
	} else if len(tokens) >= 2 && g.policy.MaxSingleCharRatio > 0 && singleRatio >= g.policy.MaxSingleCharRatio {
		isOOD = true
		oodReason = fmt.Sprintf("unlearned vocabulary (single-char ratio %.2f >= %.2f)", singleRatio, g.policy.MaxSingleCharRatio)
	} else if g.policy.MinLogSumExp > 0 && logSumExp < g.policy.MinLogSumExp {
		isOOD = true
		oodReason = fmt.Sprintf("free energy %.4f (logSumExp %.4f) below in-distribution threshold %.4f (OOD)", freeEnergy, logSumExp, g.policy.MinLogSumExp)
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
	} else if trace.Confidence < g.policy.LowThreshold {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("confidence %.4f below low threshold %.4f", trace.Confidence, g.policy.LowThreshold)
	} else {
		// Multi-intent pipeline evaluation (Softmax probability OR dual-anchor co-activation)
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
			if calibratedSecond >= g.policy.PipelineThreshold || (hasBothAnchors && hasPipeline) {
				trace.IsPipeline = true
			}
		}
		if trace.Confidence < g.policy.HighThreshold || margin < g.policy.MarginCutoff || (g.policy.RawLogitMargin > 0.0 && logitMargin < g.policy.RawLogitMargin) {
			trace.IsAmbiguous = true
		}
		if g.routes[bestIdx] == nil {
			trace.IsFallback = true
			trace.FallbackReason = fmt.Sprintf("label '%s' has no bound route handler", bestLabel)
		}
	}

	return trace
}

type gateEvaluation struct {
	primaryIdx   int
	secondaryIdx int
	isFallback   bool
	isPipeline   bool
	isAmbiguous  bool
}

// evaluateFast executes the 3-head gate on the stack without allocating diagnostic maps or traces.
func (g *NeuroGate) evaluateFast(text string) gateEvaluation {
	if g.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
		return gateEvaluation{isFallback: true}
	}

	model := g.model.Load()
	if model == nil {
		return gateEvaluation{isFallback: true}
	}

	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	tokens := model.Tokenizer.Encode(text)
	if len(tokens) == 0 {
		return gateEvaluation{isFallback: true}
	}
	if len(tokens) > MaxSequenceTokens {
		tokens = tokens[:MaxSequenceTokens]
	}

	singleRatio, _ := model.Tokenizer.AnalyzeUnlearnedRatio(tokens)
	if len(tokens) >= 2 && g.policy.MaxSingleCharRatio > 0 && singleRatio >= g.policy.MaxSingleCharRatio {
		return gateEvaluation{isFallback: true}
	}

	unkCount := 0
	unkID, hasUnk := model.Tokenizer.VocabMap["[UNK]"]
	for _, id := range tokens {
		if hasUnk && id == unkID {
			unkCount++
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

	var pooledStack [MaxGateEmbDim]float32
	var rawLogitsStack [MaxGateClasses]float32
	var normPooled [MaxGateEmbDim]float32

	_ = model.PredictFeatures(tokens, pooledStack[:embDim], rawLogitsStack[:numClasses])

	// [Head 1]: L2 Cosine OOD Guard
	L2Normalize(pooledStack[:embDim], normPooled[:embDim])
	if g.hasCentroid {
		cosineSim := DotProduct(normPooled[:embDim], g.domainCentroid[:embDim])
		if cosineSim < g.minCosineSim {
			return gateEvaluation{isFallback: true}
		}
	}

	// [Head 2]: Anchor Bitmask & Symbolic Bias
	var textBitmask uint64 = 0
	lowerText := strings.ToLower(text)
	for kw, mask := range g.anchorDict {
		if strings.Contains(lowerText, kw) {
			textBitmask |= mask
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

	// Logit margin computation
	var maxLogit1, maxLogit2 float32 = -math.MaxFloat32, -math.MaxFloat32
	for i := 0; i < numClasses; i++ {
		l := adjustedLogits[i]
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

	// [Head 3]: Softmax, Margin, and Entropy
	var probs [MaxGateClasses]float32
	_ = Softmax(adjustedLogits[:numClasses], model.Temperature, probs[:numClasses])

	var bestIdx, secondIdx int = 0, 1
	var bestScore, secondScore float32 = -1.0, -1.0
	for i := 0; i < numClasses; i++ {
		p := probs[i]
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

	calibratedConfidence := float64(bestScore) * (1.0 - unkRatio)
	calibratedSecond := float64(secondScore) * (1.0 - unkRatio)
	margin := calibratedConfidence - calibratedSecond
	entropy := float64(computeEntropy(probs[:numClasses]))

	// LogSumExp / Free Energy evaluation
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

	if (g.policy.MinLogSumExp > 0 && logSumExp < g.policy.MinLogSumExp) ||
		unkRatio >= 0.5 ||
		calibratedConfidence < g.policy.LowThreshold ||
		entropy > g.policy.MaxEntropy {
		return gateEvaluation{isFallback: true, primaryIdx: bestIdx, secondaryIdx: secondIdx}
	}

	eval := gateEvaluation{
		primaryIdx:   bestIdx,
		secondaryIdx: secondIdx,
	}

	if secondIdx >= 0 {
		pipeKey := pipelineKey(g.labels[bestIdx], g.labels[secondIdx])
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
		if calibratedSecond >= g.policy.PipelineThreshold || (hasBothAnchors && hasPipeline) {
			eval.isPipeline = true
		}
	}
	if calibratedConfidence < g.policy.HighThreshold || margin < g.policy.MarginCutoff || (g.policy.RawLogitMargin > 0.0 && logitMargin < g.policy.RawLogitMargin) {
		eval.isAmbiguous = true
	}
	if g.routes[bestIdx] == nil {
		eval.isFallback = true
	}

	return eval
}

// Filter evaluates the query and executes the appropriate handler with zero allocations.
func (g *NeuroGate) Filter(ctx context.Context, text string, payload any) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	eval := g.evaluateFast(text)

	g.mu.RLock()
	defer g.mu.RUnlock()

	// 1. Fallback Tier
	if eval.isFallback {
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	// 2. Ambiguous Tier
	if eval.isAmbiguous {
		if g.ambiguous != nil {
			pLabel := g.labels[eval.primaryIdx]
			sLabel := ""
			if eval.secondaryIdx >= 0 && eval.secondaryIdx < g.classCount {
				sLabel = g.labels[eval.secondaryIdx]
			}
			return g.ambiguous(ctx, pLabel, sLabel, payload)
		}
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	// 3. Definite Route Tier
	if eval.primaryIdx < 0 || eval.primaryIdx >= g.classCount || g.routes[eval.primaryIdx] == nil {
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	return g.routes[eval.primaryIdx](ctx, payload)
}

// FilterPipeline evaluates the query supporting both definite, pipeline, and fallback executions.
func (g *NeuroGate) FilterPipeline(ctx context.Context, text string, payload any) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	eval := g.evaluateFast(text)

	g.mu.RLock()
	defer g.mu.RUnlock()

	// 1. Fallback Tier
	if eval.isFallback {
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	pLabel := g.labels[eval.primaryIdx]
	sLabel := ""
	if eval.secondaryIdx >= 0 && eval.secondaryIdx < g.classCount {
		sLabel = g.labels[eval.secondaryIdx]
	}

	// 2. Multi-Intent Pipeline Tier
	if eval.isPipeline && sLabel != "" {
		pipeKey := pipelineKey(pLabel, sLabel)
		if pipeAction, exists := g.pipelines[pipeKey]; exists && pipeAction != nil {
			return pipeAction(ctx, pLabel, sLabel, payload)
		}
	}

	// 3. Ambiguous Tier
	if eval.isAmbiguous {
		if g.ambiguous != nil {
			return g.ambiguous(ctx, pLabel, sLabel, payload)
		}
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	// 4. Definite Route Tier
	if eval.primaryIdx < 0 || eval.primaryIdx >= g.classCount || g.routes[eval.primaryIdx] == nil {
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	return g.routes[eval.primaryIdx](ctx, payload)
}

// evaluateFastTokens executes the 3-head gate directly on token IDs with strictly zero heap allocations.
func (g *NeuroGate) evaluateFastTokens(tokens []uint32) gateEvaluation {
	model := g.model.Load()
	if model == nil || len(tokens) == 0 {
		return gateEvaluation{isFallback: true}
	}

	if len(tokens) > MaxSequenceTokens {
		tokens = tokens[:MaxSequenceTokens]
	}

	unkCount := 0
	unkID, hasUnk := model.Tokenizer.VocabMap["[UNK]"]
	for _, id := range tokens {
		if hasUnk && id == unkID {
			unkCount++
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

	var pooledStack [MaxGateEmbDim]float32
	var rawLogitsStack [MaxGateClasses]float32
	var normPooled [MaxGateEmbDim]float32

	_ = model.PredictFeatures(tokens, pooledStack[:embDim], rawLogitsStack[:numClasses])

	// [Head 1]: L2 Cosine OOD Guard
	L2Normalize(pooledStack[:embDim], normPooled[:embDim])
	if g.hasCentroid {
		cosineSim := DotProduct(normPooled[:embDim], g.domainCentroid[:embDim])
		if cosineSim < g.minCosineSim {
			return gateEvaluation{isFallback: true}
		}
	}

	// [Head 2]: 1-Cycle Bitwise Token Anchor Soft-Bias
	var textBitmask uint64 = 0
	for _, id := range tokens {
		if m, exists := g.anchorTokenMap[id]; exists {
			textBitmask |= m
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

	// Logit margin computation
	var maxLogit1, maxLogit2 float32 = -math.MaxFloat32, -math.MaxFloat32
	for i := 0; i < numClasses; i++ {
		l := adjustedLogits[i]
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

	// [Head 3]: Stack Softmax, Margin, and Entropy
	var probs [MaxGateClasses]float32
	_ = Softmax(adjustedLogits[:numClasses], model.Temperature, probs[:numClasses])

	var bestIdx, secondIdx int = 0, 1
	var bestScore, secondScore float32 = -1.0, -1.0
	for i := 0; i < numClasses; i++ {
		p := probs[i]
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

	calibratedConfidence := float64(bestScore) * (1.0 - unkRatio)
	calibratedSecond := float64(secondScore) * (1.0 - unkRatio)
	margin := calibratedConfidence - calibratedSecond
	entropy := float64(computeEntropy(probs[:numClasses]))

	// LogSumExp evaluation
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

	if (g.policy.MinLogSumExp > 0 && logSumExp < g.policy.MinLogSumExp) ||
		unkRatio >= 0.5 ||
		calibratedConfidence < g.policy.LowThreshold ||
		entropy > g.policy.MaxEntropy {
		return gateEvaluation{isFallback: true, primaryIdx: bestIdx, secondaryIdx: secondIdx}
	}

	eval := gateEvaluation{
		primaryIdx:   bestIdx,
		secondaryIdx: secondIdx,
	}

	if secondIdx >= 0 {
		pipeKey := pipelineKey(g.labels[bestIdx], g.labels[secondIdx])
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
		if calibratedSecond >= g.policy.PipelineThreshold || (hasBothAnchors && hasPipeline) {
			eval.isPipeline = true
		}
	}
	if calibratedConfidence < g.policy.HighThreshold || margin < g.policy.MarginCutoff || (g.policy.RawLogitMargin > 0.0 && logitMargin < g.policy.RawLogitMargin) {
		eval.isAmbiguous = true
	}
	if g.routes[bestIdx] == nil {
		eval.isFallback = true
	}

	return eval
}

// FilterTokens evaluates pre-tokenized inputs with strictly 0 B/op heap allocation.
func (g *NeuroGate) FilterTokens(ctx context.Context, tokens []uint32, payload any) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	eval := g.evaluateFastTokens(tokens)

	g.mu.RLock()
	defer g.mu.RUnlock()

	if eval.isFallback {
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	if eval.isAmbiguous {
		if g.ambiguous != nil {
			pLabel := g.labels[eval.primaryIdx]
			sLabel := ""
			if eval.secondaryIdx >= 0 && eval.secondaryIdx < g.classCount {
				sLabel = g.labels[eval.secondaryIdx]
			}
			return g.ambiguous(ctx, pLabel, sLabel, payload)
		}
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	if eval.primaryIdx < 0 || eval.primaryIdx >= g.classCount || g.routes[eval.primaryIdx] == nil {
		if g.fallback != nil {
			return g.fallback(ctx, payload)
		}
		return nil
	}

	return g.routes[eval.primaryIdx](ctx, payload)
}

