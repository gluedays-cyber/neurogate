package neurogate

import (
	"context"
	"math"
	"math/bits"
	"strings"
)

func pipelineKey(primary, secondary string) string {
	return primary + "->" + secondary
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

	effectiveMinEnergy := g.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model != nil && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}

	if (effectiveMinEnergy > 0 && logSumExp < effectiveMinEnergy) ||
		unkRatio >= 0.5 ||
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
		isDualActivated := coActive >= 2 && adjustedLogits[secondIdx] >= DefaultActivationThreshold
		if calibratedSecond >= g.policy.PipelineThreshold || (hasBothAnchors && hasPipeline) || (isDualActivated && hasPipeline) {
			eval.isPipeline = true
		}
	}

	effectiveMargin := g.policy.RawLogitMargin
	if effectiveMargin == 0 && model != nil && model.Header.CalibratedMargin > 0 {
		effectiveMargin = model.Header.CalibratedMargin
	}

	if calibratedConfidence < g.policy.HighThreshold || margin < g.policy.MarginCutoff || (effectiveMargin > 0.0 && logitMargin < effectiveMargin) || (coActive >= 2 && effectiveMargin > 0.0 && logitMargin < effectiveMargin*1.5) {
		eval.isAmbiguous = true
	} else if calibratedConfidence < g.policy.LowThreshold {
		eval.isFallback = true
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

	effectiveMinEnergy := g.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model != nil && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}

	if (effectiveMinEnergy > 0 && logSumExp < effectiveMinEnergy) ||
		unkRatio >= 0.5 ||
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
		isDualActivated := coActive >= 2 && adjustedLogits[secondIdx] >= DefaultActivationThreshold
		if calibratedSecond >= g.policy.PipelineThreshold || (hasBothAnchors && hasPipeline) || (isDualActivated && hasPipeline) {
			eval.isPipeline = true
		}
	}

	effectiveMargin := g.policy.RawLogitMargin
	if effectiveMargin == 0 && model != nil && model.Header.CalibratedMargin > 0 {
		effectiveMargin = model.Header.CalibratedMargin
	}

	if calibratedConfidence < g.policy.HighThreshold || margin < g.policy.MarginCutoff || (effectiveMargin > 0.0 && logitMargin < effectiveMargin) || (coActive >= 2 && effectiveMargin > 0.0 && logitMargin < effectiveMargin*1.5) {
		eval.isAmbiguous = true
	} else if calibratedConfidence < g.policy.LowThreshold {
		eval.isFallback = true
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
