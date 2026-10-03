package neurogate

import (
	"strings"
)

const (
	// DefaultMaxAnchorBoost defines the default maximum cumulative logit boost per class (prevents logit explosion).
	DefaultMaxAnchorBoost float32 = 3.0
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
