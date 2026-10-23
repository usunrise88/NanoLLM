package llamacpp

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"strings"
	"unsafe"

	"github.com/usunrise88/nanollm/internal/backend"
)

// penaltyLastN is how many recent tokens the repetition penalties look at,
// llama.cpp's default (common_params_sampling.penalty_last_n).
const penaltyLastN = 64

// defaultSeed asks llama_sampler_init_dist for a random seed.
const defaultSeed = 0xFFFFFFFF

// SetSampler builds the slot's chain in the order llama.cpp's own tools use:
// logit bias, penalties, grammar, then the truncations, temperature and the
// final pick. One chain per slot means llama_sampler_sample applies and
// accepts in one call, so the grammar and the penalties see exactly the
// tokens returned. The history goes to the penalties sampler alone: the
// grammar constrains only what is generated from here on.
func (m *model) SetSampler(seq int, spec backend.SamplerSpec, history []backend.Token) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return errClosed
	}
	if err := m.checkSeq(seq); err != nil {
		return err
	}
	if err := m.checkTokens(history); err != nil {
		return err
	}
	if err := checkSamplerSpec(spec); err != nil {
		return err
	}
	for t := range spec.LogitBias {
		if t < 0 || int(t) >= m.nVocab {
			return fmt.Errorf("llamacpp: logit_bias token %d is outside the vocabulary", t)
		}
	}
	// A greedy chain on the device holds no state: the next request with the
	// same settings keeps it. Attaching a new one makes llama.cpp reserve the
	// graph again on the next decode (16 ms more to the first token on CUDA).
	if sl := &m.slots[seq]; sl.onDevice && statelessChain(spec) && reflect.DeepEqual(sl.spec, spec) {
		return nil
	}

	chain := llamaSamplerChainInit(llamaSamplerChainDefaultParams())
	if chain == 0 {
		return errors.New("llamacpp: sampler chain allocation failed")
	}
	add := func(s LlamaSampler) error {
		if s == 0 {
			return errors.New("llamacpp: sampler allocation failed")
		}
		llamaSamplerChainAdd(chain, s)
		return nil
	}
	fail := func(err error) error {
		llamaSamplerFree(chain)
		return err
	}

	// A model may list tokens it must never produce
	// (tokenizer.ggml.suppress_tokens); llama.cpp's tools fold them into the
	// logit bias as -inf, and so does this chain.
	bias := make([]LlamaLogitBias, 0, len(spec.LogitBias))
	for t, b := range spec.LogitBias {
		bias = append(bias, LlamaLogitBias{Token: int32(t), Bias: float32(b)})
	}
	var nSuppress int32
	if p := llamaVocabGetSuppressTokens(m.vocab, &nSuppress); p != nil && nSuppress > 0 {
		for _, t := range unsafe.Slice(p, nSuppress) {
			bias = append(bias, LlamaLogitBias{Token: t, Bias: float32(math.Inf(-1))})
		}
	}
	if len(bias) > 0 {
		if err := add(llamaSamplerInitLogitBias(int32(m.nVocab), int32(len(bias)), &bias[0])); err != nil {
			return fail(err)
		}
	}
	var penalties LlamaSampler
	repeat := spec.RepeatPenalty
	if repeat == 0 {
		repeat = 1
	}
	if repeat != 1 || spec.FrequencyPenalty != 0 || spec.PresencePenalty != 0 {
		penalties = llamaSamplerInitPenalties(int32(m.nVocab), penaltyLastN,
			float32(repeat), float32(spec.FrequencyPenalty), float32(spec.PresencePenalty))
		if err := add(penalties); err != nil {
			return fail(err)
		}
	}
	if spec.Grammar != "" {
		g, err := m.grammarSampler(spec)
		if err != nil {
			return fail(err)
		}
		llamaSamplerChainAdd(chain, g)
	}
	if spec.Temperature <= 0 {
		if err := add(llamaSamplerInitGreedy()); err != nil {
			return fail(err)
		}
	} else {
		if spec.TopK > 0 {
			if err := add(llamaSamplerInitTopK(int32(spec.TopK))); err != nil {
				return fail(err)
			}
		}
		if spec.TopP > 0 && spec.TopP < 1 {
			if err := add(llamaSamplerInitTopP(float32(spec.TopP), 1)); err != nil {
				return fail(err)
			}
		}
		if spec.MinP > 0 {
			if err := add(llamaSamplerInitMinP(float32(spec.MinP), 1)); err != nil {
				return fail(err)
			}
		}
		if err := add(llamaSamplerInitTemp(float32(spec.Temperature))); err != nil {
			return fail(err)
		}
		seed := uint32(defaultSeed)
		if spec.Seed >= 0 {
			seed = uint32(spec.Seed)
		}
		if err := add(llamaSamplerInitDist(seed)); err != nil {
			return fail(err)
		}
	}
	if penalties != 0 {
		start := max(0, len(history)-penaltyLastN)
		for _, t := range history[start:] {
			llamaSamplerAccept(penalties, int32(t))
		}
	}
	m.freeSampler(seq)
	sl := &m.slots[seq]
	sl.chain, sl.penalties, sl.spec = chain, penalties, spec
	sl.wholeVocab = wholeVocabulary(spec)
	m.noteHistory(sl, history)
	// The whole chain goes into the graph or none of it: an attached chain
	// skips its device part on the host, so it cannot fall back later. A
	// grammar does not run on a device and penalties need the history on it;
	// logprobs need the logits rows, which an attached chain keeps on the GPU.
	if m.spec.BackendSampling && spec.Grammar == "" && penalties == 0 && !spec.RawLogits {
		sl.onDevice = llamaSetSampler(m.ctx, int32(seq), chain)
	}
	return nil
}

// statelessChain is a greedy chain without penalties or grammar: what it
// picks depends on the logits alone.
func statelessChain(s backend.SamplerSpec) bool {
	return s.Temperature <= 0 && s.Grammar == "" &&
		(s.RepeatPenalty == 0 || s.RepeatPenalty == 1) && s.FrequencyPenalty == 0 && s.PresencePenalty == 0
}

func checkSamplerSpec(s backend.SamplerSpec) error {
	for name, v := range map[string]float64{
		"temperature": s.Temperature, "top_p": s.TopP, "min_p": s.MinP,
		"repeat_penalty": s.RepeatPenalty, "presence_penalty": s.PresencePenalty, "frequency_penalty": s.FrequencyPenalty,
	} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("llamacpp: %s is not finite", name)
		}
	}
	for t, b := range s.LogitBias {
		if math.IsNaN(b) {
			return fmt.Errorf("llamacpp: logit_bias for token %d is NaN", t)
		}
	}
	if s.RepeatPenalty < 0 {
		return errors.New("llamacpp: repeat_penalty must be positive")
	}
	if s.TopK < 0 || s.TopP < 0 || s.TopP > 1 || s.MinP < 0 || s.MinP > 1 {
		return errors.New("llamacpp: top_k, top_p or min_p out of range")
	}
	if strings.ContainsRune(s.Grammar, 0) {
		return errors.New("llamacpp: grammar contains a NUL byte")
	}
	if s.GrammarRoot != "" && !isGrammarIdent(s.GrammarRoot) {
		return fmt.Errorf("llamacpp: grammar root %q is not a rule name", s.GrammarRoot)
	}
	if s.Grammar == "" && (s.GrammarRoot != "" || len(s.GrammarTriggers) > 0) {
		return errors.New("llamacpp: grammar root or triggers without a grammar")
	}
	for _, t := range s.GrammarTriggers {
		if t == "" || strings.ContainsRune(t, 0) {
			return errors.New("llamacpp: empty grammar trigger")
		}
	}
	return nil
}

// isGrammarIdent matches a GBNF rule name: letters, digits and dashes.
func isGrammarIdent(s string) bool {
	for _, r := range s {
		ok := r == '-' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9'
		if !ok {
			return false
		}
	}
	return s != ""
}

// grammarSampler parses the grammar. A grammar llama.cpp cannot parse comes
// back as NULL, with the reason in its log, which goes into the error. The
// lazy variant gets trigger words as escaped regular expressions, the way
// llama.cpp's common library builds them: the patterns are compiled with
// std::regex, whose constructor would throw on a stray bracket.
func (m *model) grammarSampler(spec backend.SamplerSpec) (LlamaSampler, error) {
	root := spec.GrammarRoot
	if root == "" {
		root = "root"
	}
	var why []string
	untap := addLogTap(func(level GgmlLogLevel, line string) {
		if level == GgmlLogLevelError || strings.Contains(line, "grammar") {
			why = append(why, line)
		}
	})
	defer untap()
	var g LlamaSampler
	if len(spec.GrammarTriggers) == 0 {
		g = llamaSamplerInitGrammar(m.vocab, spec.Grammar, root)
	} else {
		pats := make([][]byte, len(spec.GrammarTriggers))
		ptrs := make([]*byte, len(spec.GrammarTriggers))
		for i, t := range spec.GrammarTriggers {
			pats[i] = append([]byte(regexEscape(t)), 0)
			ptrs[i] = &pats[i][0]
		}
		g = llamaSamplerInitGrammarLazyPatterns(m.vocab, spec.Grammar, root, &ptrs[0], uintptr(len(ptrs)), nil, 0)
		runtime.KeepAlive(pats)
		runtime.KeepAlive(ptrs)
	}
	if g == 0 {
		msg := "llamacpp: the grammar does not parse"
		if len(why) > 0 {
			msg += ": " + strings.Join(why, "; ")
		}
		return 0, errors.New(msg)
	}
	return g, nil
}

// regexEscape escapes the ECMAScript metacharacters, as common's
// regex_escape does.
func regexEscape(s string) string {
	var b strings.Builder
	for _, r := range s {
		if strings.ContainsRune(`.^$|()*+?[]{}\`, r) {
			b.WriteByte('\\')
		}
		b.WriteRune(r)
	}
	return b.String()
}

func (m *model) freeSampler(seq int) {
	sl := &m.slots[seq]
	if sl.onDevice {
		// detached first: the context would sample through a freed chain
		llamaSetSampler(m.ctx, int32(seq), 0)
	}
	if sl.chain != 0 {
		llamaSamplerFree(sl.chain)
	}
	sl.chain, sl.penalties, sl.spec, sl.onDevice = 0, 0, backend.SamplerSpec{}, false
}

func (m *model) checkSeq(seq int) error {
	if seq < 0 || seq >= len(m.slots) {
		return fmt.Errorf("llamacpp: slot %d out of range (0..%d)", seq, len(m.slots)-1)
	}
	return nil
}
