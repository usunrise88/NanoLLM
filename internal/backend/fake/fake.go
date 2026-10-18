// Package fake is a scripted backend for developing and testing everything
// above the engine without one (SPEC §12, ROADMAP «Фейковый бэкенд»).
//
// It tokenizes by bytes with the Gemma 4 control tokens, and answers by a
// scenario the prompt selects (see scenario.go), so the server, the scheduler,
// the dialects and the UI can be exercised end to end under -race on a machine
// with no model and no native library. It keeps the real backend's contract:
// batched Steps across slots, several tokens per step when MTP drafting is
// enabled, a per-slot KV cache that trims and is saved and restored, and a
// batch budget that is enforced rather than assumed.
//
// It is registered as "fake" and runs only when backends.enabled names it.
package fake

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
)

// Name is the registered backend name.
const Name = "fake"

func init() {
	backend.Register(Name, func(string) (backend.Backend, error) { return New(Options{}), nil })
}

// Options shape the fake engine. Zero values are sensible defaults.
type Options struct {
	// PrefillRate and DecodeRate are tokens per second a Step is slowed to;
	// zero means no delay. They let tests and the UI see streaming happen.
	PrefillRate float64
	DecodeRate  float64
	// MaxDraft is the drafter's limit when a model is loaded with one;
	// Acceptance is the fraction of draft tokens accepted, deterministically.
	MaxDraft   int
	Acceptance float64
	// Memory is what the CPU device and the fake GPU report, in bytes.
	Memory int64
	// NoGPU hides the fake GPU device.
	NoGPU bool
}

// Backend is the fake engine.
type Backend struct {
	opts Options
}

// New returns a fake backend with opts.
func New(opts Options) *Backend {
	if opts.Memory == 0 {
		opts.Memory = 16 << 30
	}
	if opts.MaxDraft == 0 {
		opts.MaxDraft = 3
	}
	if opts.Acceptance == 0 {
		opts.Acceptance = 0.7
	}
	return &Backend{opts: opts}
}

func (b *Backend) Name() string    { return Name }
func (b *Backend) Version() string { return "fake" }
func (b *Backend) Close() error    { return nil }

func (b *Backend) Devices(context.Context) ([]backend.Device, error) {
	devs := []backend.Device{{
		Backend: Name, ID: "cpu", Type: backend.CPU, Name: "Fake CPU",
		MemoryTotal: b.opts.Memory, MemoryFree: b.opts.Memory,
	}}
	if !b.opts.NoGPU {
		devs = append(devs, backend.Device{
			Backend: Name, ID: "gpu0", Type: backend.GPU, Name: "Fake GPU",
			MemoryTotal: b.opts.Memory / 2, MemoryFree: b.opts.Memory / 2,
		})
	}
	return devs, nil
}

func (b *Backend) Load(ctx context.Context, spec backend.LoadSpec) (backend.Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	nctx, slots := spec.NCtx, spec.Slots
	if nctx <= 0 {
		nctx = 4096
	}
	if slots <= 0 {
		slots = 1
	}
	nbatch := spec.NBatch
	if nbatch <= 0 {
		nbatch = 512
	}
	m := &Model{
		opts: b.opts,
		caps: backend.Capabilities{
			NCtx: nctx, Slots: slots, NBatch: nbatch,
			SWAFull: spec.SWAFull, SWAWindow: 512, NVocab: NVocab,
		},
		path:  spec.Path,
		slots: make([]slot, slots),
	}
	if spec.DrafterPath != "" {
		m.caps.MaxDraft = b.opts.MaxDraft
		if spec.DraftMax > 0 && spec.DraftMax < m.caps.MaxDraft {
			m.caps.MaxDraft = spec.DraftMax
		}
	}
	for i := range m.slots {
		m.slots[i].genStart = -1
	}
	return m, nil
}

// slot is one sequence: what is in its cache, and the script it is replaying.
type slot struct {
	kv []backend.Token
	// genStart is the cache position where the current answer began, or -1
	// before the first sample; out is the answer being replayed from there.
	genStart int
	out      []backend.Token
	endless  bool
	fail     bool
	// next is the token the last Step returned last and the caller has to
	// feed back; matching it tells a continued answer from a new prompt.
	next    backend.Token
	hasNext bool

	sampler backend.SamplerSpec
	// drafted and accepted count this answer's draft tokens, for
	// deterministic acceptance.
	drafted, accepted int
}

// Model is a loaded fake model.
type Model struct {
	opts Options
	caps backend.Capabilities
	path string

	mu     sync.Mutex
	slots  []slot
	closed bool
}

func (m *Model) Tokenize(text string, addSpecial, parseSpecial bool) ([]backend.Token, error) {
	return tokenize(text, addSpecial, parseSpecial), nil
}

func (m *Model) Detokenize(tokens []backend.Token) (string, error) { return detokenize(tokens) }
func (m *Model) TokenPiece(t backend.Token) (string, error)        { return piece(t) }
func (m *Model) IsEOG(t backend.Token) bool                        { return t == TokEOS || t == TokEndTurn }

func (m *Model) Metadata() map[string]string {
	return map[string]string{
		"general.architecture":  "gemma4",
		"general.name":          "fake",
		"gemma4.context_length": fmt.Sprint(m.caps.NCtx),
		"fake.path":             m.path,
	}
}

func (m *Model) Capabilities() backend.Capabilities { return m.caps }

func (m *Model) MemoryReport() backend.MemoryReport {
	kv := int64(m.caps.NCtx*m.caps.Slots) * 1024
	return backend.MemoryReport{Devices: []backend.DeviceMemory{{
		Device: "cpu", Weights: 64 << 20, KV: kv, Compute: 16 << 20,
	}}}
}

func (m *Model) slot(seq int) (*slot, error) {
	if seq < 0 || seq >= len(m.slots) {
		return nil, fmt.Errorf("fake: slot %d out of range (%d slots)", seq, len(m.slots))
	}
	return &m.slots[seq], nil
}

func (m *Model) SetSampler(seq int, spec backend.SamplerSpec, history []backend.Token) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, err := m.slot(seq)
	if err != nil {
		return err
	}
	s.sampler = spec
	return nil
}

func (m *Model) Step(ctx context.Context, steps []backend.SeqStep) ([]backend.SeqOut, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, fmt.Errorf("fake: model is closed")
	}

	// Validate the whole batch before touching any slot, as the real backend
	// must: a bad batch reaching the engine aborts the process.
	budget, prefill, decode := 0, 0, 0
	seen := map[int]bool{}
	for _, st := range steps {
		s, err := m.slot(st.Seq)
		if err != nil {
			return nil, err
		}
		if seen[st.Seq] {
			return nil, fmt.Errorf("fake: slot %d twice in one step", st.Seq)
		}
		seen[st.Seq] = true
		if st.Pos != len(s.kv) {
			return nil, fmt.Errorf("fake: slot %d: step at position %d, cache holds %d", st.Seq, st.Pos, len(s.kv))
		}
		if len(st.Tokens) == 0 && !st.Sample {
			return nil, fmt.Errorf("fake: slot %d: empty step", st.Seq)
		}
		if st.Pos+len(st.Tokens)+m.draftFor(st)+1 > m.caps.NCtx {
			return nil, fmt.Errorf("fake: slot %d: context of %d tokens exceeded", st.Seq, m.caps.NCtx)
		}
		for _, t := range st.Tokens {
			if t < 0 || int(t) >= NVocab {
				return nil, fmt.Errorf("fake: slot %d: token %d out of vocabulary", st.Seq, t)
			}
		}
		budget += len(st.Tokens) + m.draftFor(st)
		if len(st.Tokens) > 1 {
			prefill += len(st.Tokens)
		} else {
			decode += len(st.Tokens) + m.draftFor(st)
		}
	}
	if budget > m.caps.NBatch {
		return nil, fmt.Errorf("fake: step of %d tokens exceeds n_batch %d", budget, m.caps.NBatch)
	}
	if err := m.delay(ctx, prefill, decode); err != nil {
		return nil, err
	}

	var outs []backend.SeqOut
	for _, st := range steps {
		s := &m.slots[st.Seq]
		continued := s.hasNext && len(st.Tokens) == 1 && st.Tokens[0] == s.next && s.genStart >= 0
		s.kv = append(s.kv, st.Tokens...)
		s.hasNext = false
		if !st.Sample {
			if !continued {
				s.genStart = -1
			}
			continue
		}
		if !continued {
			prompt, _ := detokenize(s.kv)
			sc := resume(prompt)
			s.genStart, s.out, s.endless, s.fail = len(s.kv), sc.tokens(), sc.endless, sc.fail
			s.drafted, s.accepted = 0, 0
		}
		if s.fail {
			return nil, errScenario
		}
		out := m.emit(s, st)
		outs = append(outs, out)
	}
	return outs, nil
}

// draftFor is how many draft tokens a step's verification costs.
func (m *Model) draftFor(st backend.SeqStep) int {
	if !st.Sample {
		return 0
	}
	return min(st.MaxDraft, m.caps.MaxDraft)
}

// emit produces one sampling slot's tokens: the next one of its script, plus
// accepted drafts. Accepted tokens but the last go into the cache, as they
// would after verification.
func (m *Model) emit(s *slot, st backend.SeqStep) backend.SeqOut {
	out := backend.SeqOut{Seq: st.Seq}
	nDraft := m.draftFor(st)
	accepted := 0
	for i := 0; i < nDraft; i++ {
		s.drafted++
		// Accept while the slot's running acceptance stays at or under the
		// target: deterministic, and the long-run rate is Acceptance.
		if float64(s.accepted+1) > m.opts.Acceptance*float64(s.drafted) {
			break
		}
		s.accepted++
		accepted++
	}
	out.Drafted, out.Accepted = nDraft, accepted

	for i := 0; i < accepted+1; i++ {
		t := s.at(len(s.kv) - s.genStart + i)
		out.Tokens = append(out.Tokens, t)
		if st.Logprobs >= 0 {
			out.Logprobs = append(out.Logprobs, logprobs(t, st.Logprobs))
		}
		if m.IsEOG(t) {
			out.EOG = true
			break
		}
	}
	// Everything but the last token is in the cache now.
	s.kv = append(s.kv, out.Tokens[:len(out.Tokens)-1]...)
	if !out.EOG {
		s.next, s.hasNext = out.Tokens[len(out.Tokens)-1], true
	}
	return out
}

// at is the script token at generation index i. An endless script repeats; a
// finished one keeps returning end-of-turn, which a caller that ignores EOG
// (a benchmark forcing a length) gets as a steady stream.
func (s *slot) at(i int) backend.Token {
	if len(s.out) == 0 {
		return TokEndTurn
	}
	if s.endless {
		return s.out[i%len(s.out)]
	}
	if i < len(s.out) {
		return s.out[i]
	}
	return TokEndTurn
}

// logprobs makes up a distribution in which the chosen token is the most
// likely, so clients see well-formed numbers.
func logprobs(t backend.Token, top int) backend.TokenLogprobs {
	lp := backend.TokenLogprobs{Token: t, Logprob: math.Log(0.9)}
	for i := 0; i < top; i++ {
		alt, p := t, 0.9
		if i > 0 {
			alt, p = backend.Token((int(t)+i)%NVocab), 0.1/float64(top)
		}
		lp.Top = append(lp.Top, backend.TokenLogprob{Token: alt, Logprob: math.Log(p)})
	}
	return lp
}

func (m *Model) delay(ctx context.Context, prefill, decode int) error {
	var d time.Duration
	if m.opts.PrefillRate > 0 {
		d += time.Duration(float64(prefill) / m.opts.PrefillRate * float64(time.Second))
	}
	if m.opts.DecodeRate > 0 && decode > 0 {
		// A decode step costs one forward pass whatever the batch width,
		// which is the point of batching.
		d += time.Duration(float64(time.Second) / m.opts.DecodeRate)
	}
	if d == 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Model) TrimSeq(seq int, pos int) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, err := m.slot(seq)
	if err != nil || pos < 0 {
		return false
	}
	if pos >= len(s.kv) {
		return true
	}
	if !m.caps.SWAFull && len(s.kv)-pos > m.caps.SWAWindow {
		return false
	}
	s.kv = s.kv[:pos]
	s.hasNext = false
	if s.genStart > pos {
		s.genStart = -1
	}
	return true
}

func (m *Model) ClearSeq(seq int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, err := m.slot(seq); err == nil {
		*s = slot{genStart: -1}
	}
}

// savedSlot is a slot's state: its cache, and enough of the answer in
// progress to continue it after a restore, as a real state round-trip does.
type savedSlot struct {
	KV       []backend.Token `json:"kv"`
	GenStart int             `json:"gen_start"`
	Next     *backend.Token  `json:"next,omitempty"`
}

func (m *Model) SaveSeq(seq int) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, err := m.slot(seq)
	if err != nil {
		return nil, err
	}
	saved := savedSlot{KV: s.kv, GenStart: s.genStart}
	if s.hasNext {
		next := s.next
		saved.Next = &next
	}
	return json.Marshal(saved)
}

func (m *Model) RestoreSeq(seq int, state []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, err := m.slot(seq)
	if err != nil {
		return err
	}
	var saved savedSlot
	if err := json.Unmarshal(state, &saved); err != nil {
		return fmt.Errorf("fake: restore slot %d: %w", seq, err)
	}
	if saved.GenStart > len(saved.KV) {
		return fmt.Errorf("fake: restore slot %d: answer starts past the cache", seq)
	}
	*s = slot{kv: saved.KV, genStart: saved.GenStart}
	if saved.GenStart >= 0 {
		prompt, err := detokenize(saved.KV[:saved.GenStart])
		if err != nil {
			return err
		}
		sc := scenario(prompt)
		s.out, s.endless, s.fail = sc.tokens(), sc.endless, sc.fail
	}
	if saved.Next != nil {
		s.next, s.hasNext = *saved.Next, true
	}
	return nil
}

func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}
