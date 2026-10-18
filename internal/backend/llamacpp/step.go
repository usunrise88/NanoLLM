package llamacpp

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/usunrise88/nanollm/internal/backend"
)

// maxLogprobs is the most alternatives a position reports (OpenAI's limit).
const maxLogprobs = 20

// Step evaluates one batch: every slot's tokens go into one llama_batch_ext
// at the positions given, the batch is decoded with llama_process, and each
// sampling slot draws one token from the logits after its last token.
//
// Without a drafter (until 1.5.M) a sampling slot gets exactly one token per
// step. The MTP path will extend it here: the drafter proposes up to
// MaxDraft tokens from the carried hidden state, they ride in the same
// batch, the chain samples position by position and stops at the first
// mismatch, and the rejected tail is removed with llama_memory_seq_rm.
func (m *model) Step(ctx context.Context, steps []backend.SeqStep) ([]backend.SeqOut, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return nil, errClosed
	}
	if err := m.checkSteps(steps); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	llamaBatchExtClear(m.batch)
	outIdx := make([]int32, len(steps))
	for i, s := range steps {
		outIdx[i] = -1
		for j, t := range s.Tokens {
			idx := llamaBatchExtAddToken(m.batch, int32(s.Seq), int32(t))
			if idx < 0 {
				return nil, fmt.Errorf("llamacpp: batch entry for slot %d rejected (%d)", s.Seq, idx)
			}
			pos := int32(s.Pos + j)
			if !llamaBatchExtSetPos(m.batch, idx, &pos) {
				return nil, fmt.Errorf("llamacpp: position %d for slot %d rejected", pos, s.Seq)
			}
			if s.Sample && j == len(s.Tokens)-1 {
				llamaBatchExtSetOutputLogits(m.batch, idx, true)
				outIdx[i] = idx
			}
		}
	}
	if rc := llamaProcess(m.ctx, LlamaProcessTypeDecode, m.batch); rc != 0 {
		return nil, m.decodeError(rc, steps)
	}

	var outs []backend.SeqOut
	for i, s := range steps {
		if !s.Sample {
			continue
		}
		row, err := m.logitsRow(outIdx[i])
		if err != nil {
			return nil, err
		}
		var lp *logprobs
		if s.Logprobs >= 0 {
			lp = m.logprobs(row, s.Logprobs)
		}
		tok := backend.Token(llamaSamplerSample(m.slots[s.Seq].chain, m.ctx, outIdx[i]))
		out := backend.SeqOut{Seq: s.Seq, Tokens: []backend.Token{tok}, EOG: llamaVocabIsEog(m.vocab, int32(tok))}
		if lp != nil {
			lp.Token = tok
			lp.Logprob = lp.logprobOf(row, tok)
			out.Logprobs = []backend.TokenLogprobs{lp.TokenLogprobs}
		}
		outs = append(outs, out)
	}
	return outs, nil
}

// checkSteps validates a batch before any of it reaches llama.cpp, which
// asserts on an empty batch and on a sampled token without logits.
func (m *model) checkSteps(steps []backend.SeqStep) error {
	if len(steps) == 0 {
		return errors.New("llamacpp: empty step")
	}
	seen := map[int]bool{}
	total := 0
	for _, s := range steps {
		if err := m.checkSeq(s.Seq); err != nil {
			return err
		}
		if seen[s.Seq] {
			return fmt.Errorf("llamacpp: slot %d appears twice in one step", s.Seq)
		}
		seen[s.Seq] = true
		if len(s.Tokens) == 0 {
			return fmt.Errorf("llamacpp: slot %d has no tokens to evaluate", s.Seq)
		}
		if err := m.checkTokens(s.Tokens); err != nil {
			return err
		}
		next := int(llamaMemorySeqPosMax(m.mem, int32(s.Seq))) + 1
		if s.Pos != next {
			return fmt.Errorf("llamacpp: slot %d continues at position %d, not %d", s.Seq, next, s.Pos)
		}
		if s.Pos+len(s.Tokens) > m.caps.NCtx {
			return fmt.Errorf("llamacpp: slot %d would reach position %d, past its context of %d", s.Seq, s.Pos+len(s.Tokens), m.caps.NCtx)
		}
		if s.Sample && m.slots[s.Seq].chain == 0 {
			return fmt.Errorf("llamacpp: slot %d samples without a sampler; call SetSampler first", s.Seq)
		}
		if s.Logprobs < -1 || s.Logprobs > maxLogprobs {
			return fmt.Errorf("llamacpp: logprobs %d out of range (-1..%d)", s.Logprobs, maxLogprobs)
		}
		if s.MaxDraft < 0 {
			return fmt.Errorf("llamacpp: slot %d: negative MaxDraft", s.Seq)
		}
		total += len(s.Tokens)
	}
	if total > m.caps.NBatch {
		return fmt.Errorf("llamacpp: step of %d tokens exceeds n_batch %d", total, m.caps.NBatch)
	}
	return nil
}

// decodeError explains a failed llama_process. After an abort or a fatal
// error the ubatches already computed stay in the cache, so each slot is cut
// back to where this step started and the caller may retry it.
func (m *model) decodeError(rc int32, steps []backend.SeqStep) error {
	if rc == 2 || rc < -1 {
		for _, s := range steps {
			llamaMemorySeqRm(m.mem, int32(s.Seq), int32(s.Pos), -1)
		}
	}
	switch rc {
	case 1:
		return errors.New("llamacpp: no KV cache space for this step")
	case 2:
		return errors.New("llamacpp: decode aborted")
	case -1:
		return errors.New("llamacpp: decode rejected the batch")
	}
	return fmt.Errorf("llamacpp: decode failed (%d)", rc)
}

type logprobs struct {
	backend.TokenLogprobs
	logZ float64
}

func (l *logprobs) logprobOf(row []float32, t backend.Token) float64 {
	return float64(row[t]) - l.logZ
}

// logprobs is the log-softmax of the raw logits, before any sampler, with
// the top n alternatives (STAGE1 1.0: logprobs come from the logits, not
// from the sampler's truncated distribution).
func (m *model) logprobs(row []float32, n int) *logprobs {
	maxv := math.Inf(-1)
	for _, v := range row {
		if f := float64(v); f > maxv {
			maxv = f
		}
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v) - maxv)
	}
	lp := &logprobs{logZ: maxv + math.Log(sum)}
	if n > 0 {
		// Top n by a bounded insertion: n ≤ 20 against a vocabulary of
		// 262144, so a sort of the whole row would dominate.
		top := make([]backend.TokenLogprob, 0, n+1)
		for i, v := range row {
			f := float64(v)
			if len(top) == n && f <= top[n-1].Logprob {
				continue
			}
			j := sort.Search(len(top), func(k int) bool { return top[k].Logprob < f })
			top = append(top, backend.TokenLogprob{})
			copy(top[j+1:], top[j:])
			top[j] = backend.TokenLogprob{Token: backend.Token(i), Logprob: f}
			if len(top) > n {
				top = top[:n]
			}
		}
		for i := range top {
			top[i].Logprob -= lp.logZ
		}
		lp.Top = top
	}
	return lp
}

// TrimSeq removes positions pos.. of a slot. With sliding-window layers and
// swa_full off, the SWA cache only holds the last window, so a trim is
// possible only while the window before pos is still there — the condition
// llama-server checks before reusing a prompt. Recurrent state cannot be
// rolled back at all.
func (m *model) TrimSeq(seq int, pos int) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 || m.checkSeq(seq) != nil || pos < 0 {
		return false
	}
	pmax := int(llamaMemorySeqPosMax(m.mem, int32(seq)))
	if pos > pmax {
		return true
	}
	if pos == 0 {
		llamaMemorySeqRm(m.mem, int32(seq), -1, -1)
		return true
	}
	if m.recurrent {
		return false
	}
	if m.nSWA > 0 && !m.swaFull {
		pmin := int(llamaMemorySeqPosMin(m.mem, int32(seq)))
		if pmin > max(0, pos-m.nSWA) {
			return false
		}
	}
	return llamaMemorySeqRm(m.mem, int32(seq), int32(pos), -1)
}

// ClearSeq empties a slot's cache. Removing a whole sequence never fails.
func (m *model) ClearSeq(seq int) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 || m.checkSeq(seq) != nil {
		return
	}
	llamaMemorySeqRm(m.mem, int32(seq), -1, -1)
}

// Saved slot state: a header of ours, then llama.cpp's own sequence state.
// llama_state_seq_set_data trusts its input — counts and sizes in it are
// used as given — so a blob from another model, another build or another
// context shape is refused here, before it reaches C.
const (
	stateMagic   = "NLLS"
	stateVersion = 1
	stateHeader  = 4 + 4 + 16 + 8*6
)

func (m *model) stateFingerprint() [6]uint64 {
	return [6]uint64{
		uint64(m.nVocab), uint64(llamaModelNEmbd(m.mdl)), uint64(llamaModelNLayer(m.mdl)),
		llamaModelSize(m.mdl), uint64(m.caps.NCtx), uint64(m.caps.Slots),
	}
}

func (m *model) SaveSeq(seq int) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return nil, errClosed
	}
	if err := m.checkSeq(seq); err != nil {
		return nil, err
	}
	size := llamaStateSeqGetSizeExt(m.ctx, int32(seq), 0)
	buf := make([]byte, stateHeader+int(size))
	copy(buf, stateMagic)
	binary.LittleEndian.PutUint32(buf[4:], stateVersion)
	copy(buf[8:24], bindingCommit)
	for i, v := range m.stateFingerprint() {
		binary.LittleEndian.PutUint64(buf[24+8*i:], v)
	}
	if size == 0 {
		return buf, nil
	}
	n := llamaStateSeqGetDataExt(m.ctx, &buf[stateHeader], size, int32(seq), 0)
	if n == 0 {
		return nil, fmt.Errorf("llamacpp: saving slot %d failed", seq)
	}
	return buf[:stateHeader+int(n)], nil
}

func (m *model) RestoreSeq(seq int, state []byte) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return errClosed
	}
	if err := m.checkSeq(seq); err != nil {
		return err
	}
	if len(state) < stateHeader || string(state[:4]) != stateMagic {
		return errors.New("llamacpp: not a saved slot state")
	}
	if v := binary.LittleEndian.Uint32(state[4:]); v != stateVersion {
		return fmt.Errorf("llamacpp: slot state version %d, want %d", v, stateVersion)
	}
	if want := bindingCommit[:16]; string(state[8:24]) != want {
		return fmt.Errorf("llamacpp: slot state was saved by llama.cpp %q, this is %q", state[8:24], want)
	}
	for i, v := range m.stateFingerprint() {
		if got := binary.LittleEndian.Uint64(state[24+8*i:]); got != v {
			return errors.New("llamacpp: slot state belongs to another model or context shape")
		}
	}
	llamaMemorySeqRm(m.mem, int32(seq), -1, -1)
	if len(state) == stateHeader {
		return nil
	}
	data := state[stateHeader:]
	if n := llamaStateSeqSetDataExt(m.ctx, &data[0], uintptr(len(data)), int32(seq), 0); n == 0 {
		llamaMemorySeqRm(m.mem, int32(seq), -1, -1)
		return fmt.Errorf("llamacpp: restoring slot %d failed", seq)
	}
	return nil
}
