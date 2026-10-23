package llamacpp

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"unsafe"

	"github.com/usunrise88/nanollm/internal/backend"
)

// maxLogprobs is the most alternatives a position reports (OpenAI's limit).
const maxLogprobs = 20

// Step evaluates one batch: every slot's tokens go into one llama_batch_ext
// at the positions given, the batch is decoded with llama_process, and each
// sampling slot draws its tokens from the logits.
//
// Without a drafter a sampling slot gets one token. With one, a decode step
// (one token, at the position after the slot's carried state) first drafts
// up to MaxDraft tokens (mtp.go); the drafts ride in the same batch, the
// slot's chain samples them position by position, and the slot gets the
// accepted drafts plus the token that ended the run.
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

	var drafts [][]backend.Token
	if m.dft != nil {
		if limits := m.draftLimits(steps); limits != nil {
			drafts = m.draft(steps, limits)
		}
	} else if m.ngN > 0 {
		if limits := m.draftLimits(steps); limits != nil {
			drafts = m.lookup(steps, limits)
		}
	}

	llamaBatchExtClear(m.batch)
	// outIdx[i] are the batch entries whose logits step i samples: the last
	// token, then one per draft.
	outIdx := make([][]int32, len(steps))
	add := func(seq, pos int, t backend.Token, output bool) (int32, error) {
		idx := llamaBatchExtAddToken(m.batch, int32(seq), int32(t))
		if idx < 0 {
			return 0, fmt.Errorf("llamacpp: batch entry for slot %d rejected (%d)", seq, idx)
		}
		p := int32(pos)
		if !llamaBatchExtSetPos(m.batch, idx, &p) {
			return 0, fmt.Errorf("llamacpp: position %d for slot %d rejected", p, seq)
		}
		if output {
			llamaBatchExtSetOutputLogits(m.batch, idx, true)
		}
		return idx, nil
	}
	for i, s := range steps {
		for j, t := range s.Tokens {
			last := j == len(s.Tokens)-1
			idx, err := add(s.Seq, s.Pos+j, t, s.Sample && last)
			if err != nil {
				return nil, err
			}
			if s.Sample && last {
				outIdx[i] = append(outIdx[i], idx)
			}
		}
		if drafts != nil {
			for j, t := range drafts[i] {
				idx, err := add(s.Seq, s.Pos+1+j, t, true)
				if err != nil {
					return nil, err
				}
				outIdx[i] = append(outIdx[i], idx)
			}
		}
	}
	m.tuneThreads(m.batchTokens(steps, drafts))
	m.chooseHead(steps)
	if rc := llamaProcess(m.ctx, LlamaProcessTypeDecode, m.batch); rc != 0 {
		return nil, m.decodeError(rc, steps)
	}

	var outs []backend.SeqOut
	for i, s := range steps {
		sl := &m.slots[s.Seq]
		if !s.Sample {
			// A prefill chunk: the carried state no longer precedes the
			// next token.
			sl.pendingPos = -1
			m.noteTokens(s, nil, 0)
			continue
		}
		var dr []backend.Token
		if drafts != nil {
			dr = drafts[i]
		}
		out, err := m.verify(s, outIdx[i], dr)
		if err != nil {
			for _, s := range steps {
				m.slots[s.Seq].histBad = true
			}
			return nil, err
		}
		m.noteTokens(s, dr, out.Accepted)
		if m.ngN > 0 {
			sl.ng.Result(len(dr), out.Accepted)
		}
		outs = append(outs, out)
	}
	return outs, nil
}

// verify samples one slot's output positions in order: the token after the
// last input, then one per draft while the sample equals the draft. The
// chain accepts each sample, so the grammar and the penalties see exactly
// the returned tokens. The rejected drafts leave the cache, and the hidden
// state of the input behind the last returned token is carried to the next
// step.
func (m *model) verify(s backend.SeqStep, idxs []int32, drafts []backend.Token) (backend.SeqOut, error) {
	sl := &m.slots[s.Seq]
	out := backend.SeqOut{Seq: s.Seq, Drafted: len(drafts)}
	for i, idx := range idxs {
		var row []float32
		var lp *logprobs
		if s.Logprobs >= 0 {
			if sl.onDevice {
				return out, fmt.Errorf("llamacpp: slot %d samples on the device and has no logits rows for logprobs", s.Seq)
			}
			var err error
			if row, err = m.logitsRow(idx); err != nil {
				return out, err
			}
			lp = m.logprobs(row, s.Logprobs)
		}
		tok := backend.Token(llamaSamplerSample(sl.chain, m.ctx, idx))
		out.Tokens = append(out.Tokens, tok)
		if lp != nil {
			lp.Token = tok
			lp.Logprob = lp.logprobOf(row, tok)
			out.Logprobs = append(out.Logprobs, lp.TokenLogprobs)
		}
		matched := i < len(drafts) && tok == drafts[i]
		if matched {
			out.Accepted++
		}
		if llamaVocabIsEog(m.vocab, int32(tok)) {
			out.EOG = true
			break
		}
		if !matched {
			break
		}
	}
	// Inputs Pos .. Pos+n-1 produced the n returned tokens; the cache keeps
	// them and drops the drafts after.
	n := len(out.Tokens)
	if len(drafts) > 0 && n < len(drafts)+1 {
		if !llamaMemorySeqRm(m.mem, int32(s.Seq), int32(s.Pos+n), -1) {
			return out, fmt.Errorf("llamacpp: slot %d: removing rejected drafts failed", s.Seq)
		}
	}
	sl.pendingPos = -1
	if m.dft != nil {
		lastIn := s.Pos + len(s.Tokens) - 1 + (n - 1)
		if h := llamaGetEmbeddingsNextnIth(m.ctx, idxs[n-1]); h != nil {
			if len(sl.pendingH) != m.dft.nEmbd {
				sl.pendingH = make([]float32, m.dft.nEmbd)
			}
			copy(sl.pendingH, unsafe.Slice(h, m.dft.nEmbd))
			sl.pendingPos = lastIn
		}
	}
	return out, nil
}

// draftLimits is how many tokens each step may draft: decode steps whose
// carried state sits right before them, within the step's MaxDraft, the
// drafter's limit, the slot's context, and what the batch has left of
// n_batch tokens and n_outputs_max output rows — given out in step order,
// as the scheduler orders its slots. nil when no step drafts.
func (m *model) draftLimits(steps []backend.SeqStep) []int {
	tokens, outputs := m.caps.NBatch, m.nOutputsMax
	for _, s := range steps {
		tokens -= len(s.Tokens)
		if s.Sample {
			outputs--
		}
	}
	var limits []int
	for i, s := range steps {
		if !s.Sample || len(s.Tokens) != 1 || s.MaxDraft <= 0 {
			continue
		}
		sl := &m.slots[s.Seq]
		// The MTP drafter needs the hidden state that sits right before the
		// step; prompt lookup needs only the slot's token history.
		if m.dft != nil && (sl.pendingPos < 0 || sl.pendingPos != s.Pos-1) {
			continue
		}
		n := min(s.MaxDraft, m.caps.MaxDraft, tokens, outputs, m.caps.NCtx-s.Pos-1)
		if n <= 0 {
			continue
		}
		if limits == nil {
			limits = make([]int, len(steps))
		}
		limits[i] = n
		tokens -= n
		outputs -= n
	}
	return limits
}

// checkSteps validates a batch before any of it reaches llama.cpp, which
// asserts on an empty batch and on a sampled token without logits.
func (m *model) checkSteps(steps []backend.SeqStep) error {
	if len(steps) == 0 {
		return errors.New("llamacpp: empty step")
	}
	seen := map[int]bool{}
	total, outputs := 0, 0
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
		if s.Sample {
			outputs++
		}
	}
	if total > m.caps.NBatch {
		return fmt.Errorf("llamacpp: step of %d tokens exceeds n_batch %d", total, m.caps.NBatch)
	}
	// llama.cpp asserts on more output rows than n_outputs_max.
	if outputs > m.nOutputsMax {
		return fmt.Errorf("llamacpp: %d sampling slots exceed n_outputs_max %d", outputs, m.nOutputsMax)
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
	m.slots[seq].pendingPos = -1
	if pos == 0 {
		llamaMemorySeqRm(m.mem, int32(seq), -1, -1)
		m.resetHistory(seq)
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
	if !llamaMemorySeqRm(m.mem, int32(seq), int32(pos), -1) {
		return false
	}
	if sl := &m.slots[seq]; len(sl.hist) > pos {
		sl.hist = sl.hist[:pos]
	}
	return true
}

// ClearSeq empties a slot's cache. Removing a whole sequence never fails.
func (m *model) ClearSeq(seq int) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 || m.checkSeq(seq) != nil {
		return
	}
	m.slots[seq].pendingPos = -1
	llamaMemorySeqRm(m.mem, int32(seq), -1, -1)
	m.resetHistory(seq)
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
	// The MTP carry-over is not part of llama.cpp's sequence state: the
	// first step after a restore runs without drafts.
	m.slots[seq].pendingPos = -1
	// The tokens behind a restored state are not in it.
	m.slots[seq].hist, m.slots[seq].histBad = nil, true
	m.slots[seq].outside = m.head != nil
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

// batchThreadsMinTokens is the batch size from which the prompt-processing
// team of threads (LoadSpec.ThreadsBatch) is used. llama.cpp switches to it for
// any batch of more than one token, which would put the verification of a
// few MTP drafts on the big team: decode is limited by memory bandwidth and
// peaks at fewer threads than prefill, which is limited by arithmetic (E4B on
// 16 cores: 20.5 tok/s decoding on 16 threads, 4.2 on 32; prefill 44.6 on 16
// and 50.6 on 32).
const batchThreadsMinTokens = 16

// batchTokens is the number of tokens the batch holds.
func (m *model) batchTokens(steps []backend.SeqStep, drafts [][]backend.Token) int {
	n := 0
	for i, s := range steps {
		n += len(s.Tokens)
		if drafts != nil {
			n += len(drafts[i])
		}
	}
	return n
}

// tuneThreads picks the size of the team for a batch of n tokens: the decode
// team (Threads) for small batches, ThreadsBatch for prompt chunks.
func (m *model) tuneThreads(n int) {
	threads, batch := int32(m.spec.Threads), int32(m.spec.ThreadsBatch)
	if threads <= 0 || batch <= 0 || batch == threads {
		return // llama.cpp's own choice is the only one
	}
	want := threads
	if n >= batchThreadsMinTokens {
		want = batch
	}
	if want != m.batchThreads {
		llamaSetNThreads(m.ctx, threads, want)
		m.batchThreads = want
	}
}
