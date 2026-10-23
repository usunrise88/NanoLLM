package llamacpp

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
)

// SetThreads moves the decode and the prompt-processing teams of the target and
// of the drafter (backend.Tuner). It runs on the model's loop, between steps:
// the next Step picks the team by its batch size as always.
func (m *model) SetThreads(threads, batch int) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return errClosed
	}
	if threads <= 0 || batch <= 0 {
		return fmt.Errorf("llamacpp: %d decode and %d prompt threads, both must be positive", threads, batch)
	}
	m.spec.Threads, m.spec.ThreadsBatch = threads, batch
	llamaSetNThreads(m.ctx, int32(threads), int32(batch))
	m.batchThreads = int32(batch)
	if m.dft != nil {
		llamaSetNThreads(m.dft.ctx, int32(threads), int32(batch))
	}
	return nil
}

// SetDraft changes the drafting limits (backend.Tuner). The fields are the ones
// native_mtp_test.go sweeps: the scheduler reads MaxDraft from Capabilities
// (sched.Loop.Do re-reads it), the drafter reads the rest.
func (m *model) SetDraft(max int, pMin float64) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return errClosed
	}
	if max < 0 || pMin < 0 || pMin > 1 || math.IsNaN(pMin) {
		return errors.New("llamacpp: draft max or p_min out of range")
	}
	if m.dft == nil && m.ngN == 0 {
		return errors.New("llamacpp: no drafter is loaded")
	}
	if max > m.draftCeiling {
		return fmt.Errorf("llamacpp: draft max %d is above the %d the context was created for", max, m.draftCeiling)
	}
	m.caps.MaxDraft = max
	if m.dft != nil {
		m.dft.maxDraft, m.dft.pMin = max, pMin
	}
	return nil
}

func (m *model) DraftCeiling() int { return m.draftCeiling }

// TimeBatch evaluates tokens after the slot's cache, waits for the device and
// cuts the cache back (backend.BatchTimer). The batch has the shape of a
// verification step: the last outputs tokens ask for logits.
func (m *model) TimeBatch(ctx context.Context, seq int, tokens []backend.Token, outputs int) (time.Duration, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return 0, errClosed
	}
	if err := m.checkSeq(seq); err != nil {
		return 0, err
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	n := len(tokens)
	if n == 0 || n > m.caps.NBatch || outputs < 0 || outputs > n || outputs > m.nOutputsMax {
		return 0, fmt.Errorf("llamacpp: a timed batch of %d tokens with %d outputs does not fit n_batch %d and n_outputs_max %d", n, outputs, m.caps.NBatch, m.nOutputsMax)
	}
	if err := m.checkTokens(tokens); err != nil {
		return 0, err
	}
	pmax := int(llamaMemorySeqPosMax(m.mem, int32(seq)))
	pos := pmax + 1
	if pos+n > m.caps.NCtx {
		return 0, fmt.Errorf("llamacpp: a timed batch would reach position %d, past the context of %d", pos+n, m.caps.NCtx)
	}
	// With sliding-window layers the batch may displace cells of the window the
	// cache would need once the batch is gone: the same test TrimSeq applies
	// before a cut.
	if m.nSWA > 0 && !m.swaFull && pos > 0 && int(llamaMemorySeqPosMin(m.mem, int32(seq))) > max(0, pos-m.nSWA) {
		return 0, backend.ErrNoRollback
	}

	llamaBatchExtClear(m.batch)
	for i, t := range tokens {
		idx := llamaBatchExtAddToken(m.batch, int32(seq), int32(t))
		p := int32(pos + i)
		if idx < 0 || !llamaBatchExtSetPos(m.batch, idx, &p) {
			return 0, fmt.Errorf("llamacpp: batch entry %d of a timed batch rejected", i)
		}
		if i >= n-outputs {
			llamaBatchExtSetOutputLogits(m.batch, idx, true)
		}
	}
	m.tuneThreads(n)
	start := time.Now()
	rc := llamaProcess(m.ctx, LlamaProcessTypeDecode, m.batch)
	llamaSynchronize(m.ctx)
	d := time.Since(start)
	// Whatever the outcome, the evaluated positions leave the cache and the
	// carried drafting state no longer precedes the next token.
	rolled := llamaMemorySeqRm(m.mem, int32(seq), int32(pos), -1)
	m.slots[seq].pendingPos = -1
	if rc != 0 {
		return 0, m.decodeError(rc, []backend.SeqStep{{Seq: seq, Pos: pos}})
	}
	if !rolled {
		return d, backend.ErrNoRollback
	}
	return d, nil
}

var (
	_ backend.Tuner      = (*model)(nil)
	_ backend.BatchTimer = (*model)(nil)
)
