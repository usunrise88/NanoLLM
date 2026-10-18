// Package pool keeps loaded models resident (SPEC §2): reference counting, LRU
// eviction, pinning and hot revision swap without interrupting in-flight
// requests. The pool key is (model, revision, backend, device, mode, preset,
// n_ctx).
//
// Each loaded model owns N context slots — parallel sequences in one llama.cpp
// context — and the prompt cache that lives in them: plain prefix reuse for
// Gemma 4, state snapshots at checkpoints for the Qwen3.5 fallback, whose
// Gated DeltaNet state cannot be trimmed.
//
// Lands with v1.0.
package pool
