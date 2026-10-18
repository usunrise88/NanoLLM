// Package pool keeps loaded models resident (SPEC §2): reference counting, LRU
// eviction, pinning and hot swap without interrupting in-flight requests. The
// pool key is (model, revision, backend, device, mode, preset, n_ctx).
//
// Each instance owns its scheduling loop (package sched), and through it the
// model's slots and the prompt cache that lives in them. A swap loads the new
// key and retires the old instances of the model: new requests go to the new
// one, running ones finish where they started, and a retired instance unloads
// when its last request releases it.
package pool
