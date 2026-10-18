// Package sched decides when work runs on a loaded model (SPEC §2): one loop
// per model owns its slots, and new requests join the running batch between
// steps (continuous batching).
//
// A step carries every active slot: prefill chunks and decode steps together,
// up to the model's batch size, with MTP drafts counted against the same
// budget — each decoding slot gets its token before any slot gets drafts, so
// a full batch slows drafting rather than starving a slot. Interactive
// requests go first. A free slot is chosen by session id, then by the longest
// prefix shared with the prompt, then by least recent use; the cache it holds
// is trimmed to the shared prefix, or cleared when the model cannot trim that
// far back.
//
// Device-level admission (the governor across models on one device) sits
// above this package, in the pool.
package sched
