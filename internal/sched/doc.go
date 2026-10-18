// Package sched decides when work runs (SPEC §2): a weighted semaphore per
// device — CPU by thread budget, each GPU and NPU by one batch queue — and the
// continuous batching loop, in which new requests join the running batch
// between decode steps.
//
// Lands with v1.0.
package sched
