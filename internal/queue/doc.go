// Package queue is the batch job queue, carried over from NanoASR (SPEC §2):
// SQLite (pure Go) for metadata, priorities and resumption after restart, an
// SSE event stream and HMAC-signed webhooks.
//
// Lands with v1.0.
package queue
