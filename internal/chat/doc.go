// Package chat renders chat templates and parses what the model writes back
// (SPEC §2, §5): messages, tools and the thinking flag go in through the
// model's template; tool calls and the reasoning block come out through a
// per-family parser.
//
// Whether templates are rendered by a Go Jinja implementation or by llama.cpp's
// built-in templates plus hand-written Gemma 4 and Qwen3.5 ones is an open
// question in SPEC §14.
//
// Lands with v1.0.
package chat
