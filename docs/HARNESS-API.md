# Harness API

NanoBird's native backend (SPEC §7.2 of the NanoBird specification) reads the features below from NanoLLM's native API (`/api/v1`). Each one is additive: a request that omits it behaves as before.

## Model capabilities (RFC R1)

`GET /api/v1/models/{id}/capabilities` reports what a catalog model can do through this API: `thinking`, `tool_formats`, `reasoning_budget`, `parse_report`, `assistant_prefix`, `prefix_cache` and `tokenize`. The answer follows the model's family, and Gemma 4 has all of them. A model that is not in the catalog answers 404.

## Tool format (RFC R4)

`tool_format` on `POST /api/v1/chat` chooses how tool calls are read:

- `strict`, the default when the field is omitted: with tools offered, the output is constrained by the tool grammar, lazily, as before. Text stays free until the model opens a call.
- `rescue`: no grammar. The Gemma 4 parser reads the calls from the text, and reports what it cannot read (see the parse report). `tool_choice: required` is refused with a 400, because it needs the grammar.

An unknown value is a 400 with `param: tool_format`.

## Parse report (RFC R4)

A chat result from a request that the parser reads carries `parse`:

- `status: ok`: every call and channel was read as written.
- `status: repaired`: the parser dropped words that the model wrote after its calls. `raw` holds the whole reply text.
- `status: failed`: a call could not be read. `raw` holds the whole reply text and `error` says why. Under `rescue`, the message content is that raw text and holds no calls, so the caller reads the calls from the text itself.

The final `done` event of a stream carries the same report. After a failure, the deltas streamed before it may hold partial calls, so the final record is the authority. Outside `rescue`, a call that breaks the stream still fails the request, as before; a failure found at the end of the stream stays a `parse_error` warning, and the report says `failed`.

## Assistant prefix (RFC R5)

`assistant_prefix` is text that the model continues in its own turn. The template opens the model's turn and appends the prefix, and the reply holds only what the model adds after it.

## Tokenization (RFC R5)

`POST /api/v1/tokenize` and `POST /api/v1/detokenize`, as before.

## Prompt cache (RFC R2)

Prefix handles are not built. `cache_prompt` and `session_id` reuse the cached prompt, and `usage.cached_tokens` reports the reuse. NanoBird relies on this implicit cache, as its specification allows.

## Not built

- Prefix handles (R2): see above.
- Cancellation that keeps partial output (R5): a cancelled request ends without its partial reply, and the client discards the partial turn.
- Session snapshots (R6): later.
- The reasoning budget (R3) is `max_reasoning_tokens`, which closes the thinking block and lets the answer follow. It was already in the native API.
