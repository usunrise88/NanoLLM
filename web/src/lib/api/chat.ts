import { ApiError, BASE, OPENAI_BASE } from './client'
import { postStream, readFrames } from './sse'
import type {
  ChatRequest,
  ChatResult,
  Delta,
  Logprob,
  Message,
  OpenAIChunk,
  Stats,
  ToolCall,
  Usage,
  Warning,
} from './types'

/**
 * Streaming chat in both dialects, reduced to one shape.
 *
 * The native dialect streams core.Delta and ends with the whole ChatResult;
 * the OpenAI dialect streams chunks whose last one carries finish_reason and
 * the `stats` extension. The screen does not care which it called: both are
 * folded into a Turn while streaming and into a ChatResult at the end.
 */
export type Dialect = 'native' | 'openai'

/** An assistant message as it is being streamed. */
export interface Turn {
  content: string
  reasoning: string
  toolCalls: ToolCall[]
  logprobs: Logprob[]
}

export function emptyTurn(): Turn {
  return { content: '', reasoning: '', toolCalls: [], logprobs: [] }
}

/** Folds one native delta into the turn. Pure: returns a new turn. */
export function applyDelta(turn: Turn, d: Delta): Turn {
  return {
    content: turn.content + (d.content ?? ''),
    reasoning: turn.reasoning + (d.reasoning_content ?? ''),
    toolCalls: d.tool_calls?.length ? [...turn.toolCalls, ...d.tool_calls] : turn.toolCalls,
    logprobs: d.logprobs?.length ? [...turn.logprobs, ...d.logprobs] : turn.logprobs,
  }
}

/** What an OpenAI stream has said so far. */
export interface OpenAIState {
  turn: Turn
  finish?: string
  stats?: Stats
  warnings?: Warning[]
  usage?: Usage
}

/**
 * Folds one OpenAI chunk into the state.
 *
 * Tool calls arrive as fragments keyed by `index`: the first fragment of a
 * call carries its id and name, later ones append to its arguments. This
 * server sends each call whole, but other servers do not, and a parser that
 * assumed whole calls would break the day someone points it elsewhere.
 */
export function applyOpenAIChunk(state: OpenAIState, chunk: OpenAIChunk): OpenAIState {
  const next: OpenAIState = { ...state, turn: { ...state.turn } }
  for (const choice of chunk.choices ?? []) {
    const d = choice.delta
    if (d) {
      if (d.content) next.turn.content += d.content
      if (d.reasoning_content) next.turn.reasoning += d.reasoning_content
      if (d.tool_calls?.length) {
        const calls = [...next.turn.toolCalls]
        for (const frag of d.tool_calls) {
          const i = frag.index ?? calls.length
          const prev = calls[i]
          if (!prev || (frag.id && frag.id !== prev.id)) {
            calls[i] = {
              id: frag.id ?? '',
              name: frag.function.name ?? '',
              arguments: frag.function.arguments ?? '',
            }
          } else {
            calls[i] = {
              ...prev,
              name: prev.name || (frag.function.name ?? ''),
              arguments: prev.arguments + (frag.function.arguments ?? ''),
            }
          }
        }
        next.turn.toolCalls = calls
      }
    }
    if (choice.logprobs?.content?.length) {
      next.turn.logprobs = [
        ...next.turn.logprobs,
        ...choice.logprobs.content.map((c) => ({
          token: c.token,
          logprob: c.logprob,
          ...(c.top_logprobs?.length ? { top_logprobs: c.top_logprobs } : {}),
        })),
      ]
    }
    if (choice.finish_reason) next.finish = choice.finish_reason
  }
  if (chunk.stats) next.stats = chunk.stats
  if (chunk.warnings) next.warnings = chunk.warnings
  if (chunk.usage) {
    next.usage = {
      prompt_tokens: chunk.usage.prompt_tokens,
      completion_tokens: chunk.usage.completion_tokens,
      cached_tokens: chunk.usage.prompt_tokens_details?.cached_tokens ?? 0,
      reasoning_tokens: chunk.usage.completion_tokens_details?.reasoning_tokens ?? 0,
    }
  }
  return next
}

/** The final OpenAI state as the ChatResult the native dialect would have sent. */
export function openAIResult(state: OpenAIState): ChatResult {
  const s = state.stats
  const message: Message = { role: 'assistant', content: state.turn.content }
  if (state.turn.reasoning) message.reasoning_content = state.turn.reasoning
  if (state.turn.toolCalls.length) message.tool_calls = state.turn.toolCalls
  return {
    message,
    finish_reason: state.finish ?? 'stop',
    ...(state.turn.logprobs.length ? { logprobs: state.turn.logprobs } : {}),
    usage: state.usage ?? {
      prompt_tokens: s?.prompt_tokens ?? 0,
      cached_tokens: s?.cached_tokens ?? 0,
      completion_tokens: s?.completion_tokens ?? 0,
      reasoning_tokens: s?.reasoning_tokens ?? 0,
    },
    stats: s ?? emptyStats(),
    ...(state.warnings?.length ? { warnings: state.warnings } : {}),
  }
}

function emptyStats(): Stats {
  return {
    stages_ms: {},
    ttft_ms: 0,
    prefill_tok_s: 0,
    decode_tok_s: 0,
    prompt_tokens: 0,
    cached_tokens: 0,
    completion_tokens: 0,
    reasoning_tokens: 0,
    backend: '',
    device: '',
    slot: 0,
  }
}

/**
 * The native request as the OpenAI dialect spells it. session_id and device
 * have no OpenAI spelling and are dropped: prefix reuse still happens, slot
 * pinning does not.
 */
export function toOpenAIRequest(req: ChatRequest): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: req.model ?? '',
    stream: true,
    stream_options: { include_usage: true },
    messages: req.messages.map((m) => {
      const out: Record<string, unknown> = { role: m.role, content: m.content }
      if (m.reasoning_content) out.reasoning_content = m.reasoning_content
      if (m.tool_call_id) out.tool_call_id = m.tool_call_id
      if (m.name) out.name = m.name
      if (m.tool_calls?.length) {
        out.tool_calls = m.tool_calls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: c.arguments },
        }))
      }
      return out
    }),
    ...req.sampling,
  }
  if (req.max_tokens) body.max_tokens = req.max_tokens
  if (req.stop?.length) body.stop = req.stop
  if (req.logprobs) body.logprobs = true
  if (req.top_logprobs) body.top_logprobs = req.top_logprobs
  if (req.enable_thinking !== undefined) body.enable_thinking = req.enable_thinking
  if (req.max_reasoning_tokens) body.max_reasoning_tokens = req.max_reasoning_tokens
  if (req.tools?.length) {
    body.tools = req.tools.map((t) => ({
      type: 'function',
      function: {
        name: t.name,
        ...(t.description ? { description: t.description } : {}),
        ...(t.parameters !== undefined ? { parameters: t.parameters } : {}),
      },
    }))
  }
  if (req.response_format && req.response_format.type !== 'text') {
    body.response_format =
      req.response_format.type === 'json_schema'
        ? { type: 'json_schema', json_schema: { name: 'schema', schema: req.response_format.schema } }
        : { type: req.response_format.type }
  }
  return body
}

/** A problem object or an OpenAI error envelope sent as an event mid-stream. */
function streamError(payload: unknown): ApiError {
  const p = payload as {
    status?: number
    code?: string
    detail?: string
    title?: string
    error?: { message?: string; code?: string; param?: string }
  }
  if (p.error) {
    return new ApiError({
      status: 500,
      code: p.error.code ?? 'internal',
      detail: p.error.message ?? 'stream failed',
      param: p.error.param,
    })
  }
  return new ApiError({
    status: p.status ?? 500,
    code: p.code ?? 'internal',
    detail: p.detail ?? p.title ?? 'stream failed',
  })
}

export interface ChatStreamOptions {
  dialect: Dialect
  signal?: AbortSignal
  /** Called with the whole turn so far after every piece. */
  onTurn: (turn: Turn) => void
}

/**
 * Runs one streamed chat request to its end.
 *
 * Resolves with the result, or null when the stream ended without one (the
 * person pressed stop: the server frees the slot on disconnect and sends
 * nothing more). Rejects with an ApiError for a refusal, whether it came as a
 * status before the stream or as an error event inside it.
 */
export async function streamChat(req: ChatRequest, opts: ChatStreamOptions): Promise<ChatResult | null> {
  if (opts.dialect === 'openai') return streamOpenAI(req, opts)

  const response = await postStream(`${BASE}/chat`, { ...req, stream: true }, opts.signal)
  let turn = emptyTurn()
  let result: ChatResult | null = null
  let failure: ApiError | undefined
  try {
    await readFrames(response, (frame) => {
      let payload: unknown
      try {
        payload = JSON.parse(frame.data)
      } catch {
        return
      }
      switch (frame.event) {
        case 'delta':
          turn = applyDelta(turn, payload as Delta)
          opts.onTurn(turn)
          return
        case 'done':
          result = payload as ChatResult
          return true
        case 'error':
          failure = streamError(payload)
          return true
        default:
          return
      }
    })
  } catch (err) {
    if (opts.signal?.aborted) return null
    throw err
  }
  if (failure) throw failure
  return result
}

async function streamOpenAI(req: ChatRequest, opts: ChatStreamOptions): Promise<ChatResult | null> {
  const response = await postStream(`${OPENAI_BASE}/chat/completions`, toOpenAIRequest(req), opts.signal)
  let state: OpenAIState = { turn: emptyTurn() }
  let finished = false
  let failure: ApiError | undefined
  try {
    await readFrames(response, (frame) => {
      if (frame.data === '[DONE]') {
        finished = true
        return true
      }
      let chunk: OpenAIChunk & { error?: unknown }
      try {
        chunk = JSON.parse(frame.data) as OpenAIChunk & { error?: unknown }
      } catch {
        return
      }
      if (chunk.error) {
        failure = streamError(chunk)
        return true
      }
      state = applyOpenAIChunk(state, chunk)
      opts.onTurn(state.turn)
      return
    })
  } catch (err) {
    if (opts.signal?.aborted) return null
    throw err
  }
  if (failure) throw failure
  return finished || state.finish ? openAIResult(state) : null
}
