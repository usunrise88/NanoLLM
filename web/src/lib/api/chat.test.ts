import { afterEach, describe, expect, it, vi } from 'vitest'

import { applyDelta, applyOpenAIChunk, emptyTurn, openAIResult, streamChat, toOpenAIRequest } from './chat'
import { ApiError } from './client'
import { chunked } from './testing'
import type { ChatRequest, OpenAIChunk } from './types'

afterEach(() => {
  vi.unstubAllGlobals()
})

function sse(events: [string, unknown][]): string {
  return events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join('')
}

const stats = {
  stages_ms: { prefill: 1 },
  ttft_ms: 12.5,
  prefill_tok_s: 900,
  decode_tok_s: 40,
  prompt_tokens: 45,
  cached_tokens: 9,
  completion_tokens: 76,
  reasoning_tokens: 45,
  backend: 'fake',
  device: 'gpu0',
  draft_acceptance: 0.58,
  slot: 0,
}

function chunk(delta: OpenAIChunk['choices'][number]['delta'], extra: Partial<OpenAIChunk> = {}, finish: string | null = null): OpenAIChunk {
  return {
    id: 'c',
    object: 'chat.completion.chunk',
    model: 'm',
    choices: [{ index: 0, ...(delta ? { delta } : {}), finish_reason: finish }],
    ...extra,
  }
}

describe('applyDelta', () => {
  it('keeps reasoning and content apart and appends tool calls', () => {
    let t = emptyTurn()
    t = applyDelta(t, { reasoning_content: 'think' })
    t = applyDelta(t, { reasoning_content: 'ing' })
    t = applyDelta(t, { content: 'Hi' })
    t = applyDelta(t, { tool_calls: [{ id: '1', name: 'f', arguments: '{}' }] })
    expect(t).toEqual({
      content: 'Hi',
      reasoning: 'thinking',
      toolCalls: [{ id: '1', name: 'f', arguments: '{}' }],
      logprobs: [],
    })
  })

  it('does not mutate the turn it was given', () => {
    const before = emptyTurn()
    applyDelta(before, { content: 'x', logprobs: [{ token: 'x', logprob: -0.1 }] })
    expect(before).toEqual(emptyTurn())
  })
})

describe('applyOpenAIChunk', () => {
  // Other servers stream a call as fragments: id and name first, then pieces
  // of the arguments. The fold must glue them back together by index.
  it('joins tool-call fragments by index', () => {
    let s = { turn: emptyTurn() }
    s = applyOpenAIChunk(s, chunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'get', arguments: '{"a"' } }] }))
    s = applyOpenAIChunk(s, chunk({ tool_calls: [{ index: 0, function: { arguments: ':1}' } }] }))
    s = applyOpenAIChunk(s, chunk({ tool_calls: [{ index: 1, id: 'call_2', function: { name: 'put', arguments: '{}' } }] }))
    expect(s.turn.toolCalls).toEqual([
      { id: 'call_1', name: 'get', arguments: '{"a":1}' },
      { id: 'call_2', name: 'put', arguments: '{}' },
    ])
  })

  it('takes stats and finish_reason from the last chunk, usage from the usage chunk', () => {
    let s = applyOpenAIChunk({ turn: emptyTurn() }, chunk({ role: 'assistant', content: '' }))
    s = applyOpenAIChunk(s, chunk({ reasoning_content: 'hmm' }))
    s = applyOpenAIChunk(s, chunk({ content: 'Answer' }))
    s = applyOpenAIChunk(s, chunk({}, { stats, warnings: [{ code: 'device_fallback', message: 'cpu' }] }, 'stop'))
    s = applyOpenAIChunk(s, {
      id: 'c',
      object: 'chat.completion.chunk',
      model: 'm',
      choices: [],
      usage: {
        prompt_tokens: 45,
        completion_tokens: 76,
        prompt_tokens_details: { cached_tokens: 9 },
        completion_tokens_details: { reasoning_tokens: 45 },
      },
    })
    const r = openAIResult(s)
    expect(r.message).toEqual({ role: 'assistant', content: 'Answer', reasoning_content: 'hmm' })
    expect(r.finish_reason).toBe('stop')
    expect(r.stats.draft_acceptance).toBe(0.58)
    expect(r.usage).toEqual({ prompt_tokens: 45, completion_tokens: 76, cached_tokens: 9, reasoning_tokens: 45 })
    expect(r.warnings).toHaveLength(1)
  })

  it('collects logprobs from choices', () => {
    const s = applyOpenAIChunk(
      { turn: emptyTurn() },
      {
        id: 'c',
        object: 'chat.completion.chunk',
        model: 'm',
        choices: [{ index: 0, delta: { content: 'a' }, logprobs: { content: [{ token: 'a', logprob: -0.5 }] }, finish_reason: null }],
      },
    )
    expect(s.turn.logprobs).toEqual([{ token: 'a', logprob: -0.5 }])
  })
})

describe('toOpenAIRequest', () => {
  it('spells tools, tool calls, sampling and json_schema the OpenAI way', () => {
    const req: ChatRequest = {
      model: 'gemma',
      session_id: 's1',
      device: 'fake:cpu',
      messages: [
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'f', arguments: '{}' }] },
        { role: 'tool', content: '42', tool_call_id: 'c1', name: 'f' },
      ],
      sampling: { temperature: 0.2, top_k: 40 },
      tools: [{ name: 'f', parameters: { type: 'object' } }],
      response_format: { type: 'json_schema', schema: { type: 'object' } },
      enable_thinking: true,
      max_tokens: 10,
    }
    const body = toOpenAIRequest(req)
    expect(body).toMatchObject({
      model: 'gemma',
      stream: true,
      temperature: 0.2,
      top_k: 40,
      max_tokens: 10,
      enable_thinking: true,
      tools: [{ type: 'function', function: { name: 'f', parameters: { type: 'object' } } }],
      response_format: { type: 'json_schema', json_schema: { name: 'schema', schema: { type: 'object' } } },
    })
    expect((body.messages as unknown[])[1]).toEqual({
      role: 'assistant',
      content: '',
      tool_calls: [{ id: 'c1', type: 'function', function: { name: 'f', arguments: '{}' } }],
    })
    // No OpenAI spelling exists for these; sending them would be a 400.
    expect(body).not.toHaveProperty('session_id')
    expect(body).not.toHaveProperty('device')
  })
})

describe('streamChat', () => {
  const req: ChatRequest = { messages: [{ role: 'user', content: 'hi' }], session_id: 's1' }

  it('streams native deltas and resolves with the done result', async () => {
    const fetch = vi.fn(async (_url: string, _init?: RequestInit) =>
      chunked(
        sse([
          ['delta', { reasoning_content: 'R' }],
          ['delta', { content: 'He' }],
        ]),
        sse([
          ['delta', { content: 'llo' }],
          ['done', { message: { role: 'assistant', content: 'Hello', reasoning_content: 'R' }, finish_reason: 'stop', usage: {}, stats }],
        ]),
      ),
    )
    vi.stubGlobal('fetch', fetch)
    const turns: string[] = []
    const result = await streamChat(req, { dialect: 'native', onTurn: (t) => turns.push(t.content) })

    expect(result?.message.content).toBe('Hello')
    expect(result?.stats.ttft_ms).toBe(12.5)
    expect(turns).toEqual(['', 'He', 'Hello'])
    const [url, init] = fetch.mock.calls[0] ?? []
    expect(url).toBe('/api/v1/chat')
    expect(JSON.parse(String(init?.body))).toEqual({ ...req, stream: true })
  })

  it('throws the problem of an error event', async () => {
    vi.stubGlobal('fetch', async () =>
      chunked(sse([['delta', { content: 'x' }], ['error', { status: 500, code: 'internal', detail: 'scripted' }]])),
    )
    await expect(streamChat(req, { dialect: 'native', onTurn: () => {} })).rejects.toMatchObject({
      code: 'internal',
      message: 'scripted',
    })
  })

  // A refusal before the stream opens is an ordinary problem response.
  it('throws an ApiError for a problem response', async () => {
    vi.stubGlobal(
      'fetch',
      async () =>
        new Response(JSON.stringify({ status: 404, code: 'model_not_found', detail: 'no model "x"', param: 'model' }), {
          status: 404,
          headers: { 'Content-Type': 'application/problem+json' },
        }),
    )
    const err = await streamChat(req, { dialect: 'native', onTurn: () => {} }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err).toMatchObject({ status: 404, code: 'model_not_found', param: 'model' })
  })

  it('reads the OpenAI stream to [DONE]', async () => {
    const fetch = vi.fn(async (_url: string, _init?: RequestInit) =>
      chunked(
        `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }))}\n\n`,
        `data: ${JSON.stringify(chunk({ content: 'Hi' }))}\n\n`,
        `data: ${JSON.stringify(chunk({}, { stats }, 'stop'))}\n\ndata: [DONE]\n\n`,
      ),
    )
    vi.stubGlobal('fetch', fetch)
    const result = await streamChat(req, { dialect: 'openai', onTurn: () => {} })
    expect(fetch.mock.calls[0]?.[0]).toBe('/v1/chat/completions')
    expect(result?.message.content).toBe('Hi')
    expect(result?.stats.device).toBe('gpu0')
  })

  it('throws the OpenAI error envelope sent inside the stream', async () => {
    vi.stubGlobal('fetch', async () =>
      chunked(`data: ${JSON.stringify({ error: { message: 'boom', type: 'server_error', code: 'internal' } })}\n\ndata: [DONE]\n\n`),
    )
    await expect(streamChat(req, { dialect: 'openai', onTurn: () => {} })).rejects.toMatchObject({ message: 'boom' })
  })

  // The stop button aborts the fetch; that is not a failure to report.
  it('resolves null when aborted mid-stream', async () => {
    const controller = new AbortController()
    vi.stubGlobal('fetch', async () => {
      const enc = new TextEncoder()
      return new Response(
        new ReadableStream({
          start(c) {
            c.enqueue(enc.encode(sse([['delta', { content: 'la ' }]])))
            controller.signal.addEventListener('abort', () => c.error(new DOMException('aborted', 'AbortError')))
          },
        }),
      )
    })
    const result = streamChat(req, {
      dialect: 'native',
      signal: controller.signal,
      onTurn: () => controller.abort(),
    })
    await expect(result).resolves.toBeNull()
  })
})
