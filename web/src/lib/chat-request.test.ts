import { describe, expect, it } from 'vitest'

import { buildRequest, parseTools, PrefsError } from './chat-request'
import type { ChatMessage } from './conversations'
import { DEFAULT_PREFS, type ChatPrefs } from './prefs'

const history: ChatMessage[] = [
  { id: 'm1', role: 'user', content: 'hi' },
  {
    id: 'm2',
    role: 'assistant',
    content: 'hello',
    reasoning_content: 'greet back',
    result: {
      finish_reason: 'stop',
      usage: { prompt_tokens: 1, cached_tokens: 0, completion_tokens: 1, reasoning_tokens: 0 },
      stats: {
        stages_ms: {},
        ttft_ms: 1,
        prefill_tok_s: 1,
        decode_tok_s: 1,
        prompt_tokens: 1,
        cached_tokens: 0,
        completion_tokens: 1,
        reasoning_tokens: 0,
        backend: 'fake',
        device: 'cpu',
        slot: 0,
      },
    },
  },
  { id: 'm3', role: 'assistant', content: '', error: 'Generation failed: boom' },
  { id: 'm4', role: 'user', content: 'again' },
]

function build(prefs: Partial<ChatPrefs> = {}) {
  return buildRequest({ model: 'gemma', sessionId: 'ui-1', history, prefs: { ...DEFAULT_PREFS, ...prefs } })
}

describe('buildRequest', () => {
  // The session id is what lets the next turn land in the slot that already
  // holds the conversation's prefix.
  it('carries the session id and the thinking switch on every request', () => {
    const req = build({ thinking: true })
    expect(req.session_id).toBe('ui-1')
    expect(req.enable_thinking).toBe(true)
    expect(build().enable_thinking).toBe(false)
  })

  it('sends the history without UI fields and without failed turns', () => {
    const req = build({ system: 'be brief' })
    expect(req.messages).toEqual([
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello', reasoning_content: 'greet back' },
      { role: 'user', content: 'again' },
    ])
  })

  // An empty field means "the model card's value", which is not zero.
  it('leaves unset sampling out entirely', () => {
    const req = build()
    expect(req).not.toHaveProperty('sampling')
    expect(req).not.toHaveProperty('max_tokens')
    expect(req).not.toHaveProperty('response_format')
    expect(req).not.toHaveProperty('cache_prompt')
  })

  it('maps the options onto core.ChatRequest', () => {
    const req = build({
      temperature: '0',
      topK: '40',
      seed: '7',
      maxTokens: '128',
      stop: 'END\n\\n\\n',
      device: 'fake:cpu',
      cachePrompt: false,
      logprobs: true,
      topLogprobs: '3',
      responseFormat: 'json_schema',
      schema: '{"type":"object"}',
      tools: '[{"type":"function","function":{"name":"get","parameters":{"type":"object"}}}]',
    })
    expect(req).toMatchObject({
      sampling: { temperature: 0, top_k: 40, seed: 7 },
      max_tokens: 128,
      stop: ['END', '\n\n'],
      device: 'fake:cpu',
      cache_prompt: false,
      logprobs: true,
      top_logprobs: 3,
      response_format: { type: 'json_schema', schema: { type: 'object' } },
      tools: [{ name: 'get', parameters: { type: 'object' } }],
    })
  })

  it('sends max_reasoning_tokens only with thinking on', () => {
    expect(build({ maxReasoningTokens: '64' })).not.toHaveProperty('max_reasoning_tokens')
    expect(build({ maxReasoningTokens: '64', thinking: true }).max_reasoning_tokens).toBe(64)
  })

  it('names the field that does not parse', () => {
    expect(() => build({ topK: '1.5' })).toThrow(PrefsError)
    try {
      build({ responseFormat: 'json_schema', schema: '{' })
    } catch (e) {
      expect((e as PrefsError).field).toBe('schema')
    }
  })
})

describe('parseTools', () => {
  it('accepts the core and the OpenAI spelling', () => {
    expect(parseTools('[{"name":"a"},{"type":"function","function":{"name":"b","description":"d"}}]')).toEqual([
      { name: 'a' },
      { name: 'b', description: 'd' },
    ])
  })

  it('refuses a tool without a name', () => {
    expect(() => parseTools('[{"description":"x"}]')).toThrow(/needs a name/)
  })

  it('is undefined for an empty field', () => {
    expect(parseTools('  ')).toBeUndefined()
  })
})
