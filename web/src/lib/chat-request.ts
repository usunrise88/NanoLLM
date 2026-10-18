import type { ChatRequest, Message, Sampling, Tool } from './api/types'
import type { ChatMessage } from './conversations'
import type { ChatPrefs } from './prefs'

/**
 * Turns a conversation and the chat settings into the request the server
 * gets. Pure, so what is sent can be tested without a browser.
 */
export class PrefsError extends Error {
  readonly field: keyof ChatPrefs

  constructor(field: keyof ChatPrefs, message: string) {
    super(message)
    this.field = field
  }
}

function num(prefs: ChatPrefs, field: keyof ChatPrefs, integer = false): number | undefined {
  const raw = String(prefs[field]).trim()
  if (raw === '') return undefined
  const v = Number(raw)
  if (!Number.isFinite(v) || (integer && !Number.isInteger(v))) {
    throw new PrefsError(field, `${field}: "${raw}" is not a ${integer ? 'whole ' : ''}number`)
  }
  return v
}

export function parseTools(text: string): Tool[] | undefined {
  if (text.trim() === '') return undefined
  let v: unknown
  try {
    v = JSON.parse(text)
  } catch (e) {
    throw new PrefsError('tools', `tools: ${e instanceof Error ? e.message : String(e)}`)
  }
  if (!Array.isArray(v)) throw new PrefsError('tools', 'tools: expected a JSON array')
  return v.map((t, i) => {
    // The OpenAI spelling {type: function, function: {...}} is accepted too,
    // since that is what people have on their clipboard.
    const tool = (t && typeof t === 'object' && 'function' in t ? (t as { function: unknown }).function : t) as Tool
    if (!tool || typeof tool.name !== 'string' || tool.name === '') {
      throw new PrefsError('tools', `tools[${i}]: a tool needs a name`)
    }
    return {
      name: tool.name,
      ...(tool.description ? { description: tool.description } : {}),
      ...(tool.parameters !== undefined ? { parameters: tool.parameters } : {}),
    }
  })
}

/** The wire message, without the fields only this UI keeps. */
export function wireMessage(m: ChatMessage | Message): Message {
  const out: Message = { role: m.role, content: m.content }
  if (m.reasoning_content) out.reasoning_content = m.reasoning_content
  if (m.tool_calls?.length) out.tool_calls = m.tool_calls
  if (m.tool_call_id) out.tool_call_id = m.tool_call_id
  if (m.name) out.name = m.name
  return out
}

export function buildRequest(opts: {
  model: string
  sessionId: string
  history: (ChatMessage | Message)[]
  prefs: ChatPrefs
}): ChatRequest {
  const { prefs } = opts
  const sampling: Sampling = {}
  const temperature = num(prefs, 'temperature')
  const topP = num(prefs, 'topP')
  const topK = num(prefs, 'topK', true)
  const seed = num(prefs, 'seed', true)
  if (temperature !== undefined) sampling.temperature = temperature
  if (topP !== undefined) sampling.top_p = topP
  if (topK !== undefined) sampling.top_k = topK
  if (seed !== undefined) sampling.seed = seed

  const messages: Message[] = []
  if (prefs.system.trim()) messages.push({ role: 'system', content: prefs.system })
  // Failed turns are shown but never sent back: the model did not say them.
  for (const m of opts.history) {
    if ('error' in m && m.error && !m.content && !m.tool_calls?.length) continue
    messages.push(wireMessage(m))
  }

  const req: ChatRequest = {
    messages,
    enable_thinking: prefs.thinking,
    session_id: opts.sessionId,
  }
  if (opts.model) req.model = opts.model
  if (Object.keys(sampling).length) req.sampling = sampling

  const maxTokens = num(prefs, 'maxTokens', true)
  if (maxTokens !== undefined) req.max_tokens = maxTokens
  const maxReasoning = num(prefs, 'maxReasoningTokens', true)
  if (maxReasoning !== undefined && prefs.thinking) req.max_reasoning_tokens = maxReasoning

  const stop = prefs.stop
    .split('\n')
    .map((s) => s.replace(/\\n/g, '\n'))
    .filter((s) => s !== '')
  if (stop.length) req.stop = stop

  if (prefs.device) req.device = prefs.device
  if (!prefs.cachePrompt) req.cache_prompt = false

  if (prefs.logprobs) {
    req.logprobs = true
    const top = num(prefs, 'topLogprobs', true)
    if (top) req.top_logprobs = top
  }

  if (prefs.responseFormat === 'json_object') {
    req.response_format = { type: 'json_object' }
  } else if (prefs.responseFormat === 'json_schema') {
    let schema: unknown
    try {
      schema = JSON.parse(prefs.schema)
    } catch (e) {
      throw new PrefsError('schema', `schema: ${e instanceof Error ? e.message : String(e)}`)
    }
    req.response_format = { type: 'json_schema', schema }
  }

  const tools = parseTools(prefs.tools)
  if (tools?.length) req.tools = tools
  return req
}
