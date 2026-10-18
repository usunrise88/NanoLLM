import { useSyncExternalStore } from 'react'

import type { Dialect } from './api/chat'
import type { ResponseFormatType } from './api/types'

/**
 * Chat settings that belong to this browser rather than to the server.
 *
 * Empty strings mean "not set": the request then leaves the field out and the
 * server applies the model card's recommended value, which is not the same as
 * zero (temperature 0 is greedy decoding).
 */
const STORAGE_KEY = 'nanollm.chat'

export interface ChatPrefs {
  dialect: Dialect
  thinking: boolean
  system: string
  temperature: string
  topP: string
  topK: string
  maxTokens: string
  maxReasoningTokens: string
  seed: string
  stop: string
  device: string
  logprobs: boolean
  topLogprobs: string
  responseFormat: ResponseFormatType
  /** JSON Schema text, for responseFormat json_schema. */
  schema: string
  /** A JSON array of tools, {name, description?, parameters?}. */
  tools: string
  cachePrompt: boolean
}

export const DEFAULT_PREFS: ChatPrefs = {
  dialect: 'native',
  thinking: false,
  system: '',
  temperature: '',
  topP: '',
  topK: '',
  maxTokens: '',
  maxReasoningTokens: '',
  seed: '',
  stop: '',
  device: '',
  logprobs: false,
  topLogprobs: '',
  responseFormat: 'text',
  schema: '',
  tools: '',
  cachePrompt: true,
}

let prefs: ChatPrefs = load()
const listeners = new Set<() => void>()

function load(): ChatPrefs {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? { ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<ChatPrefs>) } : DEFAULT_PREFS
  } catch {
    return DEFAULT_PREFS
  }
}

export function setPrefs(next: Partial<ChatPrefs>): void {
  prefs = { ...prefs, ...next }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs))
  } catch {
    // A preference that lasts only this session still works.
  }
  for (const l of listeners) l()
}

export function resetPrefs(): void {
  setPrefs({ ...DEFAULT_PREFS, dialect: prefs.dialect, thinking: prefs.thinking })
}

export function usePrefs(): ChatPrefs {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb)
      return () => listeners.delete(cb)
    },
    () => prefs,
    () => prefs,
  )
}
