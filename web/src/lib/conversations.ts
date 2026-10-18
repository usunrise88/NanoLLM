import { useSyncExternalStore } from 'react'

import type { Dialect } from './api/chat'
import type { ChatResult, Message } from './api/types'
import { randomId } from './id'

/**
 * Conversations, kept in this browser.
 *
 * The server keeps no chat history (SPEC §6: it is stateless per request), so
 * the transcript lives here. What the server does keep is the prompt cache,
 * and `sessionId` is how a conversation finds its slot again: every request
 * of one conversation carries the same session_id, so the next turn reuses
 * the prefix instead of prefilling the whole history again.
 */
const STORAGE_KEY = 'nanollm.conversations'
const LIMIT = 50

export interface ChatMessage extends Message {
  id: string
  /** What the server said about the turn: stats, usage, warnings. Assistant turns only. */
  result?: Omit<ChatResult, 'message'>
  dialect?: Dialect
  /** The turn was cut short by the stop button. */
  stopped?: boolean
  error?: string
}

export interface Conversation {
  id: string
  sessionId: string
  title: string
  model: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
}

let items: Conversation[] = load()
const listeners = new Set<() => void>()

function load(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as Conversation[]) : []
  } catch {
    return []
  }
}

function emit(): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items))
  } catch {
    // A full or blocked storage loses history, never the page.
  }
  for (const l of listeners) l()
}

export function newConversation(model: string): Conversation {
  const now = Date.now()
  const c: Conversation = {
    id: randomId('c_'),
    sessionId: randomId('ui-'),
    title: '',
    model,
    createdAt: now,
    updatedAt: now,
    messages: [],
  }
  items = [c, ...items].slice(0, LIMIT)
  emit()
  return c
}

/** A title from the first user line, short enough for a sidebar. */
export function titleOf(text: string): string {
  const line = text.replace(/\[\[fake:[^\]]*\]\]/g, '').trim().split('\n')[0] ?? ''
  return line.length > 48 ? `${line.slice(0, 47)}…` : line
}

export function updateConversation(id: string, fn: (c: Conversation) => Conversation): void {
  items = items.map((c) => (c.id === id ? { ...fn(c), updatedAt: Date.now() } : c))
  emit()
}

export function deleteConversation(id: string): void {
  items = items.filter((c) => c.id !== id)
  emit()
}

export function clearConversations(): void {
  items = []
  emit()
}

export function useConversations(): Conversation[] {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb)
      return () => listeners.delete(cb)
    },
    () => items,
    () => items,
  )
}
