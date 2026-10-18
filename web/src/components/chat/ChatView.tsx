import { useEffect, useMemo, useRef, useState } from 'react'
import { ChatBubbleEmpty, Plus, Send, Settings, Square, Trash } from 'iconoir-react'

import { Badge, Button, EmptyState, IconButton, Select, Switch, Textarea } from '@/components/ui'
import { emptyTurn, streamChat, type Turn } from '@/lib/api/chat'
import { ApiError, errorText, tokenize } from '@/lib/api/client'
import { useModels } from '@/lib/api/hooks'
import type { ModelInfo, ToolCall } from '@/lib/api/types'
import { buildRequest, PrefsError, wireMessage } from '@/lib/chat-request'
import { cn } from '@/lib/cn'
import {
  deleteConversation,
  newConversation,
  titleOf,
  updateConversation,
  useConversations,
  type ChatMessage,
  type Conversation,
} from '@/lib/conversations'
import { randomId } from '@/lib/id'
import { useT } from '@/lib/i18n'
import { setPrefs, usePrefs, type ChatPrefs } from '@/lib/prefs'
import { toast } from '@/lib/toast'

import { ChatOptions } from './ChatOptions'
import { MessageView, StreamingView } from './MessageView'

interface Live {
  conversationId: string
  turn: Turn
  controller: AbortController
}

/** The model a new conversation starts with: the server's default, else the first loaded one. */
function defaultModel(models: ModelInfo[] | undefined): string {
  if (!models?.length) return ''
  return (models.find((m) => m.default) ?? models.find((m) => m.loaded?.length) ?? models[0])?.id ?? ''
}

export function ChatView() {
  const t = useT()
  const prefs = usePrefs()
  const models = useModels()
  const conversations = useConversations()
  const [activeId, setActiveId] = useState<string | undefined>(() => conversations[0]?.id)
  const [draftModel, setDraftModel] = useState('')
  const [draft, setDraft] = useState('')
  const [live, setLive] = useState<Live>()
  const liveTurn = useRef<Turn>(emptyTurn())
  const [optionsOpen, setOptionsOpen] = useState(false)
  const [invalid, setInvalid] = useState<{ field: keyof ChatPrefs; message: string }>()
  const bottom = useRef<HTMLDivElement>(null)

  const active = conversations.find((c) => c.id === activeId)
  const model = active?.model || draftModel || defaultModel(models.data)
  const modelInfo = models.data?.find((m) => m.id === model)
  const streaming = live !== undefined && live.conversationId === active?.id

  // Leaving the page stops the generation, which frees the slot on the server.
  const [controllers] = useState(() => new Set<AbortController>())
  useEffect(() => () => controllers.forEach((c) => c.abort()), [controllers])
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [active?.messages.length, live?.turn])

  const pendingTools = useMemo(() => pendingToolCalls(active), [active])

  async function run(conversation: Conversation, history: ChatMessage[]) {
    let req
    try {
      req = buildRequest({ model: conversation.model, sessionId: conversation.sessionId, history, prefs })
      setInvalid(undefined)
    } catch (err) {
      if (err instanceof PrefsError) {
        setInvalid({ field: err.field, message: err.message })
        setOptionsOpen(true)
        return
      }
      throw err
    }

    const controller = new AbortController()
    controllers.add(controller)
    liveTurn.current = emptyTurn()
    setLive({ conversationId: conversation.id, turn: liveTurn.current, controller })
    const id = randomId('m_')
    let message: ChatMessage
    try {
      const result = await streamChat(req, {
        dialect: prefs.dialect,
        signal: controller.signal,
        onTurn: (turn) => {
          liveTurn.current = turn
          setLive((l) => (l && l.controller === controller ? { ...l, turn } : l))
        },
      })
      if (result) {
        const { message: m, ...rest } = result
        message = { ...wireMessage(m), role: 'assistant', id, result: rest, dialect: prefs.dialect }
      } else {
        message = partial(id, liveTurn.current, { stopped: true })
      }
    } catch (err) {
      message = controller.signal.aborted
        ? partial(id, liveTurn.current, { stopped: true })
        : partial(id, liveTurn.current, { error: describe(err, t('chat.failed')) })
    }
    updateConversation(conversation.id, (c) => ({ ...c, messages: [...c.messages, message] }))
    controllers.delete(controller)
    setLive((l) => (l?.controller === controller ? undefined : l))
  }

  function send() {
    const text = draft.trim()
    if (!text || live) return
    if (!model) {
      toast.error(t('chat.noModel'))
      return
    }
    const conversation = active ?? newConversation(model)
    if (!active) setActiveId(conversation.id)
    const user: ChatMessage = { id: randomId('m_'), role: 'user', content: text }
    const history = [...conversation.messages, user]
    updateConversation(conversation.id, (c) => ({
      ...c,
      title: c.title || titleOf(text),
      messages: history,
    }))
    setDraft('')
    void run({ ...conversation, messages: history }, history)
  }

  function sendToolResults(results: Record<string, string>) {
    if (!active || live) return
    const replies: ChatMessage[] = pendingTools.map((c) => ({
      id: randomId('m_'),
      role: 'tool',
      content: results[c.id] ?? '',
      tool_call_id: c.id,
      name: c.name,
    }))
    const history = [...active.messages, ...replies]
    updateConversation(active.id, (c) => ({ ...c, messages: history }))
    void run({ ...active, messages: history }, history)
  }

  function changeModel(id: string) {
    if (active) updateConversation(active.id, (c) => ({ ...c, model: id }))
    else setDraftModel(id)
  }

  return (
    <div className="grid min-h-[60vh] grid-cols-1 gap-4 md:grid-cols-[14rem_1fr]">
      <aside className="flex flex-col gap-1" aria-label={t('chat.conversations')}>
        <Button
          size="sm"
          icon={<Plus width={13} height={13} />}
          onClick={() => {
            setActiveId(undefined)
            setDraft('')
          }}
          disabled={streaming}
        >
          {t('chat.new')}
        </Button>
        <ul className="mt-2 flex flex-col gap-0.5" data-conversations="">
          {conversations.map((c) => (
            <li key={c.id} className="group flex items-center">
              <button
                type="button"
                onClick={() => setActiveId(c.id)}
                disabled={live !== undefined}
                className={cn(
                  'min-w-0 flex-1 truncate rounded-[var(--radius-md)] px-2 py-1 text-left text-[13px] hover:bg-[var(--bg-hover)]',
                  c.id === activeId ? 'bg-[var(--bg-raised)] font-medium' : 'text-[var(--text-secondary)]',
                )}
              >
                {c.title || t('chat.untitled')}
              </button>
              <IconButton
                label={t('common.delete')}
                className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                disabled={live?.conversationId === c.id}
                onClick={() => {
                  deleteConversation(c.id)
                  if (c.id === activeId) setActiveId(undefined)
                }}
              >
                <Trash width={13} height={13} />
              </IconButton>
            </li>
          ))}
        </ul>
      </aside>

      <section className="flex min-w-0 flex-col gap-3">
        <div className="flex min-h-[40vh] flex-col gap-4 rounded-[var(--radius-lg)] border border-[var(--border-subtle)] bg-[var(--bg-canvas)] p-4" data-messages="">
          {!active?.messages.length && !streaming && (
            <EmptyState title={t('chat.emptyTitle')} description={t('chat.emptyHint')} />
          )}
          {active?.messages.map((m) => <MessageView key={m.id} message={m} />)}
          {streaming && live && <StreamingView turn={live.turn} />}
          {!streaming && pendingTools.length > 0 && <ToolReplyForm calls={pendingTools} onSend={sendToolResults} />}
          <div ref={bottom} />
        </div>

        <Composer
          draft={draft}
          setDraft={setDraft}
          onSend={send}
          onStop={() => live?.controller.abort()}
          streaming={live !== undefined}
          model={model}
          models={models.data ?? []}
          modelInfo={modelInfo}
          onModel={changeModel}
          prefs={prefs}
          onOptions={() => setOptionsOpen(true)}
          history={active?.messages ?? []}
        />
        {active && (
          <p className="text-[11px] text-[var(--text-muted)]" data-session={active.sessionId}>
            {t('chat.session', { id: active.sessionId })}
          </p>
        )}
      </section>

      <ChatOptions open={optionsOpen} onOpenChange={setOptionsOpen} invalid={invalid} />
    </div>
  )
}

function partial(id: string, turn: Turn, extra: { stopped?: boolean; error?: string }): ChatMessage {
  const m: ChatMessage = { id, role: 'assistant', content: turn.content, ...extra }
  if (turn.reasoning) m.reasoning_content = turn.reasoning
  if (turn.toolCalls.length) m.tool_calls = turn.toolCalls
  return m
}

function describe(err: unknown, fallback: string): string {
  if (err instanceof ApiError) return `${fallback}: ${err.message} (${err.code})`
  return `${fallback}: ${errorText(err)}`
}

/** Tool calls of the last assistant turn that no tool message has answered yet. */
function pendingToolCalls(c: Conversation | undefined): ToolCall[] {
  const last = c?.messages.at(-1)
  if (!last || last.role !== 'assistant' || !last.tool_calls?.length || last.stopped || last.error) return []
  return last.tool_calls
}

function ToolReplyForm({ calls, onSend }: { calls: ToolCall[]; onSend: (r: Record<string, string>) => void }) {
  const t = useT()
  const [values, setValues] = useState<Record<string, string>>({})
  return (
    <div className="flex flex-col gap-2 rounded-[var(--radius-md)] border border-dashed border-[var(--border-default)] p-3" data-tool-reply="">
      <p className="text-[12px] text-[var(--text-secondary)]">{t('chat.toolReplyHint')}</p>
      {calls.map((c) => (
        <label key={c.id} className="flex flex-col gap-1">
          <span className="font-[var(--font-mono)] text-[12px]">{c.name}</span>
          <Textarea
            rows={2}
            aria-label={t('chat.toolResultFor', { name: c.name })}
            value={values[c.id] ?? ''}
            onChange={(e) => setValues({ ...values, [c.id]: e.target.value })}
            placeholder='{"temperature": 21}'
          />
        </label>
      ))}
      <div>
        <Button size="sm" variant="primary" onClick={() => onSend(values)}>
          {t('chat.sendToolResults')}
        </Button>
      </div>
    </div>
  )
}

function Composer({
  draft,
  setDraft,
  onSend,
  onStop,
  streaming,
  model,
  models,
  modelInfo,
  onModel,
  prefs,
  onOptions,
  history,
}: {
  draft: string
  setDraft: (s: string) => void
  onSend: () => void
  onStop: () => void
  streaming: boolean
  model: string
  models: ModelInfo[]
  modelInfo: ModelInfo | undefined
  onModel: (id: string) => void
  prefs: ChatPrefs
  onOptions: () => void
  history: ChatMessage[]
}) {
  const t = useT()
  const tokens = usePromptTokens(draft, history, model, modelInfo, prefs)
  const resident = (modelInfo?.loaded?.length ?? 0) > 0
  const onDisk = modelInfo?.disk === 'complete'

  return (
    <div className="flex flex-col gap-2 rounded-[var(--radius-lg)] border border-[var(--border-default)] bg-[var(--bg-surface)] p-3">
      <Textarea
        rows={3}
        value={draft}
        aria-label={t('chat.message')}
        placeholder={t('chat.placeholder')}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            onSend()
          }
        }}
      />
      <div className="flex flex-wrap items-center gap-3">
        <span className="w-56">
          <Select
            aria-label={t('chat.model')}
            value={model}
            onChange={onModel}
            disabled={streaming}
            options={
              models.length
                ? models.map((m) => ({
                    value: m.id,
                    label: `${m.id}${m.loaded?.length ? ' ●' : m.disk === 'complete' ? '' : ` (${t(`disk.${m.disk}`)})`}`,
                  }))
                : [{ value: '', label: t('chat.noModels') }]
            }
          />
        </span>
        <Switch checked={prefs.thinking} onChange={(v) => setPrefs({ thinking: v })} label={t('chat.thinkingToggle')} />
        {modelInfo && !resident && !onDisk && <Badge tone="warning">{t('chat.notOnDisk')}</Badge>}
        {modelInfo && !resident && onDisk && <Badge>{t('chat.loadsOnFirst')}</Badge>}
        {prefs.dialect === 'openai' && <Badge tone="accent">OpenAI</Badge>}
        {tokens !== undefined && (
          <span className="text-[11px] text-[var(--text-muted)] tabular-nums" data-prompt-tokens={tokens}>
            {t('chat.promptTokens', { count: tokens })}
          </span>
        )}
        <span className="ml-auto flex items-center gap-1">
          <IconButton label={t('chat.options')} onClick={onOptions}>
            <Settings width={15} height={15} />
          </IconButton>
          {streaming ? (
            <Button variant="danger" icon={<Square width={13} height={13} />} onClick={onStop}>
              {t('chat.stop')}
            </Button>
          ) : (
            <Button variant="primary" icon={<Send width={13} height={13} />} onClick={onSend} disabled={draft.trim() === ''}>
              {t('chat.send')}
            </Button>
          )}
        </span>
      </div>
      {!models.length && (
        <p className="flex items-center gap-1 text-[12px] text-[var(--text-muted)]">
          <ChatBubbleEmpty width={13} height={13} /> {t('chat.noModelsHint')}
        </p>
      )}
    </div>
  )
}

/**
 * The prompt size the next send would have, from /api/v1/tokenize with the
 * chat template applied. Only asked of a resident model: counting tokens must
 * never be what makes the server load four gigabytes.
 */
function usePromptTokens(
  draft: string,
  history: ChatMessage[],
  model: string,
  info: ModelInfo | undefined,
  prefs: ChatPrefs,
): number | undefined {
  const [count, setCount] = useState<number>()
  const resident = (info?.loaded?.length ?? 0) > 0
  useEffect(() => {
    if (!resident || draft.trim() === '') return
    const id = setTimeout(() => {
      const messages = [
        ...(prefs.system.trim() ? [{ role: 'system' as const, content: prefs.system }] : []),
        ...history.filter((m) => !m.error).map(wireMessage),
        { role: 'user' as const, content: draft },
      ]
      tokenize({ model, messages })
        .then((r) => setCount(r.count))
        .catch(() => setCount(undefined))
    }, 400)
    return () => clearTimeout(id)
  }, [draft, history, model, resident, prefs.system])
  return resident && draft.trim() !== '' ? count : undefined
}
