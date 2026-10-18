import { useState } from 'react'
import { NavArrowDown, NavArrowRight, Tools } from 'iconoir-react'

import { Badge } from '@/components/ui'
import type { Turn } from '@/lib/api/chat'
import type { Logprob, Stats, ToolCall } from '@/lib/api/types'
import { cn } from '@/lib/cn'
import type { ChatMessage } from '@/lib/conversations'
import { ms, percent, tokS } from '@/lib/format'
import { currentLanguage, useT } from '@/lib/i18n'

/**
 * One turn of a conversation.
 *
 * An assistant turn is drawn from the same parts whether it is still
 * streaming (a Turn) or finished (a ChatMessage with the server's result):
 * the reasoning block, the answer, the tool calls, and the stats footer once
 * there is something to put in it.
 */
export function MessageView({ message }: { message: ChatMessage }) {
  if (message.role === 'user') return <UserBubble text={message.content} />
  if (message.role === 'tool') return <ToolResult message={message} />
  if (message.role === 'system') return null
  return (
    <AssistantBody
      content={message.content}
      reasoning={message.reasoning_content ?? ''}
      toolCalls={message.tool_calls ?? []}
      logprobs={message.result?.logprobs ?? []}
      streaming={false}
      message={message}
    />
  )
}

export function StreamingView({ turn }: { turn: Turn }) {
  return (
    <AssistantBody
      content={turn.content}
      reasoning={turn.reasoning}
      toolCalls={turn.toolCalls}
      logprobs={turn.logprobs}
      streaming
    />
  )
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end" data-role="user">
      <div className="max-w-[80%] rounded-[var(--radius-lg)] bg-[var(--accent-bg)] px-3 py-2 text-[14px] whitespace-pre-wrap break-words">
        {text}
      </div>
    </div>
  )
}

function ToolResult({ message }: { message: ChatMessage }) {
  const t = useT()
  return (
    <div className="flex justify-end" data-role="tool">
      <div className="max-w-[80%] rounded-[var(--radius-lg)] border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-3 py-2">
        <p className="text-[11px] text-[var(--text-muted)]">
          {t('chat.toolResult')}
          {message.name ? ` · ${message.name}` : ''}
        </p>
        <pre className="mt-1 font-[var(--font-mono)] text-[12px] whitespace-pre-wrap break-words">{message.content}</pre>
      </div>
    </div>
  )
}

function AssistantBody({
  content,
  reasoning,
  toolCalls,
  logprobs,
  streaming,
  message,
}: {
  content: string
  reasoning: string
  toolCalls: ToolCall[]
  logprobs: Logprob[]
  streaming: boolean
  message?: ChatMessage
}) {
  const t = useT()
  const thinking = streaming && reasoning !== '' && content === '' && toolCalls.length === 0
  return (
    <div className="flex flex-col gap-2" data-role="assistant" data-streaming={streaming ? '' : undefined}>
      {reasoning && <ReasoningBlock text={reasoning} live={thinking} />}

      {(content || (streaming && !reasoning && toolCalls.length === 0)) && (
        <div className="text-[14px] leading-6 whitespace-pre-wrap break-words" data-content="">
          {content}
          {streaming && <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 bg-[var(--text-muted)]" data-caret="" />}
        </div>
      )}

      {toolCalls.map((c, i) => (
        <ToolCallCard key={c.id || i} call={c} />
      ))}

      {logprobs.length > 0 && !streaming && <LogprobsView logprobs={logprobs} />}

      {message?.error && (
        <p role="alert" className="text-[13px] text-[var(--danger-text)]">
          {message.error}
        </p>
      )}
      {message?.stopped && <span className="text-[12px] text-[var(--text-muted)]">{t('chat.stopped')}</span>}
      {message?.result && <StatsFooter message={message} />}
    </div>
  )
}

/**
 * The thinking text, collapsed once the answer starts.
 *
 * Open while the model is still thinking — otherwise the screen shows nothing
 * for seconds — and folded as soon as the answer arrives, which is what the
 * person came for. Their own click wins over both.
 */
function ReasoningBlock({ text, live }: { text: string; live: boolean }) {
  const t = useT()
  const [choice, setChoice] = useState<boolean>()
  const open = choice ?? live
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--bg-surface)]" data-reasoning="">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setChoice(!open)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-[12px] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
      >
        {open ? <NavArrowDown width={12} height={12} /> : <NavArrowRight width={12} height={12} />}
        {live ? t('chat.thinking') : t('chat.reasoning')}
      </button>
      {open && (
        <div className="border-t border-[var(--border-subtle)] px-2.5 py-2 text-[13px] whitespace-pre-wrap break-words text-[var(--text-secondary)]">
          {text}
        </div>
      )}
    </div>
  )
}

function prettyArgs(args: string): string {
  try {
    return JSON.stringify(JSON.parse(args), null, 2)
  } catch {
    return args
  }
}

function ToolCallCard({ call }: { call: ToolCall }) {
  const t = useT()
  return (
    <div
      className="rounded-[var(--radius-md)] border border-[var(--accent-border)] bg-[var(--bg-surface)] px-2.5 py-2"
      data-tool-call={call.name}
    >
      <div className="flex items-center gap-1.5 text-[12px]">
        <Tools width={13} height={13} />
        <span className="text-[var(--text-muted)]">{t('chat.toolCall')}</span>
        <span className="font-[var(--font-mono)] font-medium">{call.name}</span>
        {call.id && <span className="ml-auto font-[var(--font-mono)] text-[11px] text-[var(--text-muted)]">{call.id}</span>}
      </div>
      <pre className="mt-1.5 overflow-x-auto font-[var(--font-mono)] text-[12px]">{prettyArgs(call.arguments)}</pre>
    </div>
  )
}

/** Tokens tinted by how sure the model was; hover for the number and the alternatives. */
function LogprobsView({ logprobs }: { logprobs: Logprob[] }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  return (
    <div data-logprobs="">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[12px] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
      >
        {open ? <NavArrowDown width={12} height={12} /> : <NavArrowRight width={12} height={12} />}
        {t('chat.logprobs', { count: logprobs.length })}
      </button>
      {open && (
        <div className="mt-1 flex flex-wrap gap-0.5 font-[var(--font-mono)] text-[12px]">
          {logprobs.map((lp, i) => {
            const p = Math.exp(lp.logprob)
            const alts = (lp.top_logprobs ?? []).map((a) => `${JSON.stringify(a.token)} ${Math.exp(a.logprob).toFixed(3)}`)
            return (
              <span
                key={i}
                title={[`${JSON.stringify(lp.token)} p=${p.toFixed(3)} (${lp.logprob.toFixed(3)})`, ...alts].join('\n')}
                className={cn(
                  'rounded-[var(--radius-sm)] px-0.5 whitespace-pre',
                  p >= 0.9 ? 'bg-[var(--success-bg)]' : p >= 0.5 ? 'bg-[var(--warning-bg)]' : 'bg-[var(--danger-bg)]',
                )}
              >
                {lp.token}
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}

/**
 * What the server measured for this turn (SPEC §12): time to first token,
 * decode and prefill speed, how much of the prompt came from the cache, where
 * it ran, and how many drafted tokens the target accepted.
 */
function StatsFooter({ message }: { message: ChatMessage }) {
  const t = useT()
  const r = message.result
  if (!r) return null
  const s: Stats = r.stats
  const locale = currentLanguage()
  const device = [s.backend, s.device].filter(Boolean).join(':')
  return (
    <div className="flex flex-col gap-1">
      <dl className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[var(--text-muted)] tabular-nums" data-stats="">
        <Stat name="ttft" label={t('stats.ttft')} value={ms(s.ttft_ms)} />
        <Stat name="decode" label={t('stats.decode')} value={tokS(s.decode_tok_s, locale)} />
        <Stat name="prefill" label={t('stats.prefill')} value={tokS(s.prefill_tok_s, locale)} />
        <Stat name="cached" label={t('stats.cached')} value={`${s.cached_tokens} / ${s.prompt_tokens}`} />
        <Stat
          name="tokens"
          label={t('stats.tokens')}
          value={s.reasoning_tokens ? `${s.completion_tokens} (${t('stats.reasoningTokens', { count: s.reasoning_tokens })})` : String(s.completion_tokens)}
        />
        {device && <Stat name="device" label={t('stats.device')} value={device} />}
        {s.preset && <Stat name="preset" label={t('stats.preset')} value={s.mode ? `${s.preset} / ${s.mode}` : s.preset} />}
        {s.draft_acceptance !== undefined && (
          <Stat name="draft" label={t('stats.draft')} value={percent(s.draft_acceptance)} />
        )}
        {r.finish_reason !== 'stop' && <Stat name="finish" label={t('stats.finish')} value={r.finish_reason} />}
        {message.dialect === 'openai' && <Stat name="dialect" label={t('stats.dialect')} value="OpenAI" />}
      </dl>
      {r.warnings?.map((w) => (
        <span key={w.code + w.message}>
          <Badge tone="warning">{w.message || w.code}</Badge>
        </span>
      ))}
    </div>
  )
}

function Stat({ name, label, value }: { name: string; label: string; value: string }) {
  return (
    <div className="flex gap-1" data-stat={name}>
      <dt>{label}</dt>
      <dd className="text-[var(--text-secondary)]">{value}</dd>
    </div>
  )
}
