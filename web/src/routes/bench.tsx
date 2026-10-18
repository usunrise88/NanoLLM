import { useEffect, useRef, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { Download, Play, Xmark } from 'iconoir-react'

import { JsonBlock, saveText } from '@/components/JsonView'
import { Card, Inline, Page, Section, Stack } from '@/components/layout'
import { Badge, Button, Detail, EmptyState, ErrorState, Field, Progress, Select, Switch, type Tone } from '@/components/ui'
import { benchReducer, CONSENT_VERSION, initialBench, submissionJSON, type BenchState } from '@/lib/api/bench'
import { benchEventsPath, errorText, isNotYet, startBench, submitBench } from '@/lib/api/client'
import { useModels } from '@/lib/api/hooks'
import { stream } from '@/lib/api/sse'
import type { BenchRun, BenchSuite, BenchSubmitted } from '@/lib/api/types'
import { bytes, ms, percent, seconds, tokS } from '@/lib/format'
import { currentLanguage, useT } from '@/lib/i18n'
import type { PageMeta } from '@/lib/page'
import { toast } from '@/lib/toast'

export const pageMeta: PageMeta = {
  titleKey: 'bench.title',
  descriptionKey: 'bench.description',
}

export const Route = createFileRoute('/bench')({
  component: BenchPage,
  staticData: { pageMeta },
})

type Phase = 'idle' | 'starting' | 'running' | 'unavailable' | 'failed'

function runTone(s: string): Tone {
  switch (s) {
    case 'ok':
      return 'success'
    case 'failed':
      return 'danger'
    case 'running':
      return 'accent'
    case 'skipped':
    case 'refused':
      return 'warning'
    default:
      return 'neutral'
  }
}

function BenchPage() {
  const t = useT()
  const models = useModels()
  const [model, setModel] = useState('')
  const [suite, setSuite] = useState<BenchSuite>('standard')
  const [quick, setQuick] = useState(true)
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState('')
  const [state, setState] = useState<BenchState>(initialBench())
  const abort = useRef<AbortController>(undefined)

  useEffect(() => () => abort.current?.abort(), [])

  async function run() {
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    setPhase('starting')
    setError('')
    setState(initialBench())
    let started
    try {
      started = await startBench({ suite, quick, ...(model ? { model } : {}) })
    } catch (err) {
      if (isNotYet(err)) {
        setPhase('unavailable')
      } else {
        setPhase('failed')
        setError(errorText(err))
      }
      return
    }
    let current = initialBench(started)
    setState(current)
    setPhase('running')
    await stream<unknown>(benchEventsPath(started.id), {
      signal: controller.signal,
      onEvent: (event, payload) => {
        current = benchReducer(current, event, payload)
        setState(current)
      },
      onError: (err) => {
        current = { ...current, error: errorText(err), finished: true }
        setState(current)
      },
    })
    setState({ ...current, finished: true })
    setPhase('idle')
  }

  const running = phase === 'starting' || phase === 'running'

  return (
    <Page>
      <Section title={t('bench.run')} description={t('bench.runHint')}>
        <Card>
          <Stack gap={4}>
            <Inline gap={4} align="start">
              <span className="w-56">
                <Field label={t('bench.model')}>
                  <Select
                    value={model}
                    onChange={setModel}
                    disabled={running}
                    options={[{ value: '', label: t('bench.defaultModel') }, ...(models.data ?? []).map((m) => ({ value: m.id, label: m.id }))]}
                  />
                </Field>
              </span>
              <span className="w-56">
                <Field label={t('bench.suite')} description={suite === 'standard' ? t('bench.standardHint') : t('bench.devHint')}>
                  <Select
                    value={suite}
                    onChange={(v) => setSuite(v as BenchSuite)}
                    disabled={running}
                    options={[
                      { value: 'standard', label: t('bench.standard') },
                      { value: 'dev', label: t('bench.dev') },
                    ]}
                  />
                </Field>
              </span>
              <span className="w-56">
                <Field label={t('bench.scope')} description={quick ? t('bench.quickHint') : t('bench.fullHint')}>
                  <Select
                    value={quick ? 'quick' : 'full'}
                    onChange={(v) => setQuick(v === 'quick')}
                    disabled={running}
                    options={[
                      { value: 'quick', label: t('bench.quick') },
                      { value: 'full', label: t('bench.full') },
                    ]}
                  />
                </Field>
              </span>
            </Inline>
            <Inline gap={2}>
              {running ? (
                <Button icon={<Xmark width={13} height={13} />} onClick={() => abort.current?.abort()}>
                  {t('bench.stopWatching')}
                </Button>
              ) : (
                <Button variant="primary" icon={<Play width={13} height={13} />} onClick={() => void run()}>
                  {t('bench.start')}
                </Button>
              )}
              <span className="text-[12px] text-[var(--text-muted)]">{t('bench.privacy')}</span>
            </Inline>
          </Stack>
        </Card>
      </Section>

      {phase === 'unavailable' && (
        <Section>
          <Card>
            <EmptyState title={t('bench.notYet')} description={t('bench.notYetHint')} />
          </Card>
        </Section>
      )}
      {phase === 'failed' && (
        <Section>
          <ErrorState title={t('bench.failed')} detail={error} />
        </Section>
      )}

      {state.presets.length > 0 && <ProgressSection state={state} />}
      {state.error && (
        <Section>
          <ErrorState title={t('bench.failed')} detail={state.error} />
        </Section>
      )}
      {state.result && <ResultsSection state={state} suite={suite} />}
    </Page>
  )
}

function ProgressSection({ state }: { state: BenchState }) {
  const t = useT()
  return (
    <Section
      title={t('bench.progress')}
      {...(state.estimateS !== undefined ? { description: t('bench.estimate', { time: seconds(state.estimateS) }) } : {})}
    >
      <Card>
        <Stack gap={3}>
          {state.presets.map((p) => (
            <Stack key={p.preset} gap={1}>
              <Inline gap={2}>
                <span className="font-[var(--font-mono)] text-[12px]" data-preset={p.preset}>
                  {p.preset}
                </span>
                <Badge tone={runTone(p.status)}>{t(`benchStatus.${p.status}`)}</Badge>
                {p.scenario && p.status === 'running' && <span className="text-[11px] text-[var(--text-muted)]">{p.scenario}</span>}
                {p.run?.reason && <span className="text-[11px] text-[var(--text-muted)]">{p.run.reason}</span>}
              </Inline>
              {(p.status === 'running' || p.status === 'pending') && <Progress percent={p.percent ?? 0} label={p.preset} />}
            </Stack>
          ))}
        </Stack>
      </Card>
    </Section>
  )
}

function ResultsSection({ state, suite }: { state: BenchState; suite: BenchSuite }) {
  const t = useT()
  const result = state.result
  const [anonymous, setAnonymous] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState<BenchSubmitted>()
  const [declined, setDeclined] = useState(false)
  if (!result) return null
  const json = submissionJSON(result)
  const canSubmit = suite === 'standard' && result.suite?.id !== 'dev' && state.id !== undefined

  async function submit() {
    if (!state.id) return
    setSubmitting(true)
    try {
      setSubmitted(await submitBench(state.id, CONSENT_VERSION, anonymous))
    } catch (err) {
      toast.error(isNotYet(err) ? t('bench.submitNotYet') : t('bench.submitFailed'), { description: errorText(err) })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <>
      <Section title={t('bench.results')} description={t('bench.resultsHint', { duration: seconds(result.duration_s), model: result.model?.id ?? '' })}>
        <Card className="overflow-x-auto">
          <ResultsTable runs={result.runs ?? []} />
        </Card>
      </Section>

      <Section
        title={t('bench.consentTitle')}
        actions={
          <Button size="sm" icon={<Download width={13} height={13} />} onClick={() => saveText(`nanollm-bench-${result.date}.json`, json)}>
            {t('bench.save')}
          </Button>
        }
      >
        <Stack gap={3}>
          {canSubmit ? (
            <Card>
              <Stack gap={3}>
                <p className="text-[13px]">{t('bench.consentText')}</p>
                <Detail label={t('bench.consentVersion')}>{CONSENT_VERSION}</Detail>
                <Switch checked={anonymous} onChange={setAnonymous} label={t('bench.anonymous')} description={t('bench.anonymousHint')} />
                {submitted ? (
                  <p className="text-[13px] text-[var(--success-text)]" role="status">
                    {t('bench.submitted', { id: submitted.id })}
                    {submitted.url ? ` ${submitted.url}` : ''}
                  </p>
                ) : declined ? (
                  <p className="text-[13px] text-[var(--text-secondary)]">{t('bench.declined')}</p>
                ) : (
                  <Inline gap={2}>
                    <Button variant="primary" busy={submitting} onClick={() => void submit()}>
                      {t('bench.submit')}
                    </Button>
                    <Button onClick={() => setDeclined(true)}>{t('bench.decline')}</Button>
                  </Inline>
                )}
              </Stack>
            </Card>
          ) : (
            <Card>
              <p className="text-[13px] text-[var(--text-secondary)]">{t('bench.devNotSubmitted')}</p>
            </Card>
          )}
          <JsonBlock value={json} label={t('bench.exactJson')} />
        </Stack>
      </Section>
    </>
  )
}

function ResultsTable({ runs }: { runs: BenchRun[] }) {
  const t = useT()
  const locale = currentLanguage()
  const th = 'px-2 py-1.5 text-left font-medium text-[var(--text-muted)] whitespace-nowrap'
  const td = 'px-2 py-1.5 whitespace-nowrap tabular-nums'
  return (
    <table className="w-full border-collapse text-[12px]" data-results="">
      <thead>
        <tr className="border-b border-[var(--border-subtle)]">
          <th className={th}>{t('bench.preset')}</th>
          <th className={th}>{t('bench.status')}</th>
          <th className={th}>{t('bench.load')}</th>
          <th className={th}>pp512</th>
          <th className={th}>tg128 d0</th>
          <th className={th}>tg128 dN</th>
          <th className={th}>TTFT</th>
          <th className={th}>chat</th>
          <th className={th}>MTP</th>
          <th className={th}>{t('bench.peakRss')}</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => {
          const m = r.metrics ?? {}
          return (
            <tr key={r.preset} className="border-b border-[var(--border-subtle)] last:border-0" data-run={r.preset}>
              <td className={`${td} font-[var(--font-mono)]`}>
                {r.preset}
                {r.same_as ? ` = ${r.same_as}` : ''}
              </td>
              <td className={td}>
                <Badge tone={runTone(r.status)}>{t(`benchStatus.${r.status}`)}</Badge>
              </td>
              <td className={td}>{m.load_ms ? ms(m.load_ms) : '—'}</td>
              <td className={td}>{tokS(m.pp_tok_s, locale)}</td>
              <td className={td}>{tokS(m.tg_tok_s?.d0, locale)}</td>
              <td className={td}>{tokS(m.tg_tok_s?.dN, locale)}</td>
              <td className={td}>{m.ttft_ms ? ms(m.ttft_ms) : '—'}</td>
              <td className={td}>{tokS(m.chat_tok_s, locale)}</td>
              <td className={td}>{m.mtp ? `${percent(m.mtp.acceptance)} · ×${m.mtp.speedup.toFixed(2)}` : '—'}</td>
              <td className={td}>{m.peak_rss ? bytes(m.peak_rss) : '—'}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
