import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { Download, Upload, Xmark } from 'iconoir-react'

import { saveText } from '@/components/JsonView'
import { Card, Inline, Page, Section, Stack } from '@/components/layout'
import {
  Badge,
  Button,
  EmptyState,
  ErrorState,
  Field,
  Progress,
  Select,
  SkeletonRow,
  Textarea,
  type Tone,
} from '@/components/ui'
import { errorText, getJob, isNotYet, validateJsonl } from '@/lib/api/client'
import { useCancelJob, useJob, useJobHistory, useJobStream, useModels, useSubmitJob } from '@/lib/api/hooks'
import { isTerminal, type Job, type JobStatus } from '@/lib/api/types'
import { when } from '@/lib/format'
import { currentLanguage, useT } from '@/lib/i18n'
import type { PageMeta } from '@/lib/page'
import { useDelayedPending } from '@/lib/pending'
import { toast } from '@/lib/toast'

export const pageMeta: PageMeta = {
  titleKey: 'jobs.title',
  descriptionKey: 'jobs.description',
}

export const Route = createFileRoute('/jobs')({
  component: JobsPage,
  staticData: { pageMeta },
})

const STATUSES: JobStatus[] = ['queued', 'running', 'succeeded', 'failed', 'canceled', 'expired']

const EXAMPLE = [
  '{"custom_id":"a","messages":[{"role":"user","content":"Summarise the plot of Hamlet in one line."}]}',
  '{"custom_id":"b","messages":[{"role":"user","content":"Translate to Russian: good morning"}],"max_tokens":64}',
].join('\n')

function statusTone(s: JobStatus): Tone {
  switch (s) {
    case 'succeeded':
      return 'success'
    case 'failed':
      return 'danger'
    case 'running':
      return 'accent'
    case 'canceled':
    case 'expired':
      return 'warning'
    default:
      return 'neutral'
  }
}

function JobsPage() {
  const t = useT()
  const [status, setStatus] = useState('')
  const filter = { ...(status ? { status: status as JobStatus } : {}), limit: 20 }
  const history = useJobHistory(filter)
  const loading = useDelayedPending(history.isLoading)
  const jobs = history.data?.pages.flatMap((p) => p.data ?? []) ?? []
  const notYet = history.isError && isNotYet(history.error)

  return (
    <Page>
      <Section title={t('jobs.submit')} description={t('jobs.submitHint')}>
        <SubmitForm disabled={notYet} />
      </Section>

      <Section
        title={t('jobs.history')}
        actions={
          <span className="w-44">
            <Select
              aria-label={t('jobs.filterStatus')}
              value={status}
              onChange={setStatus}
              options={[{ value: '', label: t('jobs.all') }, ...STATUSES.map((s) => ({ value: s, label: t(`jobStatus.${s}`) }))]}
            />
          </span>
        }
      >
        {notYet && (
          <Card>
            <EmptyState title={t('jobs.notYet')} description={t('jobs.notYetHint')} />
          </Card>
        )}
        {history.isError && !notYet && (
          <ErrorState
            title={t('common.error')}
            detail={errorText(history.error)}
            action={
              <Button size="sm" onClick={() => void history.refetch()}>
                {t('common.retry')}
              </Button>
            }
          />
        )}
        {loading && (
          <Card>
            <SkeletonRow />
            <SkeletonRow />
          </Card>
        )}
        {!history.isLoading && !history.isError && jobs.length === 0 && (
          <Card>
            <EmptyState title={t('jobs.empty')} description={t('jobs.emptyHint')} />
          </Card>
        )}
        <Stack gap={2}>
          {jobs.map((job) => (
            <JobRow key={job.id} job={job} />
          ))}
        </Stack>
        {history.hasNextPage && (
          <Inline>
            <Button size="sm" busy={history.isFetchingNextPage} onClick={() => void history.fetchNextPage()}>
              {t('jobs.more')}
            </Button>
          </Inline>
        )}
      </Section>
    </Page>
  )
}

function SubmitForm({ disabled }: { disabled: boolean }) {
  const t = useT()
  const models = useModels()
  const submit = useSubmitJob()
  const [text, setText] = useState('')
  const [model, setModel] = useState('')
  const check = text.trim() ? validateJsonl(text) : undefined

  async function readFile(file: File) {
    setText(await file.text())
  }

  return (
    <Card>
      <Stack gap={3}>
        <Field
          label={t('jobs.requests')}
          description={check && !check.error ? t('jobs.lines', { count: check.count }) : t('jobs.requestsHint')}
          error={check?.error ? t('jobs.badLine', { line: check.error.line, message: check.error.message }) : undefined}
        >
          <Textarea
            rows={6}
            spellCheck={false}
            className="font-[var(--font-mono)] text-[12px]"
            value={text}
            placeholder={EXAMPLE}
            onChange={(e) => setText(e.target.value)}
          />
        </Field>
        <Inline gap={3}>
          <span className="w-72">
            <Select
              aria-label={t('jobs.model')}
              value={model}
              onChange={setModel}
              options={[{ value: '', label: t('jobs.defaultModel') }, ...(models.data ?? []).map((m) => ({ value: m.id, label: m.id }))]}
            />
          </span>
          <label className="inline-flex cursor-pointer items-center gap-1.5 text-[13px] text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
            <Upload width={13} height={13} />
            {t('jobs.fromFile')}
            <input
              type="file"
              accept=".jsonl,.ndjson,application/x-ndjson,application/jsonl,text/plain"
              className="sr-only"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void readFile(f)
              }}
            />
          </label>
          <Button size="sm" onClick={() => setText(EXAMPLE)}>
            {t('jobs.example')}
          </Button>
          <Button
            size="sm"
            variant="primary"
            busy={submit.isPending}
            disabled={disabled || !check || !!check.error || check.count === 0}
            onClick={() =>
              void submit
                .mutateAsync({ jsonl: text, ...(model ? { model } : {}) })
                .then((job) => {
                  toast.success(t('jobs.submitted', { id: job.id }))
                  setText('')
                })
                .catch((err) =>
                  toast.error(isNotYet(err) ? t('jobs.notYet') : t('jobs.submitFailed'), { description: errorText(err) }),
                )
            }
          >
            {t('jobs.enqueue')}
          </Button>
        </Inline>
      </Stack>
    </Card>
  )
}

function JobRow({ job: listed }: { job: Job }) {
  const t = useT()
  const cancel = useCancelJob()
  const job = useJob(listed.id, listed).data ?? listed
  useJobStream(job)

  const p = job.progress
  async function downloadResults() {
    try {
      const full = await getJob(job.id)
      const lines = (full.results ?? []).map((r) => JSON.stringify(r)).join('\n')
      saveText(`${job.id}.jsonl`, lines + '\n', 'application/x-ndjson')
    } catch (err) {
      toast.error(t('jobs.downloadFailed'), { description: errorText(err) })
    }
  }

  return (
    <Card data-item={job.id}>
      <Stack gap={2}>
        <Inline gap={2}>
          <span className="font-[var(--font-mono)] text-[12px]">{job.id}</span>
          <Badge tone={statusTone(job.status)}>{t(`jobStatus.${job.status}`)}</Badge>
          {job.model && <span className="text-[12px] text-[var(--text-muted)]">{job.model}</span>}
          <span className="ml-auto text-[12px] text-[var(--text-muted)]">{when(job.created_at, currentLanguage())}</span>
        </Inline>
        {p && p.total > 0 && (
          <Stack gap={1}>
            <Progress percent={(p.done / p.total) * 100} label={t('jobs.progress')} />
            <span className="text-[11px] text-[var(--text-muted)] tabular-nums">
              {t('jobs.progressDetail', { done: p.done, total: p.total, failed: p.failed ?? 0 })}
            </span>
          </Stack>
        )}
        {job.error && <p className="text-[12px] text-[var(--danger-text)]">{job.error.message}</p>}
        <Inline gap={2}>
          {!isTerminal(job.status) && (
            <Button
              size="sm"
              icon={<Xmark width={13} height={13} />}
              busy={cancel.isPending}
              onClick={() => void cancel.mutateAsync(job.id).catch((err) => toast.error(t('jobs.cancelFailed'), { description: errorText(err) }))}
            >
              {t('jobs.cancel')}
            </Button>
          )}
          {job.status === 'succeeded' && (
            <Button size="sm" icon={<Download width={13} height={13} />} onClick={() => void downloadResults()}>
              {t('jobs.results')}
            </Button>
          )}
        </Inline>
      </Stack>
    </Card>
  )
}
