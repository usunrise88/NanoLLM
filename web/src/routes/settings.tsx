import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery, useQueryClient } from '@tanstack/react-query'

import { JsonBlock } from '@/components/JsonView'
import { Card, Inline, Page, Section, Stack } from '@/components/layout'
import { Button, Detail, ErrorState, Field, Input, Select, Skeleton } from '@/components/ui'
import { ApiError, errorText } from '@/lib/api/client'
import { useConfig } from '@/lib/api/hooks'
import { setApiKey, useAuth } from '@/lib/auth'
import { clearConversations } from '@/lib/conversations'
import { currentLanguage, languages, setLanguage, useT, type Language } from '@/lib/i18n'
import type { PageMeta } from '@/lib/page'
import { setTheme, useTheme, type Theme } from '@/lib/theme'
import { toast } from '@/lib/toast'

export const pageMeta: PageMeta = {
  titleKey: 'settings.title',
  descriptionKey: 'settings.description',
}

export const Route = createFileRoute('/settings')({
  component: SettingsPage,
  staticData: { pageMeta },
})

interface Health {
  status: string
  version: string
  llama_cpp: string
  backends?: { name: string; version: string }[]
}

function SettingsPage() {
  const t = useT()
  const auth = useAuth()
  const theme = useTheme()
  const qc = useQueryClient()
  const [key, setKey] = useState(auth.key)
  const config = useConfig()
  // /healthz is public: it answers even before a key is entered, which is
  // when knowing the server's version is most useful.
  const health = useQuery({
    queryKey: ['healthz'],
    queryFn: async () => (await (await fetch('/healthz')).json()) as Health,
  })

  function saveKey() {
    setApiKey(key)
    // Whatever was refused for want of a key deserves another attempt.
    void qc.invalidateQueries()
    toast.success(t('settings.saved'))
  }

  const forbidden = config.error instanceof ApiError && (config.error.status === 403 || config.error.status === 401)

  return (
    <Page>
      <Section title={t('settings.interface')}>
        <Card>
          <Stack gap={4}>
            <Field label={t('settings.language')}>
              <Select
                value={currentLanguage()}
                onChange={(v) => setLanguage(v as Language)}
                options={languages.map((l) => ({ value: l, label: l === 'ru' ? 'Русский' : 'English' }))}
              />
            </Field>
            <Field label={t('settings.theme')}>
              <Select
                value={theme}
                onChange={(v) => setTheme(v as Theme)}
                options={[
                  { value: 'light', label: t('theme.light') },
                  { value: 'dark', label: t('theme.dark') },
                  { value: 'system', label: t('theme.system') },
                ]}
              />
            </Field>
          </Stack>
        </Card>
      </Section>

      <Section title={t('settings.apiKey')}>
        <Card>
          <Stack gap={3}>
            <Field
              label={t('settings.apiKey')}
              description={t('settings.apiKeyHint')}
              error={auth.required && !auth.key ? t('settings.apiKeyRequired') : undefined}
            >
              <Input
                type="password"
                value={key}
                spellCheck={false}
                autoComplete="off"
                placeholder="sk-…"
                onChange={(e) => setKey(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveKey()
                }}
              />
            </Field>
            <Inline>
              <Button variant="primary" size="sm" onClick={saveKey}>
                {t('settings.save')}
              </Button>
            </Inline>
          </Stack>
        </Card>
      </Section>

      <Section title={t('settings.server')}>
        <Card>
          {health.data ? (
            <Stack gap={1}>
              <Detail label={t('settings.version')}>{health.data.version}</Detail>
              <Detail label="llama.cpp">{health.data.llama_cpp}</Detail>
              <Detail label={t('settings.backends')}>
                {(health.data.backends ?? []).map((b) => `${b.name} ${b.version}`).join(', ') || '—'}
              </Detail>
            </Stack>
          ) : (
            <Skeleton className="h-12 w-full" />
          )}
        </Card>
      </Section>

      <Section title={t('settings.config')} description={t('settings.configHint')}>
        {config.isLoading && <Skeleton className="h-40 w-full" />}
        {forbidden && <ErrorState title={t('settings.configForbidden')} detail={errorText(config.error)} />}
        {config.isError && !forbidden && <ErrorState title={t('common.error')} detail={errorText(config.error)} />}
        {config.data && <JsonBlock value={config.data} label={t('settings.config')} />}
      </Section>

      <Section title={t('settings.history')}>
        <Card>
          <Inline gap={3}>
            <span className="text-[13px] text-[var(--text-secondary)]">{t('settings.historyHint')}</span>
            <Button
              size="sm"
              variant="danger"
              onClick={() => {
                clearConversations()
                toast.success(t('settings.historyCleared'))
              }}
            >
              {t('settings.clearHistory')}
            </Button>
          </Inline>
        </Card>
      </Section>
    </Page>
  )
}
