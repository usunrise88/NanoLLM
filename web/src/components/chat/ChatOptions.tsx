import { Button, Field, Input, Select, Sheet, Switch, Textarea } from '@/components/ui'
import { Grid, Stack } from '@/components/layout'
import { useDevices } from '@/lib/api/hooks'
import type { ResponseFormatType } from '@/lib/api/types'
import type { Dialect } from '@/lib/api/chat'
import { useT } from '@/lib/i18n'
import { resetPrefs, setPrefs, usePrefs, type ChatPrefs } from '@/lib/prefs'

/**
 * Everything about a request that is not the conversation itself: dialect,
 * sampling, limits, structured output, tools, device. A sheet rather than a
 * row of controls because these are set once and then left alone.
 */
export function ChatOptions({
  open,
  onOpenChange,
  invalid,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** The field the last send refused, with why. */
  invalid?: { field: keyof ChatPrefs; message: string } | undefined
}) {
  const t = useT()
  const prefs = usePrefs()
  const devices = useDevices()

  const err = (f: keyof ChatPrefs) => (invalid?.field === f ? invalid.message : undefined)
  const text = (f: keyof ChatPrefs) => ({
    value: String(prefs[f]),
    onChange: (e: { target: { value: string } }) => setPrefs({ [f]: e.target.value } as Partial<ChatPrefs>),
  })

  return (
    <Sheet
      open={open}
      onOpenChange={onOpenChange}
      title={t('chat.options')}
      description={t('chat.optionsHint')}
      footer={
        <>
          <Button onClick={resetPrefs}>{t('chat.reset')}</Button>
          <Button variant="primary" onClick={() => onOpenChange(false)}>
            {t('common.close')}
          </Button>
        </>
      }
    >
      <Stack gap={4}>
        <Field label={t('chat.dialect')} description={t('chat.dialectHint')}>
          <Select
            value={prefs.dialect}
            onChange={(v) => setPrefs({ dialect: v as Dialect })}
            options={[
              { value: 'native', label: t('chat.dialectNative') },
              { value: 'openai', label: t('chat.dialectOpenAI') },
            ]}
          />
        </Field>

        <Field label={t('chat.system')}>
          <Textarea rows={3} {...text('system')} placeholder={t('chat.systemPlaceholder')} />
        </Field>

        <Grid track="sm" gap={3}>
          <Field label={t('chat.temperature')} error={err('temperature')}>
            <Input inputMode="decimal" placeholder={t('chat.auto')} {...text('temperature')} />
          </Field>
          <Field label="top_p" error={err('topP')}>
            <Input inputMode="decimal" placeholder={t('chat.auto')} {...text('topP')} />
          </Field>
          <Field label="top_k" error={err('topK')}>
            <Input inputMode="numeric" placeholder={t('chat.auto')} {...text('topK')} />
          </Field>
          <Field label={t('chat.seed')} error={err('seed')}>
            <Input inputMode="numeric" placeholder={t('chat.auto')} {...text('seed')} />
          </Field>
          <Field label={t('chat.maxTokens')} error={err('maxTokens')}>
            <Input inputMode="numeric" placeholder={t('chat.auto')} {...text('maxTokens')} />
          </Field>
          <Field label={t('chat.maxReasoning')} error={err('maxReasoningTokens')}>
            <Input inputMode="numeric" placeholder={t('chat.auto')} {...text('maxReasoningTokens')} />
          </Field>
        </Grid>

        <Field label={t('chat.stopSequences')} description={t('chat.stopHint')}>
          <Textarea rows={2} {...text('stop')} />
        </Field>

        <Field label={t('chat.device')} description={prefs.dialect === 'openai' ? t('chat.nativeOnly') : undefined}>
          <Select
            value={prefs.device}
            onChange={(v) => setPrefs({ device: v })}
            options={[
              { value: '', label: t('chat.deviceDefault') },
              ...(devices.data ?? []).map((d) => ({ value: d.selector, label: `${d.name} (${d.selector})` })),
            ]}
          />
        </Field>

        <Switch
          checked={prefs.cachePrompt}
          onChange={(v) => setPrefs({ cachePrompt: v })}
          label={t('chat.cachePrompt')}
          description={t('chat.cachePromptHint')}
        />

        <Switch checked={prefs.logprobs} onChange={(v) => setPrefs({ logprobs: v })} label={t('chat.logprobsOption')} />
        {prefs.logprobs && (
          <Field label="top_logprobs" error={err('topLogprobs')}>
            <Input inputMode="numeric" placeholder="0–20" {...text('topLogprobs')} />
          </Field>
        )}

        <Field label={t('chat.responseFormat')}>
          <Select
            value={prefs.responseFormat}
            onChange={(v) => setPrefs({ responseFormat: v as ResponseFormatType })}
            options={[
              { value: 'text', label: t('chat.formatText') },
              { value: 'json_object', label: t('chat.formatJson') },
              { value: 'json_schema', label: t('chat.formatSchema') },
            ]}
          />
        </Field>
        {prefs.responseFormat === 'json_schema' && (
          <Field label={t('chat.schema')} error={err('schema')}>
            <Textarea rows={6} spellCheck={false} className="font-[var(--font-mono)] text-[12px]" {...text('schema')} placeholder='{"type":"object","properties":{"name":{"type":"string"}}}' />
          </Field>
        )}

        <Field label={t('chat.tools')} description={t('chat.toolsHint')} error={err('tools')}>
          <Textarea
            rows={6}
            spellCheck={false}
            className="font-[var(--font-mono)] text-[12px]"
            {...text('tools')}
            placeholder='[{"name":"get_weather","description":"Weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]'
          />
        </Field>
      </Stack>
    </Sheet>
  )
}
