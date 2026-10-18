import { createFileRoute } from '@tanstack/react-router'

import { Card, Grid, Inline, Page, Section, Stack } from '@/components/layout'
import { Badge, Button, Detail, EmptyState, ErrorState, Progress, SkeletonRow } from '@/components/ui'
import { errorText } from '@/lib/api/client'
import { useDevices, useModels } from '@/lib/api/hooks'
import type { DeviceInfo, ModelInfo } from '@/lib/api/types'
import { bytes } from '@/lib/format'
import { useT } from '@/lib/i18n'
import type { PageMeta } from '@/lib/page'
import { useDelayedPending } from '@/lib/pending'

export const pageMeta: PageMeta = {
  titleKey: 'devices.title',
  descriptionKey: 'devices.description',
}

export const Route = createFileRoute('/devices')({
  component: DevicesPage,
  staticData: { pageMeta },
})

/** Models with an instance on this device, matched the way the pool keys them. */
function residentOn(d: DeviceInfo, models: ModelInfo[]): string[] {
  const out: string[] = []
  for (const m of models) {
    for (const inst of m.loaded ?? []) {
      const k = inst.key
      if (!k || inst.retired) continue
      if (k.device === d.id && (!k.backend || k.backend === d.backend)) {
        out.push(m.id)
        break
      }
    }
  }
  return out
}

function DevicesPage() {
  const t = useT()
  const devices = useDevices()
  const models = useModels()
  const loading = useDelayedPending(devices.isLoading)

  if (devices.isError) {
    return (
      <Page>
        <Section>
          <ErrorState
            title={t('common.error')}
            detail={errorText(devices.error)}
            action={
              <Button size="sm" onClick={() => void devices.refetch()}>
                {t('common.retry')}
              </Button>
            }
          />
        </Section>
      </Page>
    )
  }

  const list = devices.data ?? []
  const backends = [...new Set(list.map((d) => d.backend))]

  return (
    <Page>
      {loading && (
        <Card>
          <SkeletonRow />
        </Card>
      )}
      {!devices.isLoading && list.length === 0 && (
        <Card>
          <EmptyState title={t('devices.empty')} description={t('devices.emptyHint')} />
        </Card>
      )}
      {backends.map((b) => (
        <Section key={b} title={t('devices.backend', { name: b })}>
          <Grid track="md" gap={3}>
            {list
              .filter((d) => d.backend === b)
              .map((d) => (
                <DeviceCard key={d.selector} device={d} resident={residentOn(d, models.data ?? [])} />
              ))}
          </Grid>
        </Section>
      ))}
    </Page>
  )
}

function DeviceCard({ device, resident }: { device: DeviceInfo; resident: string[] }) {
  const t = useT()
  const used = Math.max(0, device.memory_total - device.memory_free)
  const share = device.memory_total > 0 ? (used / device.memory_total) * 100 : 0
  return (
    <Card data-item={device.selector}>
      <Stack gap={3}>
        <Inline gap={2}>
          <span className="text-[13px] font-medium">{device.name}</span>
          <Badge tone="accent">{device.type}</Badge>
          {device.unified && <Badge>{t('devices.unified')}</Badge>}
        </Inline>
        <Stack gap={1}>
          <Detail label={t('devices.selector')}>
            <span className="font-[var(--font-mono)]">{device.selector}</span>
          </Detail>
          <Detail label={t('devices.memory')}>
            {bytes(used)} / {bytes(device.memory_total)}
          </Detail>
          <Progress percent={share} label={t('devices.memory')} />
          <Detail label={t('devices.free')}>{bytes(device.memory_free)}</Detail>
        </Stack>
        <Stack gap={1}>
          <span className="text-[12px] text-[var(--text-muted)]">{t('devices.resident')}</span>
          {resident.length ? (
            <Inline gap={1}>
              {resident.map((m) => (
                <Badge key={m} tone="success">
                  {m}
                </Badge>
              ))}
            </Inline>
          ) : (
            <span className="text-[12px]">{t('devices.none')}</span>
          )}
        </Stack>
        <Stack gap={1}>
          <span className="text-[12px] text-[var(--text-muted)]">{t('devices.verified')}</span>
          {device.verified?.length ? (
            <Inline gap={1}>
              {device.verified.map((m) => (
                <Badge key={m}>{m}</Badge>
              ))}
            </Inline>
          ) : (
            <span className="text-[12px]">{t('devices.notVerified')}</span>
          )}
        </Stack>
      </Stack>
    </Card>
  )
}
