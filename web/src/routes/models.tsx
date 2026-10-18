import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { Download, Pin, PinSolid, Refresh, Xmark } from 'iconoir-react'

import { Collapsible, JsonBlock } from '@/components/JsonView'
import { Card, Inline, Page, Section, Stack } from '@/components/layout'
import {
  Badge,
  Button,
  Detail,
  Dialog,
  EmptyState,
  ErrorState,
  Progress,
  SkeletonRow,
  type Tone,
} from '@/components/ui'
import { errorText } from '@/lib/api/client'
import {
  useCatalog,
  useLoadModel,
  useModelDownload,
  useModels,
  usePinModel,
  useReloadModel,
  useUnloadModel,
} from '@/lib/api/hooks'
import type { CatalogEntry, DiskState, LoadedInstance, ModelInfo } from '@/lib/api/types'
import { bytes, when } from '@/lib/format'
import { currentLanguage, useT } from '@/lib/i18n'
import type { PageMeta } from '@/lib/page'
import { useDelayedPending } from '@/lib/pending'
import { toast } from '@/lib/toast'

export const pageMeta: PageMeta = {
  titleKey: 'models.title',
  descriptionKey: 'models.description',
}

export const Route = createFileRoute('/models')({
  component: ModelsPage,
  staticData: { pageMeta },
})

function diskTone(d: DiskState): Tone {
  switch (d) {
    case 'complete':
      return 'success'
    case 'partial':
      return 'warning'
    case 'corrupt':
      return 'danger'
    default:
      return 'neutral'
  }
}

function ModelsPage() {
  const t = useT()
  const models = useModels()
  const catalog = useCatalog()
  const loading = useDelayedPending(models.isLoading)

  if (models.isError) {
    return (
      <Page>
        <Section>
          <ErrorState
            title={t('common.error')}
            detail={errorText(models.error)}
            action={
              <Button size="sm" onClick={() => void models.refetch()}>
                {t('common.retry')}
              </Button>
            }
          />
        </Section>
      </Page>
    )
  }

  const list = models.data ?? []
  const byId = new Map((catalog.data ?? []).map((c) => [c.id, c]))

  return (
    <Page>
      <Section>
        {loading && (
          <Card>
            <SkeletonRow />
            <SkeletonRow />
          </Card>
        )}
        {!models.isLoading && list.length === 0 && (
          <Card>
            <EmptyState title={t('models.empty')} description={t('models.emptyHint')} />
          </Card>
        )}
        <Stack gap={3}>
          {list.map((m) => (
            <ModelCard key={m.id} model={m} entry={byId.get(m.id)} />
          ))}
        </Stack>
      </Section>
    </Page>
  )
}

function ModelCard({ model, entry }: { model: ModelInfo; entry: CatalogEntry | undefined }) {
  const t = useT()
  const load = useLoadModel()
  const unload = useUnloadModel()
  const pin = usePinModel()
  const reload = useReloadModel()
  const download = useModelDownload()
  const [confirmUnload, setConfirmUnload] = useState(false)

  const instances = model.loaded ?? []
  const resident = instances.some((i) => !i.retired)
  const busy = load.isPending || unload.isPending || pin.isPending || reload.isPending

  // Every action fails the same way — a 403 for a key without admin rights,
  // a 503 for a model that does not fit — so they report it the same way.
  function report(err: unknown) {
    toast.error(t('models.actionFailed'), { description: errorText(err) })
  }
  function done(key: Parameters<typeof t>[0]) {
    return () => toast.success(t(key, { id: model.id }))
  }

  return (
    <Card data-item={model.id}>
      <Stack gap={3}>
        <Inline gap={2}>
          <span className="text-[14px] font-medium">{model.id}</span>
          <Badge tone={diskTone(model.disk)}>{t(`disk.${model.disk}`)}</Badge>
          {resident && <Badge tone="success">{t('models.resident', { count: instances.filter((i) => !i.retired).length })}</Badge>}
          {model.pinned && <Badge tone="accent">{t('models.pinned')}</Badge>}
          {model.default && <Badge>{t('models.default')}</Badge>}
        </Inline>

        <Stack gap={1}>
          {(model.family || model.params || model.quant) && (
            <Detail label={t('models.variant')}>{[model.family, model.params, model.quant].filter(Boolean).join(' · ')}</Detail>
          )}
          <Detail label={t('models.size')}>{bytes(model.size_bytes)}</Detail>
          {entry?.license && (
            <Detail label={t('models.license')}>
              {entry.license}
              {entry.commercial ? ` (${entry.commercial})` : ''}
            </Detail>
          )}
          {entry?.reasoning && (
            <Detail label={t('models.thinking')}>
              {entry.reasoning.thinking_default ? t('models.thinkingOn') : t('models.thinkingOff')}
            </Detail>
          )}
        </Stack>

        {download.running && (
          <Stack gap={1}>
            <Progress percent={download.progress?.percent ?? 0} label={t('models.download')} />
            {download.progress && (
              <span className="text-[11px] text-[var(--text-muted)] tabular-nums" data-download-file="">
                {download.progress.file} · {bytes(download.progress.downloaded)} / {bytes(download.progress.total)}
              </span>
            )}
          </Stack>
        )}
        {download.error && (
          <p role="alert" className="text-[12px] text-[var(--danger-text)]">
            {download.error}
          </p>
        )}
        {download.done && <p className="text-[12px] text-[var(--success-text)]">{t('models.downloaded')}</p>}

        <Inline gap={2}>
          {model.disk !== 'complete' &&
            (download.running ? (
              <Button size="sm" icon={<Xmark width={13} height={13} />} onClick={download.cancel}>
                {t('models.cancelDownload')}
              </Button>
            ) : (
              <Button size="sm" icon={<Download width={13} height={13} />} onClick={() => download.start(model.id)}>
                {model.disk === 'partial' ? t('models.resume') : t('models.download')}
              </Button>
            ))}
          {!resident ? (
            <Button
              size="sm"
              variant="primary"
              busy={load.isPending}
              disabled={busy}
              onClick={() => void load.mutateAsync([model.id]).then(done('models.loaded'), report)}
            >
              {t('models.load')}
            </Button>
          ) : (
            <Button size="sm" busy={unload.isPending} disabled={busy} onClick={() => setConfirmUnload(true)}>
              {t('models.unload')}
            </Button>
          )}
          <Button
            size="sm"
            busy={reload.isPending}
            disabled={busy}
            icon={<Refresh width={13} height={13} />}
            title={t('models.reloadHint')}
            onClick={() => void reload.mutateAsync([model.id]).then(done('models.reloaded'), report)}
          >
            {t('models.reload')}
          </Button>
          <Button
            size="sm"
            busy={pin.isPending}
            disabled={busy}
            icon={model.pinned ? <PinSolid width={13} height={13} /> : <Pin width={13} height={13} />}
            onClick={() => void pin.mutateAsync([model.id, !model.pinned]).catch(report)}
          >
            {model.pinned ? t('models.unpin') : t('models.pin')}
          </Button>
        </Inline>

        {instances.length > 0 && <Instances instances={instances} />}

        {model.plan !== undefined && model.plan !== null && (
          <Collapsible title={t('models.plan')}>
            <JsonBlock value={model.plan} />
          </Collapsible>
        )}
        {entry?.files && entry.files.length > 0 && (
          <Collapsible title={t('models.files', { count: entry.files.length })}>
            <Stack gap={1}>
              {entry.files.map((f) => (
                <Detail key={f.path} label={`${f.role}: ${f.path}`}>
                  {bytes(f.size)}
                </Detail>
              ))}
            </Stack>
          </Collapsible>
        )}
      </Stack>

      <Dialog
        open={confirmUnload}
        onOpenChange={setConfirmUnload}
        title={t('models.confirmUnload')}
        description={t('models.confirmUnloadBody')}
        footer={
          <>
            <Button onClick={() => setConfirmUnload(false)}>{t('common.cancel')}</Button>
            <Button
              variant="danger"
              onClick={() => {
                setConfirmUnload(false)
                void unload.mutateAsync([model.id]).then(done('models.unloaded'), report)
              }}
            >
              {t('models.unload')}
            </Button>
          </>
        }
      >
        <p className="text-[13px]">{model.id}</p>
      </Dialog>
    </Card>
  )
}

/**
 * The resident instances: one per (revision, device, context) the pool holds.
 * During a hot swap there are two, and the old one is marked retiring until
 * its last request finishes.
 */
function Instances({ instances }: { instances: LoadedInstance[] }) {
  const t = useT()
  const locale = currentLanguage()
  return (
    <Stack gap={2}>
      {instances.map((inst, i) => {
        const k = inst.key ?? {}
        const info = inst.info ?? {}
        return (
          <div key={i} data-instance={`${k.backend ?? ''}:${k.device ?? ''}`}>
            <Card>
              <Stack gap={1}>
                <Inline gap={2}>
                  <span className="text-[12px] font-medium">{[k.backend, k.device].filter(Boolean).join(':')}</span>
                  {k.revision && <span className="font-[var(--font-mono)] text-[11px] text-[var(--text-muted)]">{k.revision.slice(0, 8)}</span>}
                  {inst.retired && <Badge tone="warning">{t('models.retiring')}</Badge>}
                  {inst.pinned && <Badge tone="accent">{t('models.pinned')}</Badge>}
                  {(inst.busy ?? 0) > 0 && <Badge tone="accent">{t('models.busy', { count: inst.busy ?? 0 })}</Badge>}
                </Inline>
                {k.n_ctx !== undefined && <Detail label="n_ctx">{k.n_ctx}</Detail>}
                {typeof info.Slots === 'number' && <Detail label={t('models.slots')}>{info.Slots}</Detail>}
                {typeof info.GPULayers === 'number' && (
                  <Detail label={t('models.gpuLayers')}>{info.GPULayers < 0 ? t('models.allLayers') : info.GPULayers}</Detail>
                )}
                {typeof info.KVTypeK === 'string' && <Detail label="KV">{`${info.KVTypeK} / ${String(info.KVTypeV ?? '')}`}</Detail>}
                {inst.loaded_at && <Detail label={t('models.loadedAt')}>{when(inst.loaded_at, locale)}</Detail>}
                {Object.keys(info).length > 0 && (
                  <Collapsible title={t('models.placement')}>
                    <JsonBlock value={info} />
                  </Collapsible>
                )}
              </Stack>
            </Card>
          </div>
        )
      })}
    </Stack>
  )
}
