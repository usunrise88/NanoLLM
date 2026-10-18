import { useCallback, useEffect, useRef, useState } from 'react'
import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
} from '@tanstack/react-query'

import {
  ApiError,
  cancelJob,
  downloadPath,
  errorText,
  getConfig,
  getJob,
  jobEventsPath,
  listCatalog,
  listDevices,
  listJobs,
  listModels,
  loadModel,
  pinModel,
  reloadModel,
  submitJob,
  unloadModel,
} from './client'
import { postStream, readFrames, stream } from './sse'
import { isTerminal, type DownloadProgress, type Job, type JobFilter } from './types'

/** Query keys, in one place so an invalidation cannot miss a reader. */
export const keys = {
  models: ['models'] as const,
  catalog: ['catalog'] as const,
  devices: ['devices'] as const,
  config: ['config'] as const,
  jobs: (filter: JobFilter) => ['jobs', filter] as const,
  job: (id: string) => ['job', id] as const,
}

/**
 * A 4xx or a 501 will not fix itself by being asked again; a dropped
 * connection or a 503 while a model loads might.
 */
function retry(count: number, err: unknown): boolean {
  if (err instanceof ApiError && ((err.status >= 400 && err.status < 500) || err.status === 501)) return false
  return count < 1
}

export function useModels() {
  return useQuery({ queryKey: keys.models, queryFn: listModels, retry })
}

export function useCatalog() {
  return useQuery({ queryKey: keys.catalog, queryFn: listCatalog, retry, staleTime: 60_000 })
}

export function useDevices() {
  return useQuery({ queryKey: keys.devices, queryFn: listDevices, retry })
}

export function useConfig() {
  return useQuery({ queryKey: keys.config, queryFn: getConfig, retry })
}

/** The model-state mutations, which differ only in which call they make. */
function useModelAction<A extends unknown[], R>(
  action: (...args: A) => Promise<R>,
): UseMutationResult<R, Error, A> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (args: A) => action(...args),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: keys.models })
      void qc.invalidateQueries({ queryKey: keys.devices })
    },
  })
}

export const useLoadModel = () => useModelAction(loadModel)
export const useUnloadModel = () => useModelAction(unloadModel)
export const usePinModel = () => useModelAction(pinModel)
export const useReloadModel = () => useModelAction(reloadModel)

export interface DownloadState {
  progress: DownloadProgress | undefined
  error: string | undefined
  running: boolean
  done: boolean
}

/**
 * Downloads a model, reporting progress.
 *
 * Not a mutation: the useful part is the stream of ticks, and react-query has
 * nowhere to put those. A second click while the download runs joins it on
 * the server, so leaving the page and coming back costs nothing.
 */
export function useModelDownload(): DownloadState & { start: (id: string) => void; cancel: () => void } {
  const qc = useQueryClient()
  const [state, setState] = useState<DownloadState>({
    progress: undefined,
    error: undefined,
    running: false,
    done: false,
  })
  const abort = useRef<AbortController>(undefined)

  useEffect(() => () => abort.current?.abort(), [])

  const start = useCallback(
    (id: string) => {
      abort.current?.abort()
      const controller = new AbortController()
      abort.current = controller
      setState({ progress: undefined, error: undefined, running: true, done: false })

      void (async () => {
        try {
          const response = await postStream(downloadPath(id), undefined, controller.signal)
          let finished = false
          await readFrames(response, (frame) => {
            let payload: unknown
            try {
              payload = JSON.parse(frame.data)
            } catch {
              return
            }
            if (frame.event === 'progress') {
              setState((s) => ({ ...s, progress: payload as DownloadProgress }))
            } else if (frame.event === 'done') {
              finished = true
              setState((s) => ({ ...s, running: false, done: true }))
              return true
            } else if (frame.event === 'error') {
              const p = payload as { detail?: string; title?: string }
              setState((s) => ({ ...s, running: false, error: p.detail ?? p.title ?? 'download failed' }))
              return true
            }
            return
          })
          if (!finished) setState((s) => ({ ...s, running: false }))
        } catch (err) {
          if (controller.signal.aborted) {
            setState((s) => ({ ...s, running: false }))
            return
          }
          setState((s) => ({ ...s, running: false, error: errorText(err) }))
        } finally {
          void qc.invalidateQueries({ queryKey: keys.models })
        }
      })()
    },
    [qc],
  )

  const cancel = useCallback(() => abort.current?.abort(), [])

  return { ...state, start, cancel }
}

// --- jobs ---------------------------------------------------------------------

export function useJobHistory(filter: JobFilter) {
  return useInfiniteQuery({
    queryKey: keys.jobs(filter),
    queryFn: ({ pageParam }) => listJobs(pageParam ? { ...filter, cursor: pageParam } : filter),
    initialPageParam: '',
    getNextPageParam: (page) => page.next_cursor || undefined,
    retry,
  })
}

/**
 * One job. A row of the history passes the listed job as its starting point,
 * so the row draws at once and the stream's updates land in the same entry.
 */
export function useJob(id: string, initial?: Job) {
  return useQuery({
    queryKey: keys.job(id),
    queryFn: () => getJob(id),
    retry,
    ...(initial ? { initialData: initial, staleTime: Infinity } : {}),
  })
}

export function useSubmitJob(): UseMutationResult<Job, Error, { jsonl: string; model?: string }> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ jsonl, model }) => submitJob(jsonl, model),
    onSuccess: (job) => {
      qc.setQueryData(keys.job(job.id), job)
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

export function useCancelJob(): UseMutationResult<Job, Error, string> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: cancelJob,
    onSuccess: (job) => {
      qc.setQueryData(keys.job(job.id), job)
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

/**
 * Follows a job until it is terminal. Every event overwrites the job's cache
 * entry, so anything reading it through useJob updates without knowing a
 * stream exists.
 */
export function useJobStream(job: Job | undefined): void {
  const qc = useQueryClient()
  const id = job?.id
  const active = job !== undefined && !isTerminal(job.status)

  useEffect(() => {
    if (!id || !active) return
    const controller = new AbortController()
    void stream<Job>(jobEventsPath(id), {
      signal: controller.signal,
      onEvent: (_event, next) => {
        if (next && typeof next === 'object' && 'status' in next) qc.setQueryData(keys.job(id), next)
      },
      onDone: () => {
        void qc.invalidateQueries({ queryKey: keys.job(id) })
        void qc.invalidateQueries({ queryKey: ['jobs'] })
      },
    })
    return () => controller.abort()
  }, [id, active, qc])
}
