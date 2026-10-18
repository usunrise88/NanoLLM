import { apiKey, reportAuthRequired } from '@/lib/auth'

import type {
  BenchRequest,
  BenchStarted,
  BenchSubmitted,
  CatalogEntry,
  DeviceInfo,
  Job,
  JobFilter,
  JobPage,
  ModelInfo,
  TokenizeResult,
  Message,
  Tool,
} from './types'

/** Every native path lives under here; the OpenAI dialect under /v1. */
export const BASE = '/api/v1'
export const OPENAI_BASE = '/v1'

/**
 * A refusal from the server, in the shape it actually sends.
 *
 * The native dialect answers RFC 9457 problem+json, the OpenAI dialect an
 * `{error: {message, code, param}}` envelope; both become this, so a screen
 * branches on `code` without knowing which dialect it called.
 */
export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly param: string | undefined
  readonly retryAfter: number | undefined

  constructor(init: {
    status: number
    code: string
    detail: string
    param?: string | undefined
    retryAfter?: number | undefined
  }) {
    super(init.detail)
    this.name = 'ApiError'
    this.status = init.status
    this.code = init.code
    this.param = init.param
    this.retryAfter = init.retryAfter
  }

  /**
   * The route exists in the API but not in this build yet (501), or not at
   * all (404 with no problem body — the plain mux answer). Screens built
   * ahead of their backend show "not yet" for this instead of an error.
   */
  get notYet(): boolean {
    return this.status === 501 || (this.status === 404 && this.code === 'not_found')
  }
}

/** Whatever was thrown, as a sentence a person can read. */
export function errorText(err: unknown): string {
  if (err instanceof ApiError) return err.message
  if (err instanceof Error) return err.message
  return String(err)
}

export function isNotYet(err: unknown): boolean {
  return err instanceof ApiError && err.notYet
}

export function authHeaders(): HeadersInit {
  const key = apiKey()
  return key ? { Authorization: `Bearer ${key}` } : {}
}

interface ProblemBody {
  code?: string
  detail?: string
  title?: string
  param?: string
  error?: { message?: string; code?: string; param?: string }
}

/** Reads either error shape out of a response that is not ok. */
export async function toError(response: Response): Promise<ApiError> {
  const retryAfterHeader = response.headers.get('Retry-After')
  const retryAfter = retryAfterHeader ? Number(retryAfterHeader) : undefined

  let code = response.status === 404 ? 'not_found' : 'internal'
  let detail = response.statusText || `HTTP ${response.status}`
  let param: string | undefined

  try {
    const text = await response.text()
    const body = JSON.parse(text) as ProblemBody
    if (body.error) {
      code = body.error.code ?? code
      detail = body.error.message ?? detail
      param = body.error.param
    } else {
      code = body.code ?? code
      detail = body.detail ?? body.title ?? detail
      param = body.param
    }
  } catch {
    // Not every failure comes from our handlers: the mux answers a plain-text
    // 404, a proxy in front answers HTML, and the status is the useful part.
  }

  return new ApiError({
    status: response.status,
    code,
    detail,
    param,
    retryAfter: retryAfter === undefined || Number.isNaN(retryAfter) ? undefined : retryAfter,
  })
}

/** The one place a JSON request leaves the app. `path` is relative to BASE. */
export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(BASE + path, {
    ...init,
    headers: { ...authHeaders(), ...(init.headers ?? {}) },
  })

  if (response.status === 401) reportAuthRequired()
  if (!response.ok) throw await toError(response)
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

function json(body: unknown): RequestInit {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

const enc = encodeURIComponent

// --- models and devices -------------------------------------------------------

export async function listModels(): Promise<ModelInfo[]> {
  return (await request<{ models: ModelInfo[] }>('/models')).models ?? []
}

export async function listCatalog(): Promise<CatalogEntry[]> {
  return (await request<{ models: CatalogEntry[] }>('/catalog')).models ?? []
}

export async function listDevices(): Promise<DeviceInfo[]> {
  return (await request<{ devices: DeviceInfo[] }>('/devices')).devices ?? []
}

export function loadModel(id: string): Promise<ModelInfo> {
  return request<ModelInfo>(`/models/${enc(id)}/load`, { method: 'POST' })
}

export function unloadModel(id: string): Promise<void> {
  return request<void>(`/models/${enc(id)}/unload`, { method: 'POST' })
}

export function reloadModel(id: string): Promise<ModelInfo> {
  return request<ModelInfo>(`/models/${enc(id)}/reload`, { method: 'POST' })
}

export function pinModel(id: string, pin: boolean): Promise<{ id: string; pinned: boolean }> {
  return request(`/models/${enc(id)}/pin`, json({ pin }))
}

export function downloadPath(id: string): string {
  return `${BASE}/models/${enc(id)}/download`
}

/** The effective configuration, secrets redacted by the server. Admin only. */
export function getConfig(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>('/config')
}

export function tokenize(body: {
  model?: string
  text?: string
  messages?: Message[]
  tools?: Tool[]
}): Promise<TokenizeResult> {
  return request<TokenizeResult>('/tokenize', json(body))
}

// --- jobs ---------------------------------------------------------------------

/**
 * Enqueues a batch: one chat request per JSONL line, sent as it is so the
 * server reports a bad line by its number rather than this re-encoding it.
 */
export function submitJob(jsonl: string, model?: string): Promise<Job> {
  const q = model ? `?model=${enc(model)}` : ''
  return request<Job>(`/jobs${q}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-ndjson' },
    body: jsonl,
  })
}

export function listJobs(filter: JobFilter = {}): Promise<JobPage> {
  const q = new URLSearchParams()
  if (filter.status) q.set('status', filter.status)
  if (filter.limit) q.set('limit', String(filter.limit))
  if (filter.cursor) q.set('cursor', filter.cursor)
  const query = q.toString()
  return request<JobPage>(`/jobs${query ? `?${query}` : ''}`)
}

export function getJob(id: string): Promise<Job> {
  return request<Job>(`/jobs/${enc(id)}`)
}

export function cancelJob(id: string): Promise<Job> {
  return request<Job>(`/jobs/${enc(id)}`, { method: 'DELETE' })
}

export function jobEventsPath(id: string): string {
  return `${BASE}/jobs/${enc(id)}/events`
}

/**
 * Splits a JSONL batch and checks each line parses, so a typo is pointed out
 * by line number before anything is sent. Blank lines are allowed and skipped.
 */
export function validateJsonl(text: string): { count: number; error?: { line: number; message: string } } {
  let count = 0
  const lines = text.split(/\r?\n/)
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]?.trim() ?? ''
    if (line === '') continue
    try {
      const v = JSON.parse(line) as unknown
      if (typeof v !== 'object' || v === null || Array.isArray(v)) {
        return { count, error: { line: i + 1, message: 'not a JSON object' } }
      }
    } catch (e) {
      return { count, error: { line: i + 1, message: e instanceof Error ? e.message : String(e) } }
    }
    count++
  }
  return { count }
}

// --- bench --------------------------------------------------------------------

export function startBench(req: BenchRequest): Promise<BenchStarted> {
  return request<BenchStarted>('/bench', json(req))
}

export function benchEventsPath(id: string): string {
  return `${BASE}/bench/${enc(id)}/events`
}

/**
 * Sends the result the person has just read. The body names the consent text
 * they saw, so the server can refuse a submission made under an older one.
 */
export function submitBench(id: string, consentVersion: string, anonymous: boolean): Promise<BenchSubmitted> {
  return request<BenchSubmitted>(`/bench/${enc(id)}/submit`, json({ consent_version: consentVersion, anonymous }))
}
