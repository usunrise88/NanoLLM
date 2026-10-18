/**
 * Wire shapes of the public API, as the server sends them.
 *
 * Mirrors internal/core (ChatRequest, Delta, ChatResult, Stats, ModelInfo,
 * DeviceInfo) and the OpenAI dialect's chunk. Nothing here is a view model:
 * a field that the UI derives lives next to the code that derives it.
 */

// --- chat (native dialect, core types) ---------------------------------------

export type Role = 'system' | 'user' | 'assistant' | 'tool'

export interface ToolCall {
  id: string
  name: string
  /** JSON text, as the model wrote it. */
  arguments: string
}

export interface Message {
  role: Role
  content: string
  reasoning_content?: string
  tool_calls?: ToolCall[]
  tool_call_id?: string
  name?: string
}

export interface Tool {
  name: string
  description?: string
  parameters?: unknown
}

export interface Sampling {
  temperature?: number
  top_p?: number
  top_k?: number
  min_p?: number
  repeat_penalty?: number
  presence_penalty?: number
  frequency_penalty?: number
  seed?: number
}

export type ResponseFormatType = 'text' | 'json_object' | 'json_schema'

export interface ResponseFormat {
  type: ResponseFormatType
  schema?: unknown
}

export interface ChatRequest {
  model?: string
  messages: Message[]
  sampling?: Sampling
  max_tokens?: number
  stop?: string[]
  logprobs?: boolean
  top_logprobs?: number
  response_format?: ResponseFormat
  session_id?: string
  cache_prompt?: boolean
  device?: string
  fallback?: boolean
  tools?: Tool[]
  enable_thinking?: boolean
  max_reasoning_tokens?: number
}

export interface TopToken {
  token: string
  logprob: number
}

export interface Logprob {
  token: string
  logprob: number
  top_logprobs?: TopToken[]
}

/** One streamed piece. Reasoning and content never share a delta. */
export interface Delta {
  content?: string
  reasoning_content?: string
  tool_calls?: ToolCall[]
  logprobs?: Logprob[]
}

export interface Usage {
  prompt_tokens: number
  cached_tokens: number
  completion_tokens: number
  reasoning_tokens: number
}

/** The `stats` object every response carries (SPEC §12). */
export interface Stats {
  stages_ms: Record<string, number>
  ttft_ms: number
  prefill_tok_s: number
  decode_tok_s: number
  prompt_tokens: number
  cached_tokens: number
  completion_tokens: number
  reasoning_tokens: number
  backend: string
  device: string
  preset?: string
  mode?: string
  /** Share of drafted tokens accepted; absent when no drafter ran. */
  draft_acceptance?: number
  slot: number
}

export interface Warning {
  code: string
  message: string
}

export type FinishReason = 'stop' | 'length' | 'tool_calls' | (string & {})

export interface ChatResult {
  message: Message
  finish_reason: FinishReason
  logprobs?: Logprob[]
  usage: Usage
  stats: Stats
  warnings?: Warning[]
}

export interface TokenizeResult {
  tokens: number[]
  count: number
}

// --- OpenAI dialect -----------------------------------------------------------

export interface OpenAIToolCall {
  index?: number
  id?: string
  type?: string
  function: { name?: string; arguments: string }
}

export interface OpenAIChunk {
  id: string
  object: string
  model: string
  choices: {
    index: number
    delta?: {
      role?: string
      content?: string | null
      reasoning_content?: string
      tool_calls?: OpenAIToolCall[]
    }
    logprobs?: { content: { token: string; logprob: number; top_logprobs?: TopToken[] }[] } | null
    finish_reason: string | null
  }[]
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    prompt_tokens_details?: { cached_tokens: number }
    completion_tokens_details?: { reasoning_tokens: number }
  }
  stats?: Stats
  warnings?: Warning[]
}

// --- models and devices -------------------------------------------------------

export type DiskState = 'absent' | 'partial' | 'complete' | 'corrupt'

/** One resident instance, as the pool reports it. Loosely typed on purpose. */
export interface LoadedInstance {
  key?: {
    model?: string
    revision?: string
    backend?: string
    device?: string
    n_ctx?: number
    mode?: string
    preset?: string
  }
  refs?: number
  busy?: number
  pinned?: boolean
  retired?: boolean
  loaded_at?: string
  last_use?: string
  info?: Record<string, unknown>
}

export interface ModelInfo {
  id: string
  family?: string
  params?: string
  quant?: string
  disk: DiskState
  size_bytes: number
  pinned: boolean
  loaded?: LoadedInstance[]
  /** The placement the model loads with by default, when the planner has one. */
  plan?: unknown
  default?: boolean
}

export interface CatalogFile {
  role: string
  repo?: string
  path: string
  revision?: string
  size: number
  approx_size?: string
}

export interface CatalogEntry {
  id: string
  family?: string
  arch?: string
  params?: string
  quant?: string
  role?: string
  license?: string
  commercial?: string
  files?: CatalogFile[]
  reasoning?: { thinking_default?: boolean }
  drafter?: unknown
}

export interface DownloadProgress {
  file: string
  downloaded: number
  total: number
  percent: number
}

export interface DeviceInfo {
  selector: string
  backend: string
  id: string
  type: string
  name: string
  memory_total: number
  memory_free: number
  unified?: boolean
  verified?: string[]
}

// --- jobs (planned API; the routes answer 501 until the queue lands) ---------

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled' | 'expired'

export const TERMINAL: readonly JobStatus[] = ['succeeded', 'failed', 'canceled', 'expired']

export function isTerminal(status: JobStatus): boolean {
  return TERMINAL.includes(status)
}

export interface JobProgress {
  done: number
  total: number
  failed?: number
}

export interface JobItemResult {
  /** The request's own id from the JSONL line, when it had one. */
  custom_id?: string
  index: number
  result?: ChatResult
  error?: { code: string; message: string }
}

export interface Job {
  id: string
  status: JobStatus
  model?: string
  created_at: string
  started_at?: string
  finished_at?: string
  progress?: JobProgress
  error?: { code: string; message: string }
  results?: JobItemResult[]
}

export interface JobPage {
  data: Job[]
  next_cursor?: string
}

export interface JobFilter {
  status?: JobStatus
  cursor?: string
  limit?: number
}

// --- bench (docs/BENCHMARKS.md; the routes land later) -----------------------

export type BenchSuite = 'standard' | 'dev'

export interface BenchRequest {
  model?: string
  suite: BenchSuite
  /** quick: auto and two representative presets; full: every preset that fits. */
  quick: boolean
}

export type BenchRunStatus = 'ok' | 'skipped' | 'refused' | 'failed'

/** One preset's measurements, as nanollm.bench/v1 carries them. */
export interface BenchRun {
  preset: string
  plan_hash?: string
  plan?: Record<string, unknown>
  metrics: {
    load_ms?: number
    pp_tok_s?: number
    tg_tok_s?: { d0?: number; dN?: number; depth_n?: number }
    ttft_ms?: number
    chat_tok_s?: number
    mtp?: { acceptance: number; speedup: number }
    peak_rss?: number
    peak_vram?: number
  }
  reps: number
  status: BenchRunStatus
  reason?: string
  same_as?: string
}

/** The nanollm.bench/v1 document; only what the screen reads is typed. */
export interface BenchResult {
  schema: string
  consent_version?: string
  suite: { id: string; version: string; prompts_sha256: string; quick?: boolean }
  model: { id: string; revision: string }
  runs: BenchRun[]
  date: string
  duration_s: number
  [extra: string]: unknown
}

export interface BenchStarted {
  id: string
  estimate_s?: number
  presets?: string[]
}

/** Events of GET /api/v1/bench/{id}/events. */
export interface BenchProgress {
  preset: string
  scenario?: string
  status: 'pending' | 'running' | BenchRunStatus
  percent?: number
  run?: BenchRun
}

export interface BenchSubmitted {
  id: string
  url?: string
}
