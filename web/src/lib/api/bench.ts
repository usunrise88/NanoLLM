import type { BenchProgress, BenchResult, BenchRun, BenchStarted } from './types'

/**
 * The benchmark run as the screen follows it (docs/BENCHMARKS.md).
 *
 * POST /api/v1/bench starts a run and answers its id, the presets the planner
 * picked and an estimate; GET /api/v1/bench/{id}/events streams "progress"
 * per preset, then one "result" with the nanollm.bench/v1 document, then
 * "done" (or "error"). Nothing is sent anywhere until the person has read
 * that document and pressed submit — the screen shows the exact JSON.
 */

/** The consent text's version; it travels with a submission (docs/BENCHMARKS.md). */
export const CONSENT_VERSION = 'draft-1'

export interface PresetState {
  preset: string
  status: BenchProgress['status']
  scenario?: string
  percent?: number
  run?: BenchRun
}

export interface BenchState {
  id?: string
  estimateS?: number
  presets: PresetState[]
  result?: BenchResult
  error?: string
  finished: boolean
}

export function initialBench(started?: BenchStarted): BenchState {
  const out: BenchState = {
    presets: (started?.presets ?? []).map((p) => ({ preset: p, status: 'pending' })),
    finished: false,
  }
  if (started?.id) out.id = started.id
  if (started?.estimate_s !== undefined) out.estimateS = started.estimate_s
  return out
}

/** Folds one event of the bench stream into the state. Pure. */
export function benchReducer(state: BenchState, event: string, payload: unknown): BenchState {
  switch (event) {
    case 'progress': {
      const p = payload as BenchProgress
      if (!p?.preset) return state
      const next: PresetState = { preset: p.preset, status: p.status }
      if (p.scenario) next.scenario = p.scenario
      if (p.percent !== undefined) next.percent = p.percent
      if (p.run) next.run = p.run
      const i = state.presets.findIndex((x) => x.preset === p.preset)
      const presets = [...state.presets]
      if (i === -1) presets.push(next)
      else presets[i] = { ...presets[i], ...next }
      return { ...state, presets }
    }
    case 'result': {
      const result = payload as BenchResult
      // The document is the authority on how each preset ended; a preset the
      // stream never mentioned (deduplicated by plan hash) appears here too.
      const presets = [...state.presets]
      for (const run of result.runs ?? []) {
        const i = presets.findIndex((x) => x.preset === run.preset)
        const p: PresetState = { preset: run.preset, status: run.status, run, percent: 100 }
        if (i === -1) presets.push(p)
        else presets[i] = p
      }
      return { ...state, result, presets }
    }
    case 'error': {
      const p = payload as { detail?: string; title?: string; message?: string }
      return { ...state, error: p?.detail ?? p?.message ?? p?.title ?? 'benchmark failed', finished: true }
    }
    case 'done':
      return { ...state, finished: true }
    default:
      return state
  }
}

/** The document exactly as it will be submitted: what the consent preview shows. */
export function submissionJSON(result: BenchResult): string {
  return JSON.stringify({ ...result, consent_version: CONSENT_VERSION }, null, 2)
}
