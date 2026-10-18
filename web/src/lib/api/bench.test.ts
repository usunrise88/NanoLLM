import { describe, expect, it } from 'vitest'

import { benchReducer, CONSENT_VERSION, initialBench, submissionJSON } from './bench'
import type { BenchResult } from './types'

const result: BenchResult = {
  schema: 'nanollm.bench/v1',
  suite: { id: 'standard', version: 'v1', prompts_sha256: 'a'.repeat(64), quick: true },
  model: { id: 'gemma-4-e4b-qat', revision: 'r' },
  runs: [
    { preset: 'auto', metrics: { tg_tok_s: { d0: 20 } }, reps: 3, status: 'ok' },
    { preset: 'vram-3g', metrics: { tg_tok_s: {} }, reps: 0, status: 'skipped', reason: 'no GPU', same_as: '' },
    { preset: 'ram-4g', metrics: { tg_tok_s: {} }, reps: 3, status: 'ok', same_as: 'auto' },
  ],
  date: '2026-10-18',
  duration_s: 312,
}

describe('benchReducer', () => {
  it('starts with the presets the server announced, all pending', () => {
    const s = initialBench({ id: 'b1', estimate_s: 600, presets: ['auto', 'vram-3g'] })
    expect(s).toMatchObject({ id: 'b1', estimateS: 600, finished: false })
    expect(s.presets.map((p) => p.status)).toEqual(['pending', 'pending'])
  })

  it('updates a preset in place and adds one it did not know', () => {
    let s = initialBench({ id: 'b1', presets: ['auto'] })
    s = benchReducer(s, 'progress', { preset: 'auto', status: 'running', scenario: 'pp512', percent: 40 })
    s = benchReducer(s, 'progress', { preset: 'cpu-only', status: 'pending' })
    expect(s.presets).toEqual([
      { preset: 'auto', status: 'running', scenario: 'pp512', percent: 40 },
      { preset: 'cpu-only', status: 'pending' },
    ])
  })

  // The document is the authority: a preset deduplicated by plan hash never
  // ran on its own, and shows up only there.
  it('takes final statuses from the result document', () => {
    let s = initialBench({ id: 'b1', presets: ['auto', 'vram-3g'] })
    s = benchReducer(s, 'result', result)
    expect(s.result).toBe(result)
    expect(s.presets.map((p) => [p.preset, p.status])).toEqual([
      ['auto', 'ok'],
      ['vram-3g', 'skipped'],
      ['ram-4g', 'ok'],
    ])
  })

  it('ends on done and on error', () => {
    expect(benchReducer(initialBench(), 'done', {}).finished).toBe(true)
    const failed = benchReducer(initialBench(), 'error', { status: 500, detail: 'worker crashed' })
    expect(failed).toMatchObject({ finished: true, error: 'worker crashed' })
  })

  it('ignores events it does not know', () => {
    const s = initialBench()
    expect(benchReducer(s, 'heartbeat', {})).toBe(s)
  })
})

describe('submissionJSON', () => {
  // The consent preview shows this string, and it is what the person agrees
  // to: it must carry the consent version and nothing the result lacks.
  it('is the result plus the consent version', () => {
    const doc = JSON.parse(submissionJSON(result)) as Record<string, unknown>
    expect(doc.consent_version).toBe(CONSENT_VERSION)
    expect({ ...doc, consent_version: undefined }).toEqual({ ...result, consent_version: undefined })
  })
})
