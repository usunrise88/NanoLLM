import { expect, MODEL, sse, test } from './fixtures'

/** Devices, Settings, Jobs and Bench: what each shows on the fake server. */

test('devices lists both fake devices and what is resident where', async ({ page }) => {
  await page.goto('/ui/devices')
  await expect(page.getByText('Backend fake')).toBeVisible()
  const gpu = page.locator('[data-item="fake:gpu0"]')
  await expect(gpu).toContainText('Fake GPU')
  await expect(gpu).toContainText('8.0 GiB')
  await expect(gpu).toContainText(MODEL)
  await expect(page.locator('[data-item="fake:cpu"]')).toContainText('Fake CPU')
})

test('settings shows the effective configuration and switches language', async ({ page }) => {
  await page.goto('/ui/settings')
  const config = page.getByLabel('Effective configuration')
  await expect(config).toContainText('"enabled": [')
  await expect(config).toContainText('"fake"')
  await expect(page.getByText('llama.cpp')).toBeVisible()

  await page.getByLabel('Interface language').selectOption('ru')
  await expect(page.getByRole('link', { name: 'Модели' })).toBeVisible()
  await expect(page.locator('html')).toHaveAttribute('lang', 'ru')
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Настройки' })).toBeVisible()
})

test('jobs says the queue is not in this build yet', async ({ page }) => {
  await page.goto('/ui/jobs')
  await expect(page.getByText('The job queue is not in this build yet')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Enqueue' })).toBeDisabled()
})

/**
 * The jobs screen against the planned API, played back: the routes answer
 * 501 today, and this is the test that the screen will work the day they
 * do not.
 */
test('jobs against the planned API: submit, follow, cancel, results', async ({ page }) => {
  const created = '2026-10-18T10:00:00Z'
  const jobs = [
    { id: 'job_run', status: 'running', model: MODEL, created_at: created, progress: { done: 1, total: 4 } },
    { id: 'job_ok', status: 'succeeded', model: MODEL, created_at: created, progress: { done: 2, total: 2 } },
  ]
  let submitted = ''
  await page.route('**/api/v1/jobs**', async (route) => {
    const req = route.request()
    const path = new URL(req.url()).pathname
    if (req.method() === 'POST' && path === '/api/v1/jobs') {
      submitted = req.postData() ?? ''
      return route.fulfill({ json: { id: 'job_new', status: 'queued', created_at: created } })
    }
    if (req.method() === 'GET' && path === '/api/v1/jobs') return route.fulfill({ json: { data: jobs } })
    if (path === '/api/v1/jobs/job_run/events') {
      return route.fulfill({
        headers: { 'Content-Type': 'text/event-stream' },
        body: sse([['progress', { ...jobs[0], progress: { done: 3, total: 4 } }]]),
      })
    }
    if (req.method() === 'DELETE' && path === '/api/v1/jobs/job_run') {
      return route.fulfill({ json: { ...jobs[0], status: 'canceled' } })
    }
    if (path === '/api/v1/jobs/job_ok') {
      return route.fulfill({ json: { ...jobs[1], results: [{ index: 0, custom_id: 'a' }, { index: 1, custom_id: 'b' }] } })
    }
    return route.fulfill({ status: 404, body: 'not found' })
  })

  await page.goto('/ui/jobs')
  const running = page.locator('[data-item="job_run"]')
  await expect(running).toContainText('running')
  await expect(running).toContainText('3 of 4')

  await page.getByRole('button', { name: 'Example' }).click()
  await expect(page.getByText('Requests: 2')).toBeVisible()
  await page.getByRole('button', { name: 'Enqueue' }).click()
  await expect(page.getByText('Job job_new queued')).toBeVisible()
  expect(submitted.trim().split('\n')).toHaveLength(2)

  await running.getByRole('button', { name: 'Cancel' }).click()
  await expect(running).toContainText('canceled')

  const download = page.waitForEvent('download')
  await page.locator('[data-item="job_ok"]').getByRole('button', { name: 'Results (JSONL)' }).click()
  expect((await download).suggestedFilename()).toBe('job_ok.jsonl')
})

test('a bad JSONL line is pointed out before anything is sent', async ({ page }) => {
  await page.goto('/ui/jobs')
  await page.getByLabel('Requests (JSONL)').fill('{"messages":[]}\n{"messages":')
  await expect(page.getByText(/^Line 2:/)).toBeVisible()
})

test('bench says it is not available yet on this server', async ({ page }) => {
  await page.goto('/ui/bench')
  await page.getByRole('button', { name: 'Run benchmark' }).click()
  await expect(page.getByText('Benchmarks are not available in this build yet')).toBeVisible()
})

/** The bench screen against the contract of docs/BENCHMARKS.md, played back. */
test('bench against the planned API: progress, results, consent, submit', async ({ page }) => {
  const result = {
    schema: 'nanollm.bench/v1',
    suite: { id: 'standard', version: 'v1', prompts_sha256: 'a'.repeat(64), quick: true },
    client: { nanollm: 'dev', llama_cpp: { tag: 'b1', commit: 'c' }, os: 'linux', arch: 'amd64' },
    model: { id: MODEL, revision: 'r' },
    machine: { cpu: { vendor: 'Fake', model: 'Fake CPU', cores: 8, threads: 16 } },
    runs: [
      { preset: 'auto', metrics: { load_ms: 820, pp_tok_s: 512, tg_tok_s: { d0: 21.5, dN: 18 }, ttft_ms: 95, chat_tok_s: 20, mtp: { acceptance: 0.62, speedup: 1.4 }, peak_rss: 3 << 30 }, reps: 3, status: 'ok' },
      { preset: 'vram-3g', metrics: { tg_tok_s: {} }, reps: 0, status: 'skipped', reason: 'no GPU' },
    ],
    date: '2026-10-18',
    duration_s: 312,
  }
  let started: unknown
  let submit: unknown
  await page.route('**/api/v1/bench**', async (route) => {
    const req = route.request()
    const path = new URL(req.url()).pathname
    if (path === '/api/v1/bench') {
      started = req.postDataJSON()
      return route.fulfill({ json: { id: 'b1', estimate_s: 600, presets: ['auto', 'vram-3g'] } })
    }
    if (path === '/api/v1/bench/b1/events') {
      return route.fulfill({
        headers: { 'Content-Type': 'text/event-stream' },
        body: sse([
          ['progress', { preset: 'auto', status: 'running', scenario: 'pp512', percent: 30 }],
          ['progress', { preset: 'auto', status: 'ok', percent: 100 }],
          ['progress', { preset: 'vram-3g', status: 'skipped' }],
          ['result', result],
          ['done', {}],
        ]),
      })
    }
    if (path === '/api/v1/bench/b1/submit') {
      submit = req.postDataJSON()
      return route.fulfill({ json: { id: 's_42', url: 'https://nano.llmto.day/benchmarks/s/s_42' } })
    }
    return route.fulfill({ status: 404, body: 'not found' })
  })

  await page.goto('/ui/bench')
  await page.getByLabel('Presets').selectOption('full')
  await page.getByRole('button', { name: 'Run benchmark' }).click()

  const table = page.locator('[data-results]')
  await expect(table.locator('[data-run="auto"]')).toContainText('21.5 tok/s')
  await expect(table.locator('[data-run="auto"]')).toContainText('62%')
  await expect(table.locator('[data-run="vram-3g"]')).toContainText('skipped')
  expect(started).toEqual({ suite: 'standard', quick: false })

  // The consent preview is the exact document, consent version included.
  const json = page.getByLabel('The exact document')
  await expect(json).toContainText('"schema": "nanollm.bench/v1"')
  await expect(json).toContainText('"consent_version": "2026-10"')

  await page.getByText('Anonymous', { exact: true }).click()
  await page.getByRole('button', { name: 'Submit', exact: true }).click()
  await expect(page.getByText('Submitted as s_42.')).toBeVisible()
  expect(submit).toEqual({ consent_version: '2026-10', anonymous: true })
})
