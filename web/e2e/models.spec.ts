import { expect, MODEL, sse, test } from './fixtures'

/**
 * The model lifecycle against the real pool: the fake backend loads and
 * unloads instantly but goes through the same pool, pinning and hot swap.
 * Downloads are the exception — the catalog points at real weights — so the
 * download stream is played back from a recording of its shape.
 */

test.beforeEach(async ({ page }) => {
  await page.goto('/ui/models')
})

test('shows the catalog model with its disk state and resident instance', async ({ page }) => {
  const card = page.locator(`[data-item="${MODEL}"]`)
  await expect(card).toContainText('not downloaded')
  await expect(card).toContainText('loaded: 1')
  await expect(card).toContainText('default')
  await expect(card.locator('[data-instance="fake:gpu0"]')).toContainText('n_ctx')
})

test('unload, load, pin and hot swap', async ({ page }) => {
  const card = page.locator(`[data-item="${MODEL}"]`)

  await card.getByRole('button', { name: 'Unload', exact: true }).click()
  await page.getByRole('dialog').getByRole('button', { name: 'Unload' }).click()
  await expect(card.getByRole('button', { name: 'Load', exact: true })).toBeVisible()
  await expect(card).not.toContainText('loaded: 1')

  await card.getByRole('button', { name: 'Load', exact: true }).click()
  await expect(card).toContainText('loaded: 1')

  await card.getByRole('button', { name: 'Pin', exact: true }).click()
  await expect(card).toContainText('pinned')
  await card.getByRole('button', { name: 'Unpin' }).click()
  await expect(card.getByRole('button', { name: 'Pin', exact: true })).toBeVisible()

  await card.getByRole('button', { name: 'Reload' }).click()
  await expect(page.getByText(`${MODEL} reloaded`)).toBeVisible()
  await expect(card).toContainText('loaded: 1')
})

test('a download reports progress over SSE and finishes', async ({ page }) => {
  let method = ''
  await page.route('**/api/v1/models/*/download', async (route) => {
    method = route.request().method()
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
      body:
        ': keep-alive\n\n' +
        sse([
          ['progress', { file: 'model.gguf', downloaded: 1 << 20, total: 4 << 20, percent: 25 }],
          ['progress', { file: 'model.gguf', downloaded: 4 << 20, total: 4 << 20, percent: 100 }],
          ['done', { id: MODEL }],
        ]),
    })
  })
  const card = page.locator(`[data-item="${MODEL}"]`)
  await card.getByRole('button', { name: 'Download' }).click()
  await expect(card).toContainText('Downloaded')
  expect(method).toBe('POST')
})

test('a failed download shows the server reason', async ({ page }) => {
  await page.route('**/api/v1/models/*/download', (route) =>
    route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
      body: sse([['error', { status: 503, code: 'model_unavailable', detail: 'download gemma: no route to host' }]]),
    }),
  )
  const card = page.locator(`[data-item="${MODEL}"]`)
  await card.getByRole('button', { name: 'Download' }).click()
  await expect(card.getByRole('alert')).toContainText('no route to host')
})
