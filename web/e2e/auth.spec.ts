import { AUTH_KEY, AUTH_URL, expect, lastAnswer, MODEL, say, test } from './fixtures'

/**
 * What the SPA does when the server wants an API key (SPEC §11). There is no
 * flag telling it: the first 401 is the answer. Runs against the second
 * server, in apikey mode with one non-administrative key.
 */
test.skip(!AUTH_URL, 'set NANOLLM_E2E_AUTH_URL and NANOLLM_E2E_KEY')

test('asks for a key on the first 401, then works', async ({ page }) => {
  await page.goto(`${AUTH_URL}/ui/`)

  // The sheet opens itself off the back of the first refused request.
  await expect(page.getByText('This server wants a key')).toBeVisible()
  await page.getByPlaceholder('sk-…').fill(AUTH_KEY)
  await page.getByRole('button', { name: 'Save' }).click()
  await expect(page.getByText('This server wants a key')).toHaveCount(0)

  // Only possible if the refused queries were retried with the key.
  await expect(page.getByRole('combobox', { name: 'Model' })).toHaveValue(MODEL)
  await say(page, 'Authenticated [[fake:echo]]')
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('Authenticated')

  // The key survives a reload: it is kept in this browser.
  await page.reload()
  await expect(page.getByText('This server wants a key')).toHaveCount(0)
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('Authenticated')
})

test('admin-only views say why they are empty', async ({ page }) => {
  await page.goto(`${AUTH_URL}/ui/settings`)
  await page.evaluate((key) => localStorage.setItem('nanollm.apikey', key), AUTH_KEY)
  await page.reload()
  await expect(page.getByText('The configuration needs an administrative key')).toBeVisible()

  await page.goto(`${AUTH_URL}/ui/models`)
  const card = page.locator(`[data-item="${MODEL}"]`)
  await card.getByRole('button', { name: 'Pin', exact: true }).click()
  await expect(page.getByText('The action did not go through')).toBeVisible()
  await expect(page.getByText('this endpoint needs an administrative key')).toBeVisible()
})

test('a wrong key is refused visibly, not as an empty list', async ({ page }) => {
  await page.goto(`${AUTH_URL}/ui/settings`)
  await page.evaluate(() => localStorage.setItem('nanollm.apikey', 'sk-definitely-not-the-key'))
  await page.goto(`${AUTH_URL}/ui/models`)
  await expect(page.getByText('a valid API key is required')).toBeVisible()
})
