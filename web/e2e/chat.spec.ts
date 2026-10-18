import { expect, lastAnswer, MODEL, say, test } from './fixtures'

/**
 * Chat against the fake backend. Each prompt carries a directive that picks
 * the scripted answer, so what the screen must show is known exactly.
 */

test.beforeEach(async ({ page }) => {
  await page.goto('/ui/')
  await expect(page.getByRole('combobox', { name: 'Model' })).toHaveValue(MODEL)
})

test('streams an answer and shows what the server measured', async ({ page }) => {
  await say(page, 'Hello there [[fake:echo]]')

  const answer = lastAnswer(page)
  await expect(answer.locator('[data-content]')).toHaveText('Hello there')
  const stats = answer.locator('[data-stats]')
  await expect(stats.locator('[data-stat="ttft"]')).toContainText('TTFT')
  await expect(stats.locator('[data-stat="decode"]')).toContainText('tok/s')
  await expect(stats.locator('[data-stat="cached"]')).toContainText('/')
  await expect(stats.locator('[data-stat="device"]')).toContainText('fake:gpu0')
  // The fake loads with its drafter, so acceptance is reported.
  await expect(stats.locator('[data-stat="draft"]')).toContainText('%')
  // The conversation is named after its first message, directive stripped.
  await expect(page.locator('[data-conversations]')).toContainText('Hello there')
})

test('reasoning streams into a block that folds away once the answer starts', async ({ page }) => {
  await page.getByText('Thinking', { exact: true }).click()
  await say(page, 'Why? [[fake:think]]')

  const answer = lastAnswer(page)
  await expect(answer.locator('[data-content]')).toHaveText('Here is a short answer.')
  const toggle = answer.locator('[data-reasoning] button')
  await expect(toggle).toHaveText('Reasoning')
  await expect(toggle).toHaveAttribute('aria-expanded', 'false')
  await toggle.click()
  await expect(answer.locator('[data-reasoning]')).toContainText('The user wants an answer; give a short one.')
  await expect(answer.locator('[data-stat="tokens"]')).toContainText('reasoning')
})

test('a tool call is shown, answered and continued', async ({ page }) => {
  await page.getByRole('button', { name: 'Request options' }).click()
  await page
    .getByLabel('Tools')
    .fill('[{"name":"get_weather","description":"Weather for a city","parameters":{"type":"object"}}]')
  await page.getByRole('button', { name: 'Close' }).first().click()

  await say(page, 'Weather? [[fake:tool get_weather {a:1}]]')

  const call = lastAnswer(page).locator('[data-tool-call="get_weather"]')
  await expect(call).toContainText('"a": 1')
  await expect(lastAnswer(page).locator('[data-stat="finish"]')).toContainText('tool_calls')

  await page.getByRole('textbox', { name: 'Result of get_weather' }).fill('{"temperature": 21}')
  await page.getByRole('button', { name: 'Send results' }).click()

  await expect(page.locator('[data-role="tool"]')).toContainText('{"temperature": 21}')
  // The model answers again after the tool result; the fake calls the tool
  // again because the directive is still in the prompt.
  await expect(page.locator('[data-role="assistant"]')).toHaveCount(2)
})

test('stop ends an endless generation and keeps what arrived', async ({ page, browserName }) => {
  test.skip(browserName !== 'chromium', 'throttles the network through CDP')
  // The fake decodes the whole token cap in a fraction of a second; a slow
  // link keeps the stream open long enough for a person to press Stop.
  const cdp = await page.context().newCDPSession(page)
  await cdp.send('Network.enable')
  const throttle = (bytesPerSecond: number) =>
    cdp.send('Network.emulateNetworkConditions', {
      offline: false,
      latency: 0,
      downloadThroughput: bytesPerSecond,
      uploadThroughput: -1,
    })
  await throttle(2_000)

  await say(page, 'Go on forever [[fake:loop]]')

  await expect(lastAnswer(page).locator('[data-content]')).toContainText('la la la')
  await page.getByRole('button', { name: 'Stop' }).click()

  await expect(lastAnswer(page)).toContainText('Stopped')
  await expect(page.getByRole('button', { name: 'Send' })).toBeVisible()
  await throttle(-1)
  // The slot is free again: the next message is served.
  await say(page, 'And now? [[fake:echo]]')
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('And now?')
})

test('one session id per conversation, and the prefix comes from the cache', async ({ page }) => {
  const sessions: string[] = []
  page.on('request', (r) => {
    if (r.url().endsWith('/api/v1/chat') && r.method() === 'POST') {
      sessions.push((r.postDataJSON() as { session_id: string }).session_id)
    }
  })

  await say(page, 'First [[fake:echo]]')
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('First')
  await say(page, 'Second [[fake:echo]]')
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('Second')

  expect(sessions).toHaveLength(2)
  expect(sessions[0]).toBe(sessions[1])
  await expect(page.locator(`[data-session="${sessions[0]}"]`)).toBeVisible()
  // The second turn's prompt starts with the first turn: most of it is cached.
  const cached = await lastAnswer(page).locator('[data-stat="cached"] dd').textContent()
  const [hit, total] = (cached ?? '0 / 0').split(' / ').map(Number)
  expect(hit).toBeGreaterThan(0)
  expect(hit).toBeLessThanOrEqual(total ?? 0)

  await page.getByRole('button', { name: 'New chat' }).click()
  await say(page, 'Elsewhere [[fake:echo]]')
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('Elsewhere')
  expect(sessions).toHaveLength(3)
  expect(sessions[2]).not.toBe(sessions[0])
})

test('the OpenAI dialect gives the same answer and stats', async ({ page }) => {
  await page.getByRole('button', { name: 'Request options' }).click()
  await page.getByLabel('API dialect').selectOption('openai')
  await page.getByRole('button', { name: 'Close' }).first().click()

  const urls: string[] = []
  page.on('request', (r) => {
    if (r.method() === 'POST') urls.push(new URL(r.url()).pathname)
  })
  await say(page, 'Via OpenAI [[fake:echo]]')

  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('Via OpenAI')
  await expect(lastAnswer(page).locator('[data-stat="dialect"]')).toContainText('OpenAI')
  await expect(lastAnswer(page).locator('[data-stat="device"]')).toContainText('fake:gpu0')
  expect(urls).toContain('/v1/chat/completions')
})

test('a failed generation says so and the conversation goes on', async ({ page }) => {
  await say(page, 'Break [[fake:error]]')
  await expect(lastAnswer(page).getByRole('alert')).toContainText('Generation failed')
  await say(page, 'Fine now [[fake:echo]]')
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('Fine now')
})

test('conversations survive a reload', async ({ page }) => {
  await say(page, 'Remember me [[fake:echo]]')
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('Remember me')
  await page.reload()
  await expect(page.locator('[data-conversations]')).toContainText('Remember me')
  await expect(lastAnswer(page).locator('[data-content]')).toHaveText('Remember me')
  await expect(lastAnswer(page).locator('[data-stat="ttft"]')).toBeVisible()
})

test('JSON mode returns a JSON object', async ({ page }) => {
  await page.getByRole('button', { name: 'Request options' }).click()
  await page.getByLabel('Response format').selectOption('json_object')
  await page.getByRole('button', { name: 'Close' }).first().click()
  await say(page, 'Give me JSON [[fake:json]]')
  await expect(lastAnswer(page).locator('[data-stat="ttft"]')).toBeVisible()
  const text = await lastAnswer(page).locator('[data-content]').textContent()
  expect(() => JSON.parse(text ?? '')).not.toThrow()
})
