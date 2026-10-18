import { expect, test as base, type Page } from '@playwright/test'

/**
 * Shared setup: every test starts from a clean browser profile (conversations
 * and chat options live in localStorage) and fails on any console error, the
 * cheapest way to catch a React warning or an unhandled rejection.
 *
 * Failed requests are not console errors here: several screens are built
 * ahead of their backend and expect a 501 or 404, and Chromium logs every
 * non-2xx resource as "Failed to load resource".
 */
export const AUTH_URL =
  process.env.NANOLLM_E2E_AUTH_URL ?? (process.env.NANOLLM_E2E_URL ? undefined : 'http://127.0.0.1:18091')
export const AUTH_KEY = process.env.NANOLLM_E2E_KEY ?? 'sk-e2e-key-0123456789'

export const MODEL = 'gemma-4-e4b-qat'

export const test = base.extend<{ errors: string[] }>({
  errors: [
    async ({ page }, use) => {
      const errors: string[] = []
      page.on('pageerror', (e) => errors.push(String(e)))
      page.on('console', (m) => {
        if (m.type() === 'error' && !m.text().startsWith('Failed to load resource')) errors.push(m.text())
      })
      await use(errors)
      expect(errors, `console errors: ${errors.join(' | ')}`).toHaveLength(0)
    },
    { auto: true },
  ],
})

export { expect }

/** A server-sent events body, as the server frames it. */
export function sse(events: [string, unknown][]): string {
  return events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join('')
}

/** Types a message into the composer and sends it with Enter. */
export async function say(page: Page, text: string): Promise<void> {
  const box = page.getByRole('textbox', { name: 'Message' })
  await box.fill(text)
  await box.press('Enter')
}

/** The last assistant turn on the page. */
export function lastAnswer(page: Page) {
  return page.locator('[data-role="assistant"]').last()
}
