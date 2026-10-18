import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { defineConfig } from '@playwright/test'

/**
 * The end-to-end suite runs the real binary with the SPA embedded, on the
 * scripted fake backend: no weights, no GPU, and answers chosen by
 * directives in the prompt ([[fake:think]], [[fake:tool NAME ARGS]], …).
 *
 * Two servers, because "the SPA asks for a key when the server wants one"
 * cannot be tested against a server that does not:
 *
 *   - open mode on OPEN_PORT, for everything else;
 *   - apikey mode on AUTH_PORT with one non-admin key, E2E_KEY.
 *
 * NANOLLM_BIN is the binary (default ../dist/nanollm, which `make build`
 * writes after `make web`). NANOLLM_E2E_URL points the suite at a server
 * that is already running instead; the auth tests then skip unless
 * NANOLLM_E2E_AUTH_URL is set too.
 *
 * PLAYWRIGHT_CHROMIUM is a Chromium to use instead of Playwright's own, for
 * machines where one is preinstalled.
 */
const here = dirname(fileURLToPath(import.meta.url))
const bin = process.env.NANOLLM_BIN ?? resolve(here, '../dist/nanollm')
const OPEN_PORT = 18090
const AUTH_PORT = 18091
const E2E_KEY = 'sk-e2e-key-0123456789' // also in e2e/fixtures.ts

const external = process.env.NANOLLM_E2E_URL
const data = external ? '' : mkdtempSync(join(tmpdir(), 'nanollm-e2e-'))

function server(port: number, env: Record<string, string>) {
  return {
    command: `"${bin}" serve -addr 127.0.0.1:${port}`,
    url: `http://127.0.0.1:${port}/healthz`,
    reuseExistingServer: false,
    timeout: 30_000,
    stdout: 'ignore' as const,
    stderr: 'pipe' as const,
    env: {
      NANOLLM_BACKENDS_ENABLED: 'fake',
      NANOLLM_SERVER_DATA_DIR: join(data, String(port)),
      NANOLLM_LOG_LEVEL: 'warn',
      ...env,
    },
  }
}

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  expect: { timeout: 10_000 },
  // One server, shared state (loaded models, pins): tests run one at a time.
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: external ?? `http://127.0.0.1:${OPEN_PORT}`,
    trace: 'retain-on-failure',
    locale: 'en-US',
    launchOptions: {
      ...(process.env.PLAYWRIGHT_CHROMIUM ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM } : {}),
    },
  },
  ...(external
    ? {}
    : {
        webServer: [
          server(OPEN_PORT, { NANOLLM_AUTH_MODE: 'open' }),
          server(AUTH_PORT, { NANOLLM_AUTH_MODE: 'apikey', NANOLLM_AUTH_KEYS: E2E_KEY }),
        ],
      }),
})
