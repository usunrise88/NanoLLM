# web/

The single-page app served at `/ui/` (SPEC §11): Chat, Jobs, Models, Devices,
Bench and Settings. It is built only on the public API, the same one any
client uses, and asks for an API key on the first 401 (kept in
`localStorage`).

The stack and scaffold are NanoASR's: React 19, Vite, TanStack Router and
Query, Tailwind 4 on Radix Colors tokens, Base UI wrapped once in
`src/components/ui`, i18next with typed keys (ru/en), vitest, Playwright.
ESLint holds the layout rules: routes compose `Page/Section/Stack/Inline/Grid`
and never lay themselves out, every route exports `pageMeta`, motion lives in
`styles/motion.css`.

## Build

`make web` builds it (Node 22+) and stages `dist/` in `internal/ui/dist/`,
which `internal/ui` embeds at compile time; `make build` then produces a
binary with `ui true` in `nanollm version`. Without `make web` the binary
embeds the committed placeholder and serves no UI.

For development, `npm run dev` serves the app with hot reload and proxies
`/api`, `/v1` and `/healthz` to a server on `127.0.0.1:8080`
(`make run-fake` starts one on the scripted fake backend).

## Tests

- `npm test`: unit tests of the API client, the SSE parser, both chat
  stream dialects, request building and the bench reducer.
- `npm run test:e2e`: Playwright against the real binary on the fake
  backend. `playwright.config.ts` starts two servers (open mode, and apikey
  mode with one non-admin key); `NANOLLM_BIN` names the binary,
  `PLAYWRIGHT_CHROMIUM` a preinstalled Chromium, `NANOLLM_E2E_URL` an
  already-running server instead. `make e2e` builds everything and runs it.

Answers are scripted with directives in the prompt (see
`internal/backend/fake/scenario.go`): `[[fake:echo]]`, `[[fake:think]]`,
`[[fake:tool NAME {a:1}]]`, `[[fake:json]]`, `[[fake:loop]]`,
`[[fake:error]]`.

## Endpoints

| Screen | Uses |
| --- | --- |
| Chat | `POST /api/v1/chat` (SSE `delta`/`done`/`error`) or `POST /v1/chat/completions` (chunks, `[DONE]`); `POST /api/v1/tokenize`; one `session_id` per conversation |
| Models | `GET /api/v1/models`, `GET /api/v1/catalog`, `POST /api/v1/models/{id}/download` (SSE `progress`/`done`/`error`), `…/load`, `…/unload`, `…/reload`, `…/pin {"pin": bool}` |
| Devices | `GET /api/v1/devices`, `GET /api/v1/models` |
| Settings | `GET /api/v1/config` (admin), `GET /healthz` |
| Jobs | `POST /api/v1/jobs` (JSONL body, `?model=`), `GET /api/v1/jobs?status=&limit=&cursor=` → `{data, next_cursor}`, `GET /api/v1/jobs/{id}`, `GET /api/v1/jobs/{id}/events` (SSE, resumable), `DELETE /api/v1/jobs/{id}` |
| Bench | `POST /api/v1/bench {model?, suite, quick}` → `{id, estimate_s?, presets?}`, `GET /api/v1/bench/{id}/events` (SSE `progress {preset, status, scenario?, percent?}`, `result` = the `nanollm.bench/v1` document, `done`/`error`), `POST /api/v1/bench/{id}/submit {consent_version, anonymous}` → `{id, url?}` |

The job routes answer 501 until the queue lands, and the bench routes do not
exist yet; both screens show a "not in this build yet" state for 501 and a
bare 404. The shapes in the last two rows are what the screens are written
against (job fields follow NanoASR's queue; bench follows
docs/BENCHMARKS.md), and the e2e suite plays them back to keep the screens
honest until the server side lands. The submit route and its body are this
UI's proposal: BENCHMARKS.md names only the first two bench routes.
