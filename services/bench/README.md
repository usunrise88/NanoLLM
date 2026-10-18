# benchd — the NanoLLM public benchmark service

`benchd` receives `nanollm.bench/v1` results that NanoLLM users choose to
send (`nanollm bench --submit`), keeps them, and serves the Benchmarks pages
of nano.llmto.day: medians per preset and machine, down to the laptop model.
The design is in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md) (Russian),
section "Сервис services/bench".

It is a Go module of its own (`github.com/usunrise88/nanollm/services/bench`)
that depends only on [`benchfmt`](../../benchfmt) and a pure-Go SQLite
(`modernc.org/sqlite`), so it can move to its own repository with
`git subtree split -P services/bench`; then drop the `replace` in `go.mod`
and depend on a tagged `benchfmt/v1.x.y`.

## Layout

| Path | What |
| --- | --- |
| `cmd/benchd` | the binary: `serve`, `reindex`, `backup`, `migrate`, `healthcheck` |
| `internal/ingest` | submission checks, plausibility ceiling, rate limits, cohort signals, withdrawal, reindex |
| `internal/normalize` | CPU/GPU/vendor/system names, slugs, hardware fingerprint (versioned) |
| `internal/aggregate` | median, p25/p75, MAD; one value per install and per fingerprint |
| `internal/store` | SQLite: raw submissions (immutable), moderation, aliases, derived tables, backups |
| `internal/web` | JSON API, CSV/JSON export, `/benchmarks` pages (RU/EN) |
| `internal/admin` | `/admin`: moderation queue, hide/show, aliases, vendor-verified |
| `migrations/` | numbered SQL, embedded; `schema_version` |
| `templates/` | `html/template` pages and static CSS/JS, embedded, no build step |
| `deploy/` | docker compose with Caddy (access logs without client addresses) |
| `e2e/` | benchd on a temporary database, driven like the NanoLLM client |

## Data model

- **Raw submissions are the source of truth.** The exact bytes received
  are stored once and never updated (a trigger refuses it). Moderation,
  aliases and vendor-verified installs are separate tables.
- **Everything else is derived.** `norm_submissions`, `norm_runs` and
  `aggregates` are rebuilt from the raw rows by the versioned normalizer
  (`normalize.Version`) and the alias table: `benchd reindex`, or
  automatically at start when the stored rows came from an older
  normalizer, and after every alias or badge change in `/admin`. Never edit
  them by hand.
- **Withdrawal** (`DELETE /api/v1/submissions/{id}`, signed with the
  install key) deletes the raw row and everything derived. A tombstone (id
  and day only) stays in the database and in `<db>.withdrawals`; opening a
  database re-applies the tombstone file, so a restored backup never brings
  a withdrawn submission back. `secure_delete` zeroes the freed pages.
- **Backups** are `VACUUM INTO` snapshots, `bench-<UTC time>.db`, pruned
  after the retention (`BENCHD_BACKUP_KEEP`, default 30 days): that bound is
  how long a withdrawn submission can survive in a backup.

## API

| Method and path | |
| --- | --- |
| `POST /api/v1/submissions` | a signed v1 document, ≤ 256 KB → `201 {"id","url","state","flags"}` (200 for identical bytes sent again); 410 retired suite, 426 unknown suite or format version, 422 invalid, 429 rate limit |
| `DELETE /api/v1/submissions/{id}` | `{"install_id","signature"}`, the signature is ed25519 over `benchfmt.WithdrawalMessage(id)` |
| `GET /api/v1/submissions/{id}` | the normalized view and the document without `install_id`/`signature`; 410 when withdrawn |
| `GET /api/v1/results` | `model, suite, preset, cpu, gpu, system, os, form_factor` (slugs for cpu/gpu/system); groups with median, p25, p75 and the number of installs |
| `GET /api/v1/facets` | the values each filter can take |
| `GET /api/v1/export` | `format=csv|json`, `what=results` (the same filters) or `what=runs` (every published run of the suite); licence in `X-Data-License`, `Link: rel=license` and the file itself |
| `/admin`, `/admin/api/*` | `Authorization: Bearer $BENCHD_ADMIN_TOKEN` |

Pages: `/benchmarks`, `/benchmarks/systems/<slug>`, `/benchmarks/cpu/<slug>`,
`/benchmarks/gpu/<slug>`, `/benchmarks/s/<id>`, `/benchmarks/how-to-run`,
`/benchmarks/privacy` (a draft pending legal review). `?lang=ru|en`, else
the `lang` cookie, else `Accept-Language`.

## Checks on a submission

1. size ≤ 256 KB, per-address rate limit (in memory, keyed by a hash of the
   address with a random salt drawn each UTC day; the address is never
   stored or logged);
2. `schema` and suite: only `standard/v1` is accepted, retired versions
   answer 410, others 426; `suite.prompts_sha256` must match the suite;
3. `benchfmt.Parse` (schema and ranges) and the signature (ownership only);
4. per-install-key rate limit; identical bytes return the earlier receipt;
5. the **plausibility ceiling** for tg without MTP: bytes read per token
   from the plan's placement (PLE and the drafter excluded; a partial
   offload reads its GPU layers at VRAM bandwidth and the rest at host
   bandwidth) against the memory-bandwidth tables in
   `internal/ingest/plausible.go`. Above the ceiling × `BENCHD_CEILING_SLACK`
   → 422. Unknown model or hardware → flagged `unknown`, accepted;
6. the **output signal** (soft): the longest output prefix shared with the
   cohort (model file, backend, ggml CPU variant, repack, llama.cpp tag),
   from `output.sha256` and the optional `output.prefixes`. Shown, never a
   reason to refuse;
7. **MAD outliers** within the hardware cohort (≥ 5 units, modified z > 3.5)
   and hand-entered system models go to the moderation queue; with
   `BENCHD_MODERATION=manual` everything does.

Aggregates count published, plausible `ok` runs, one value per *unit*: an
install's repeats collapse to their median, and installs sharing a coarse
hardware fingerprint (CPU, threads, memory, GPUs, system, form factor, OS
family; no versions) collapse too, so a fresh key per run does not multiply
one machine.

## Running

```sh
cd services/bench
go run ./cmd/benchd serve -addr 127.0.0.1:8080 -db ./bench.db -public-url http://127.0.0.1:8080
```

Send a result with the real client (it accepts only `https` submit URLs, so
locally use `e2e/client-e2e.sh`, which serves benchd with a throwaway
certificate), or with the Go e2e test: `go test ./e2e`.

### Configuration

Every flag has an environment variable; `benchd serve -h` lists them.

| Variable | Default | |
| --- | --- | --- |
| `BENCHD_ADDR` | `:8080` | listen address |
| `BENCHD_DB` | `bench.db` | SQLite path (WAL) |
| `BENCHD_PUBLIC_URL` | `https://nano.llmto.day` | base of receipt URLs |
| `BENCHD_ADMIN_TOKEN` | — | enables `/admin`; environment only |
| `BENCHD_DATA_LICENSE` | `CC-BY-4.0` | open data licence (SPDX); **the owner's decision**, `CC0-1.0` is the alternative |
| `BENCHD_DATA_LICENSE_URL` | from the SPDX id | |
| `BENCHD_TRUST_PROXY` | `false` (`true` in the image) | client address from `X-Forwarded-For` |
| `BENCHD_RATE_IP` | `20` | submissions per address per hour |
| `BENCHD_RATE_KEY` | `20` | submissions per install key per day |
| `BENCHD_RETIRED_SUITES` | — | e.g. `standard/v0`, answered with 410 |
| `BENCHD_MODERATION` | `auto` | or `manual` |
| `BENCHD_CEILING_SLACK` | `1.3` | multiplier on the plausibility ceiling |
| `BENCHD_BACKUP_DIR` | — (`/data/backups` in the image) | |
| `BENCHD_BACKUP_KEEP` | `720h` | backup retention, must be bounded |
| `BENCHD_BACKUP_EVERY` | `0` | `serve` takes a backup this often |
| `BENCHD_TLS_CERT`, `BENCHD_TLS_KEY` | — | serve HTTPS directly (normally Caddy does) |

A new suite version is a code change: add it to `ingest.DefaultSuites`
with its prompts hash (NanoLLM's `internal/bench.PromptsSHA256()`), and
list the old one in `BENCHD_RETIRED_SUITES` when it stops being accepted.

## Deploy

```sh
cd services/bench/deploy
BENCHD_ADMIN_TOKEN=$(openssl rand -hex 32) docker compose up -d --build
```

The image is built from the repository root (`docker build -f
services/bench/Dockerfile .`; `Dockerfile.dockerignore` limits the context
to `benchfmt/` and `services/bench/`). It runs as non-root on distroless
with the database and backups in the `/data` volume; copy `/data/backups`
and `/data/bench.db.withdrawals` off the host. To restore, stop benchd, put
a snapshot at `/data/bench.db` (remove `bench.db-wal` and `-shm`) and keep
the current `bench.db.withdrawals`: the tombstones are re-applied on start.

Caddy terminates TLS, sends `/benchmarks`, `/api/v1` and `/admin` to
benchd and the rest to the site (replace the placeholder `handle` in the
Caddyfile), and writes access logs with the client address, port and
headers removed. benchd writes no access log.

## Tests

```sh
go vet ./... && go test ./...          # unit tests and the e2e (builds benchd)
go test -short ./...                   # without the e2e
e2e/client-e2e.sh path/to/nanollm path/to/benchd   # the real client
```

CI: `.github/workflows/bench-service.yml` (benchfmt, lint and tests, the
e2e with both the Go driver and the real `nanollm bench submit`/`withdraw`,
the image with a smoke run, and the Caddyfile).
