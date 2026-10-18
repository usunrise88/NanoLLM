# NanoLLM

Offline LLM inference server: one Go binary plus a `lib/` directory, local
inference of small models on CPU, GPU and NPU, an OpenAI-compatible HTTP API
and an embedded web interface. No prompt leaves the machine; network access is
needed only to fetch weights.

Model: **Gemma 4 E4B** (Unsloth QAT `UD-Q4_K_XL` with its int4 MTP drafter).
Gemma 4 12B and Qwen3.5-9B from the specification are deferred. The design
document is the NanoLLM Development Specification; section numbers in code
comments (`SPEC §4`) refer to it. Structure and conventions follow
[NanoASR](https://github.com/usunrise88/NanoASR).

## Status

Bootstrap. What works today:

| Area | State |
| --- | --- |
| `nanollm init` | Writes a validated config with fresh admin and user keys (no download yet) |
| `nanollm serve` | `/healthz`, `/readyz` (503 until a model loads), API-key auth, `openai` and `native` routes mounted; generation routes answer 501 |
| `nanollm models catalog` | Built-in catalog, pinned to a repository commit, size and SHA-256 |
| `nanollm devices` | Reports whether `libllama` is where the binary looks for it |
| Config | Full schema from SPEC §8, strict keys, `NANOLLM_*` env overrides, validation |
| Memory modes and presets | M1–M10 and all 19 presets defined and validated (`internal/plan`) |
| Build | `CGO_ENABLED=0` cross-compilation for all release targets; llama.cpp pinned to `b11600`, `scripts/fetch-libs.sh` verified in CI for every target and the CUDA add-ons |

Next is milestone v0.1 — the engine spike: purego binding to `libllama`,
`nanollm run` / `plan` / `bench` with Gemma 4 E4B + MTP (SPEC §14). The
development plan by stage and machine is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Build

Requirements: Go 1.25+, and Node 22+ for the web UI.

```sh
make web build    # the SPA into internal/ui/dist, then dist/nanollm with it embedded
make build        # without `make web` first, a binary that serves no UI
make cross        # every release target, with and without the UI
make test lint
make web-check    # the SPA's lint, typecheck and unit tests
make e2e          # the SPA in Chromium against `serve` on the fake backend
make run-fake     # a server on the scripted fake backend; the UI is at /ui/
```

Native libraries come from one pinned llama.cpp release, the tag in
`scripts/llama.cpp.tag` (at least `b9568`, which has Gemma 4 MTP and the E4B
drafter fix). Archive hashes are in `scripts/libs.sha256`; the `libs` workflow
lists a release's digests and proves every target.

```sh
make libs                                  # dist/<os>_<arch>/lib/ for this machine
scripts/fetch-libs.sh "" windows_amd64_cuda  # a CUDA add-on
```

## Run

```sh
nanollm init                         # writes nanollm.yaml, prints the keys
nanollm serve -config nanollm.yaml
curl localhost:8080/healthz
```

Without a config file `nanollm serve` listens on `127.0.0.1:8080` with
authentication off; open mode is refused on any non-loopback address.
`configs/nanollm.example.yaml` documents every setting.

## Layout

```
cmd/nanollm/            entry point, CLI verbs
internal/core/          core.Service, request/response model, pipeline stages, errors
internal/backend/       Backend/Model interfaces; llamacpp/, ortgenai/ (v2), mlx/ (v3)
internal/plan/          memory modes M1–M10 and presets
internal/pool/          model pool, slots, prompt cache (v1.0)
internal/sched/         device governor, continuous batching (v1.0)
internal/chat/          chat templates, tool-call and reasoning parsers (v1.0)
internal/api/           adapter registry; native/, openai/ dialects
internal/httpx/         auth, problem+json, middleware
internal/registry/      catalog, manifests
internal/queue/         SQLite jobs, SSE, webhooks (v1.0)
internal/config/        YAML load/validate, env overrides
internal/ui/            embedded SPA (built from web/)
web/                    SPA sources: React, Vite, TanStack, Tailwind
scripts/                fetch-libs, release tooling
```

## License

MIT, see [LICENSE](LICENSE). Third-party components: [NOTICE](NOTICE).
