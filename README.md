# NanoLLM

**Decent LLM for any device.**

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
| Delivery | `scripts/dist.sh` release archives, `install.sh` / `install.ps1`, `nanollm service` (systemd, launchd, Windows service), Docker images, `release.yml` |

Next is milestone v0.1 — the engine spike: purego binding to `libllama`,
`nanollm run` / `plan` / `bench` with Gemma 4 E4B + MTP (SPEC §14). The
development plan by stage and machine is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Install

From a release, as a service that starts at boot (Linux, systemd) or at
login (macOS, launchd agent; `--daemon` for the whole machine):

```sh
curl -fsSL https://github.com/usunrise88/NanoLLM/releases/latest/download/install.sh | bash
curl -fsSL .../install.sh | bash -s -- --cuda          # Linux x86-64 with an NVIDIA GPU
curl -fsSL .../install.sh | bash -s -- --uninstall     # --purge also deletes the data
```

Windows, in PowerShell (it asks for administrator rights):

```powershell
irm https://github.com/usunrise88/NanoLLM/releases/latest/download/install.ps1 | iex
```

The installer verifies the archive against `sha256sums.txt`, unpacks it into
`/opt/nanollm` (`/usr/local/nanollm`, `C:\Program Files\NanoLLM`), writes
`nanollm.yaml` with an admin and a user key into the data directory
(`/var/lib/nanollm`, `~/Library/Application Support/NanoLLM`,
`C:\ProgramData\NanoLLM`), fetches the default model and starts the service.
Running it again upgrades in place and keeps the configuration, the keys and
the weights. `--help` (`-Help`) lists the flags; each has a `NANOLLM_*`
variable for the piped form, and `--archive PATH` installs a local archive.

Linux needs `libgomp.so.1` (every CPU backend links it) and, for a GPU,
`libvulkan.so.1`; Windows needs the Visual C++ runtime. None of them ships in the
archive, and the installers name what is missing and how to add it.

The service is managed by the binary on every platform:

```sh
nanollm service install | uninstall | start | stop | restart | status
nanollm --install        # the same verbs, spelled as Windows programs do
nanollm service unit     # print the systemd unit or launchd plist install would write
```

Docker, CPU + Vulkan or CUDA, with the weights in a volume and the
configuration in `NANOLLM_*` variables:

```sh
make docker              # or: docker build -f deploy/Dockerfile [--build-arg FLAVOUR=cuda] .
docker run -d -p 8080:8080 -v nanollm-data:/data --ulimit memlock=-1 \
  -e NANOLLM_AUTH_KEYS=sk-nanollm-... nanollm:<version>
```

`deploy/docker-compose.yml` runs either flavour.

## Build

Requirements: Go 1.25+, and Node 22+ for the web UI.

```sh
make web build    # the SPA into internal/ui/dist, then dist/nanollm with it embedded
make build        # without `make web` first, a binary that serves no UI
make cross        # every release target, with and without the UI
make dist         # release archives for this machine into dist/ (scripts/dist.sh)
make dist-all     # every release archive and the CUDA add-ons (darwin needs llvm-lipo)
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
scripts/                fetch-libs, dist, install.sh, install.ps1
deploy/                 systemd unit, launchd plist, Dockerfile, compose
```

## License

MIT, see [LICENSE](LICENSE). Third-party components: [NOTICE](NOTICE).
