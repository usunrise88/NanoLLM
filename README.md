# NanoLLM

**Decent LLM for any device.**

Offline LLM inference server: one Go binary plus a `lib/` directory, local
inference of small models on CPU, GPU and NPU, an OpenAI-compatible HTTP API
and an embedded web interface. No prompt leaves the machine; network access is
needed only to fetch weights.

Model: **Gemma 4 E4B** (Unsloth QAT `UD-Q4_K_XL`, Q4_0 throughout, with its int4
MTP drafter).
Gemma 4 12B and Qwen3.5-9B from the specification are deferred. The design
document is the NanoLLM Development Specification; section numbers in code
comments (`SPEC §4`) refer to it. Structure and conventions follow
[NanoASR](https://github.com/usunrise88/NanoASR).

## Status

`v0.1.0-rc.1` is out as a pre-release (libraries `b11600-n8`). Stage 1 (code that does not need fast
hardware) and stage 2 (a 16-core EPYC server, CPU only) are done; stage 3 (RTX 3070 Ti laptop) is measured
but its gate is not closed; Apple M1 passes golden, KLD and conformance on Metal. What is left, what each
machine showed and the audit of the roadmap are in [docs/ROADMAP.md](docs/ROADMAP.md), [docs/STAGE2.md](docs/STAGE2.md)
and [docs/AUTOTUNE.md](docs/AUTOTUNE.md) (every optimization variant with its results, kept behind a switch).

| Area | State |
| --- | --- |
| Engine | purego binding to `libllama` (no cgo) from our build of the pinned llama.cpp `b11600` with a series of eight patches (`patches/llama.cpp`): Q4_0 repack kernels for AVX2 and AVX-VNNI, a CUDA backend tuned to the model, Vulkan on Intel iGPUs. Gemma 4 E4B QAT (Q4_0) with its MTP drafter, greedy output identical to `llama-server`. Verified on CPU (EPYC and a laptop), Metal (M1), CUDA and Vulkan (RTX, Iris Xe: measured, golden not yet recorded) |
| Placement | `nanollm plan` places the model from its measured tensor inventory for every preset (`vram-*`, `uma-*`, `ram-*`): VRAM, locked RAM, file-backed; range-lock keeps the locked part resident |
| Server | OpenAI-compatible `/v1/chat/completions` (streaming, tools, `response_format`, logprobs), the native API and an Ollama-compatible one (`api.dialects: [openai, native, ollama]`, opt-in); continuous batching, prompt cache, model pool, API keys and rate limits; SQLite job queue with SSE progress and signed webhooks; a 24 h soak on the real model passed |
| CLI | `init` (config, keys, model download with resume), `serve`, `run` (REPL or `--prompt`), `models` (catalog, pull, inspect), `plan`, `devices`, `key`, `service`, `bench`, `eval`, `optimize` |
| Autotune | `nanollm optimize` surveys the machine, measures for one of three modes (`silent`, `optimal`, `powerful`) in about a minute, reports problems (slow disk, old driver, battery, a missing CUDA add-on) and writes a profile the server runs by; no administrator rights; [docs/OPTIMIZE.md](docs/OPTIMIZE.md) |
| Web UI | Chat, Jobs, Models, Devices, Settings, Bench; English and Russian |
| Benchmark | `nanollm bench` runs presets one process each and submits results only on explicit consent; the service in `services/bench` runs at [nano.llmto.day](https://nano.llmto.day/benchmarks) |
| Evaluation | `nanollm eval` suites (Russian and English instructions, JSON and tools, long context, reasoning, 50 prompts for blind rating), KLD tooling against the reference configuration |
| Delivery | release archives for Linux, Windows and macOS, `install.sh` / `install.ps1` as a service, Docker images (CPU/Vulkan, CUDA); macOS signing waits for Apple credentials |

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

Requirements: Go 1.27+, and Node 22+ for the web UI.

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

```sh
nanollm optimize                     # pick a mode (silent, optimal, powerful); measures and asks before it applies
nanollm optimize --dry-run           # what the machine is and what each setup would cost; measures nothing
nanollm optimize --mode optimal --yes
```

With a service running, the measurement happens inside it (the model is
unavailable for the minute it takes); the installers offer it after an install
and a recheck after an upgrade that made the profile stale (`--mode` /
`-Mode`, `--no-optimize` / `-NoOptimize`). A config older than this feature
rejects the `tune:` key, so upgrade the binary before adding it.

## Embed in an application

`nanollm managed` runs NanoLLM as the background process of another
application, with no administrator rights:

- a child process on a loopback port the OS picks, with a token made up for
  each start; both arrive on stdout as one JSON line;
- it stops when its stdin closes or the `-parent-pid` process exits;
- it reads no configuration file: the application's choices (model, mode)
  live in `<data-dir>/settings.json` and change through
  `PATCH /api/v1/managed/settings`;
- it tunes itself: when the mode (`silent` until the application picks
  another) has no current profile, it measures as soon as the model is on
  disk and applies the result.

```sh
nanollm managed -data-dir "$HOME/.local/share/softphone/nanollm" -parent-pid $$
```

The integrator's guide, in Russian, is [nanollm-embedded.md](nanollm-embedded.md).

## Layout

```
cmd/nanollm/            entry point, CLI verbs
internal/core/          core.Service, request/response model, pipeline stages, errors
internal/engine/        core.Service on top of the pool and the scheduler
internal/backend/       Backend/Model interfaces; llamacpp/ (purego), fake/, ortgenai/ (v2), mlx/ (v3)
internal/gguf/          GGUF header and tensor inventory
internal/plan/          memory modes M1–M10, presets, the planner
internal/memlock/       range-lock of the mapped model
internal/pool/          model pool, slots, prompt cache
internal/sched/         device governor, continuous batching
internal/chat/          Gemma 4 template, tool-call and reasoning parsers
internal/grammar/       JSON Schema to GBNF
internal/api/           adapter registry; native/, openai/, ollama/ dialects
internal/server/        runtime wiring for serve and run
internal/managed/       nanollm managed: settings, events, the autotune loop
internal/httpx/         auth, problem+json, SSE, middleware
internal/registry/      catalog, manifests, downloads, tensor inventories
internal/queue/         SQLite jobs, SSE, webhooks
internal/bench/         benchmark runner and submission client
internal/hwinfo/        hardware profile for benchmark results
internal/eval/          eval suites, checkers, blind rating
internal/config/        YAML load/validate, env overrides
internal/ui/            embedded SPA (built from web/)
benchfmt/               benchmark result format (its own module)
services/bench/         benchmark service (its own module)
web/                    SPA sources: React, Vite, TanStack, Tailwind
tests/conformance/      OpenAI client conformance suite
tools/                  bindgen, fixtures, golden, optimize (KLD), chat fixtures
scripts/                fetch-libs, dist, install.sh, install.ps1
deploy/                 systemd unit, launchd plist, Dockerfile, compose
```

## License

MIT, see [LICENSE](LICENSE). Third-party components: [NOTICE](NOTICE).
