# NanoLLM build (SPEC §10).
#
# One Go binary with the SPA embedded, plus a lib/ directory of native
# libraries from one pinned llama.cpp release. No C is compiled here: every
# target is built with CGO_ENABLED=0 and cross-compiles from one runner.

SHELL       := /bin/bash
GO          ?= go
NPM         ?= npm
VERSION     ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LLAMA_TAG   := $(shell tr -d '[:space:]' < scripts/llama.cpp.tag)
LLAMA_COMMIT := $(shell tr -d '[:space:]' < scripts/llama.cpp.commit)
LIBS_RELEASE := $(shell tr -d '[:space:]' < scripts/libs.release)
LLAMACPP    := github.com/usunrise88/nanollm/internal/backend/llamacpp
LDFLAGS     := -X main.version=$(VERSION) \
               -X $(LLAMACPP).PinnedTag=$(LLAMA_TAG) \
               -X $(LLAMACPP).PinnedCommit=$(LLAMA_COMMIT) \
               -X $(LLAMACPP).LibsRelease=$(LIBS_RELEASE)
DIST        := dist
UI_DIST     := internal/ui/dist
export CGO_ENABLED := 0

# The five release platforms; darwin is built per architecture and merged.
TARGETS     := linux/amd64 linux/arm64 windows/amd64 windows/arm64 darwin/arm64 darwin/amd64

.PHONY: all web web-check e2e build build-race build-noui cross dist dist-all docker docker-cuda libs tools bindgen layout test-native lint test test-race conformance run run-fake tidy clean help

all: web build

## web: build the SPA and stage it for embedding (Node 22+)
web:
	cd web && $(NPM) ci --no-audit --no-fund
	cd web && $(NPM) run build
	rm -rf $(UI_DIST)
	cp -r web/dist $(UI_DIST)

## web-check: lint, typecheck and unit-test the SPA, as CI's web job does
web-check:
	cd web && $(NPM) ci --no-audit --no-fund
	cd web && $(NPM) run lint && $(NPM) run typecheck && $(NPM) test

## e2e: the SPA in Chromium against `serve` on the fake backend (PLAYWRIGHT_CHROMIUM picks a browser)
e2e: web build
	NANOLLM_REQUIRE_UI=1 $(GO) test -count=1 -run 'TestHasSPA|TestEmbeddedSPA' ./internal/ui
	cd web && NANOLLM_BIN=$(abspath $(DIST)/nanollm) $(NPM) run test:e2e

## build: the server binary for this machine (run `make web` first, or it starts without a UI)
build:
	$(GO) build -ldflags "$(LDFLAGS)" -o $(DIST)/nanollm ./cmd/nanollm

## build-race: the server under the race detector, for a soak run on the real library (needs cgo)
build-race:
	CGO_ENABLED=1 $(GO) build -race -ldflags "$(LDFLAGS)" -o $(DIST)/nanollm-race ./cmd/nanollm

## build-noui: the same without the SPA
build-noui:
	$(GO) build -tags noui -ldflags "$(LDFLAGS)" -o $(DIST)/nanollm-noui ./cmd/nanollm

## cross: every release target, with and without the UI, into dist/<os>_<arch>/
cross:
	@set -e; for t in $(TARGETS); do \
		os=$${t%/*}; arch=$${t#*/}; ext=; [ $$os = windows ] && ext=.exe; \
		out=$(DIST)/$${os}_$${arch}; \
		echo "build $$os/$$arch"; \
		GOOS=$$os GOARCH=$$arch $(GO) build -ldflags "$(LDFLAGS)" -o $$out/nanollm$$ext ./cmd/nanollm; \
		GOOS=$$os GOARCH=$$arch $(GO) build -tags noui -ldflags "$(LDFLAGS)" -o $$out/nanollm-noui$$ext ./cmd/nanollm; \
	done

## dist: the release archives for this machine, with and without the UI, into dist/
dist: web
	VERSION=$(VERSION) ./scripts/dist.sh

## dist-all: every release archive and both CUDA add-ons (darwin needs lipo or llvm-lipo)
dist-all: web
	VERSION=$(VERSION) ./scripts/dist.sh all

## docker: the CPU + Vulkan image, nanollm:$(VERSION) (deploy/Dockerfile)
docker:
	docker build -f deploy/Dockerfile --build-arg VERSION=$(VERSION) -t nanollm:$(VERSION) .

## docker-cuda: the CUDA image, nanollm:$(VERSION)-cuda (linux/amd64)
docker-cuda:
	docker build -f deploy/Dockerfile --build-arg FLAVOUR=cuda --build-arg VERSION=$(VERSION) -t nanollm:$(VERSION)-cuda .

## libs: fetch the pinned llama.cpp libraries for this machine into dist/<os>_<arch>/lib
libs:
	./scripts/fetch-libs.sh

## tools: the pinned llama.cpp tools (llama-server, llama-cli, …) into dist/<os>_<arch>/tools, for golden data
tools:
	./scripts/fetch-libs.sh --tools

## bindgen: regenerate the libllama binding; LLAMA_SRC is a llama.cpp checkout of the pinned tag
bindgen:
	@test -n "$(LLAMA_SRC)" || { echo "set LLAMA_SRC to a llama.cpp checkout of $(LLAMA_TAG)"; exit 1; }
	$(GO) run ./tools/bindgen -src $(LLAMA_SRC)
	$(GO) run ./tools/bindgen -src $(LLAMA_SRC) $(foreach d,$(wildcard $(DIST)/*/lib),-check-exports $(d))

## layout: check the binding's struct layout with zig cc for every target (LLAMA_SRC as above)
layout:
	./scripts/check-layout.sh $(LLAMA_SRC)

## test-native: tests against the real libllama (run `make libs`; set NANOLLM_TEST_MODEL or NANOLLM_TEST_MODELS)
test-native:
	$(GO) test -tags native -count=1 -timeout 60m ./internal/backend/llamacpp/...

## lint: vet, gofmt and golangci-lint when it is installed
lint:
	$(GO) vet ./...
	@gofmt -l cmd internal tools | tee /dev/stderr | (! read)
	@if command -v golangci-lint >/dev/null; then golangci-lint run; fi

## test: unit tests
test:
	$(GO) test ./...

## test-race: the pool, the scheduler and the queue are only correct under -race (needs cgo)
test-race:
	CGO_ENABLED=1 $(GO) test -race ./...

## conformance: the OpenAI client conformance suite against the fake backend (pip install -r tests/conformance/requirements.txt)
conformance: build
	NANOLLM_BIN=$(abspath $(DIST)/nanollm) python3 -m pytest -q tests/conformance

## run-fake: the server on the scripted fake backend, for UI work without a model
run-fake: build
	NANOLLM_BACKENDS_ENABLED=fake $(DIST)/nanollm serve

## run: run the server against the example configuration's defaults (open, loopback)
run: build
	$(DIST)/nanollm serve

tidy:
	$(GO) mod tidy

# `make web` replaces the committed placeholder index.html; clean restores it.
clean:
	rm -rf $(DIST) web/dist $(UI_DIST)/assets
	git checkout -- $(UI_DIST)/index.html 2>/dev/null || true

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'
