# syntax=docker/dockerfile:1
# check=skip=SecretsUsedInArgOrEnv

# NanoLLM in a container (SPEC §10): one image for CPU + Vulkan and one for
# CUDA. The weights live in the /data volume, never in the image, and the
# configuration comes from NANOLLM_* variables (SPEC §8), so no file needs
# mounting.
#
#   docker build -f deploy/Dockerfile -t nanollm .                        # CPU + Vulkan
#   docker build -f deploy/Dockerfile --build-arg FLAVOUR=cuda -t nanollm:cuda .
#   docker buildx build -f deploy/Dockerfile --platform linux/amd64,linux/arm64 .
#
#   docker run -d -p 8080:8080 -v nanollm-data:/data --ulimit memlock=-1 \
#     -e NANOLLM_AUTH_KEYS=sk-... nanollm
#   docker run --rm -v nanollm-data:/data -e NANOLLM_AUTH_KEYS=sk-... nanollm models pull
#
# Every command validates the same configuration, so the others need the
# keys too: the image listens on 0.0.0.0, where the server never runs open.
#
# deploy/docker-compose.yml does the same with both flavours.
#
# FLAVOUR is declared before the first FROM because that is the only scope a
# FROM line can take a variable from.
ARG FLAVOUR=cpu

# ---- SPA ---------------------------------------------------------------------
FROM node:22-bookworm-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- native libraries ----------------------------------------------------------
# The pinned llama.cpp release, verified against scripts/libs.sha256 by
# fetch-libs exactly as for the release archives. The CUDA flavour adds the
# add-on: ggml-cuda and the CUDA runtime. The driver (libcuda.so) is not in
# the image; the NVIDIA container toolkit mounts the host's with --gpus.
FROM debian:bookworm-slim AS libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY scripts/fetch-libs.sh scripts/libs.sha256 scripts/libs.release scripts/llama.cpp.tag scripts/llama.cpp.commit scripts/llama.cpp.base scripts/
ARG TARGETARCH
ARG FLAVOUR
RUN set -eu; \
    target="linux_${TARGETARCH:-amd64}"; \
    ./scripts/fetch-libs.sh "" "$target"; \
    if [ "$FLAVOUR" = cuda ]; then \
      [ "$target" = linux_amd64 ] || { echo "the CUDA flavour is linux/amd64 only" >&2; exit 1; }; \
      ./scripts/fetch-libs.sh "" linux_amd64_cuda; \
      cp -P dist/linux_amd64_cuda/lib/* "dist/$target/lib/"; \
    fi; \
    mv "dist/$target/lib" /out-lib

# ---- server ------------------------------------------------------------------
# No cgo: the binary loads the libraries with purego, as in the release.
FROM golang:1.25-bookworm AS build
WORKDIR /src
ENV GOWORK=off CGO_ENABLED=0
COPY go.mod go.sum ./
COPY benchfmt/ ./benchfmt/
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY . .
COPY --from=web /src/web/dist ./internal/ui/dist
ARG VERSION=dev
ARG TARGETOS TARGETARCH
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    llamacpp=github.com/usunrise88/nanollm/internal/backend/llamacpp; \
    GOOS="${TARGETOS:-linux}" GOARCH="${TARGETARCH:-amd64}" go build -trimpath \
      -ldflags "-X main.version=${VERSION} \
                -X ${llamacpp}.PinnedTag=$(tr -d '[:space:]' < scripts/llama.cpp.tag) \
                -X ${llamacpp}.PinnedCommit=$(tr -d '[:space:]' < scripts/llama.cpp.commit)" \
      -o /out/nanollm ./cmd/nanollm

# ---- runtime -----------------------------------------------------------------
# glibc 2.36 and libstdc++ GLIBCXX_3.4.30 cover what the llama.cpp Ubuntu
# builds need. libgomp1 is the OpenMP runtime every CPU backend links;
# libvulkan1 is the Vulkan loader. A GPU's Vulkan driver is the host's
# business: NVIDIA's comes in through the container toolkit (the graphics
# capability below); for AMD and Intel pass --device /dev/dri and build with
# --build-arg EXTRA_PACKAGES=mesa-vulkan-drivers. curl is for the healthcheck.
FROM debian:bookworm-slim
ARG EXTRA_PACKAGES=""
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libvulkan1 ${EXTRA_PACKAGES} && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --home-dir /data --shell /usr/sbin/nologin nanollm && \
    install -d -o nanollm -g nanollm -m 0750 /data

COPY --from=build /out/nanollm /opt/nanollm/nanollm
COPY --from=libs /out-lib/ /opt/nanollm/lib/
COPY configs/nanollm.example.yaml /opt/nanollm/nanollm.example.yaml
COPY LICENSE NOTICE /opt/nanollm/

ARG VERSION=dev
ARG FLAVOUR
LABEL org.opencontainers.image.title="NanoLLM" \
      org.opencontainers.image.description="Decent LLM for any device: offline inference server (${FLAVOUR})" \
      org.opencontainers.image.source="https://github.com/usunrise88/NanoLLM" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT"

# The configuration is the defaults plus these, and whatever else is passed
# with -e (every key has a NANOLLM_* name, SPEC §8). A non-loopback address
# needs keys: the server refuses to start in apikey mode without
# NANOLLM_AUTH_KEYS, and refuses open mode on 0.0.0.0 outright.
ENV PATH=/opt/nanollm:$PATH \
    NANOLLM_SERVER_ADDR=0.0.0.0:8080 \
    NANOLLM_SERVER_DATA_DIR=/data \
    NANOLLM_AUTH_MODE=apikey \
    NANOLLM_LOG_FORMAT=json \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

USER nanollm
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8080

# /healthz is public and answers as soon as the server listens; /readyz is
# the one that waits for the default model.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD ["sh", "-c", "curl -fsS \"http://127.0.0.1:${NANOLLM_SERVER_ADDR##*:}/healthz\" >/dev/null"]

ENTRYPOINT ["/opt/nanollm/nanollm"]
CMD ["serve"]
