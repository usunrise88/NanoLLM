# benchd image. Build from the repository root, while the service lives in
# the NanoLLM repository (it needs ../../benchfmt):
#
#   docker build -f services/bench/Dockerfile -t nanollm-benchd .
#
# Dockerfile.dockerignore next to this file keeps the context to benchfmt/
# and services/bench/. After the service moves out (git subtree split), it
# depends on a tagged benchfmt and the COPY of benchfmt/ goes away.

FROM golang:1.25-alpine AS build
WORKDIR /src
COPY benchfmt/ benchfmt/
COPY services/bench/go.mod services/bench/go.sum services/bench/
WORKDIR /src/services/bench
RUN go mod download
COPY services/bench/ ./
ARG VERSION=dev
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w -X main.version=${VERSION}" -o /out/benchd ./cmd/benchd \
 && mkdir -p /out/data/backups

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/benchd /benchd
COPY --from=build --chown=65532:65532 /out/data /data
ENV BENCHD_ADDR=:8080 \
    BENCHD_DB=/data/bench.db \
    BENCHD_BACKUP_DIR=/data/backups \
    BENCHD_TRUST_PROXY=true
VOLUME /data
EXPOSE 8080
USER nonroot:nonroot
HEALTHCHECK --interval=30s --timeout=5s CMD ["/benchd", "healthcheck"]
ENTRYPOINT ["/benchd"]
CMD ["serve"]
