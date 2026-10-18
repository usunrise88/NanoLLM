# web/

Sources of the single-page app served at `/ui/` (SPEC §11): Chat, Jobs,
Models, Devices and Settings, built only on the public API.

The SPA lands with v1.0. `make web` builds it (Node 22+) and stages the output
in `internal/ui/dist/`, which `internal/ui` embeds at compile time; until then
the binary embeds a placeholder and reports `ui false` in `nanollm version`.
