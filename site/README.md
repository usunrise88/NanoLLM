# site: nano.llmto.day

The landing page of the Nano* family and the product pages: **Nano\*** (home),
**NanoLLM** (with the way to the benchmark at `/benchmarks`), and placeholders for
**NanoASR** and **NanoTTS**. The idea: decent AI for everyone, no GPU required.

Design: shadcn conventions (Button, Card, Badge in `src/components/ui`) on Base UI
primitives, Radix Colors (the `gray` scale only: the site is black and white),
Iconoir icons, Tailwind 4. The theme follows the system (`prefers-color-scheme`),
with a system / light / dark switch whose choice is remembered; an inline script in
`index.html` sets it before the first paint. English only for now. No third-party
requests: system fonts, no analytics.

The pages are React, prerendered at build time to plain HTML (`scripts/prerender.mjs`),
so they read without scripts; the client bundle only hydrates the theme switch.

    npm ci
    npm run dev          # vite, with hot reload
    npm run typecheck
    npm run build        # dist/: index.html, nanollm/, nanoasr/, nanotts/, 404.html, assets/

A page is a component in `src/pages/` plus a line in `src/pages.tsx`. The product list
(names, one-line descriptions, repositories, status) is `src/lib/products.tsx`: the
home page cards, the header and the footer all read it.

On hyper it is served by a small Caddy container from `dist/`, behind the host's Caddy,
next to benchd: `scripts/staging-bench-service.sh site` builds and (re)starts it, no
sudo (`services/bench/README.md`, "Staging на hyper").
