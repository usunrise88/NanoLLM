import { ArrowRight, ArrowUpRight, Check, Cpu, Github, Server, Sparks, StatsReport, Timer } from 'iconoir-react'

import { Container, Page } from '@/components/Layout'
import { Badge } from '@/components/ui/badge'
import { LinkButton } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { BENCHMARK, product } from '@/lib/products'

const p = product('nanollm')

const features = [
  {
    Icon: Cpu,
    title: 'Built for the CPU',
    text: 'Hand-tuned 4-bit kernels for AVX2 processors make prompt processing about 1.8 to 1.9 times faster than stock llama.cpp, with bit-identical output. Model loading is parallel and NUMA-aware.',
  },
  {
    Icon: Server,
    title: 'OpenAI-compatible API',
    text: 'Chat completions with streaming, tool calls and JSON output, plus a native API with sessions and a prompt cache. One Go binary, no Python.',
  },
  {
    Icon: Sparks,
    title: 'Speculative decoding',
    text: 'Multi-token prediction and prompt lookup draft several tokens at once. Chat gets up to 1.4 times faster on top of the kernels.',
  },
  {
    Icon: Timer,
    title: 'Fits in 8 GB',
    text: 'The default preset peaks near 6.5 GiB. Smaller memory budgets trade speed for space, down to modes that stream the weights from disk.',
  },
  {
    Icon: Check,
    title: 'Optimizations never change the answer',
    text: 'Every build is checked against golden outputs and a KL-divergence gate, so a faster kernel cannot quietly change what the model says.',
  },
  {
    Icon: StatsReport,
    title: 'Measured in the open',
    text: 'The benchmark runs the same suite on any machine and shows medians over many installations.',
  },
] as const

const rows = [
  { cores: '16 cores', generation: '25', prompt: '140' },
  { cores: '4 cores', generation: '12', prompt: '44' },
] as const

export function NanoLLM() {
  return (
    <Page>
      <section className="border-b">
        <Container className="py-16 sm:py-24">
          <div className="flex flex-wrap items-center gap-3">
            <Badge variant="outline">Language</Badge>
            <Badge>{p.status}</Badge>
          </div>
          <h1 className="mt-6 text-4xl font-semibold tracking-tight sm:text-6xl">NanoLLM</h1>
          <p className="mt-6 max-w-2xl text-lg text-muted-foreground text-pretty">
            A cross-platform LLM runtime for edge computing. It serves Google&apos;s Gemma 4 E4B, quantized to 4 bits,
            from a single binary with an OpenAI-compatible API, on the CPU you already have.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <LinkButton href={BENCHMARK} size="lg">
              Open the benchmark
              <ArrowRight />
            </LinkButton>
            <LinkButton href={p.repo} variant="outline" size="lg">
              <Github />
              Source on GitHub
            </LinkButton>
          </div>
        </Container>
      </section>

      <section className="py-16">
        <Container>
          <h2 className="text-2xl font-semibold tracking-tight">Speed on a plain server CPU</h2>
          <p className="mt-2 max-w-2xl text-muted-foreground">
            No GPU involved. Tokens per second on an AMD EPYC 7351P, a 2017 server processor with AVX2.
          </p>
          <div className="mt-6 overflow-hidden rounded-xl border">
            <table className="w-full text-left text-sm">
              <thead className="bg-muted text-muted-foreground">
                <tr>
                  <th className="px-4 py-3 font-medium">Setup</th>
                  <th className="px-4 py-3 font-medium">Generation</th>
                  <th className="px-4 py-3 font-medium">Prompt processing</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.cores} className="border-t">
                    <th scope="row" className="px-4 py-3 font-medium">
                      {r.cores}
                    </th>
                    <td className="px-4 py-3 tabular-nums">about {r.generation} tok/s</td>
                    <td className="px-4 py-3 tabular-nums">about {r.prompt} tok/s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-sm text-muted-foreground">
            Gemma 4 E4B, 4-bit weights, measured with llama-bench on our test server. With speculative decoding, chat on
            16 cores reaches about 33 tok/s. Your numbers will differ: run the benchmark.
          </p>
        </Container>
      </section>

      <section className="border-t py-16">
        <Container>
          <h2 className="text-2xl font-semibold tracking-tight">What it does</h2>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {features.map(({ Icon, title, text }) => (
              <Card key={title} className="gap-4">
                <CardHeader>
                  <Icon className="size-6" />
                  <CardTitle className="mt-2 text-base">{title}</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">{text}</CardContent>
              </Card>
            ))}
          </div>
        </Container>
      </section>

      <section className="border-t py-16">
        <Container className="grid gap-10 md:grid-cols-2">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Status</h2>
            <p className="mt-3 text-muted-foreground">
              NanoLLM is in development. Linux on x86-64 is verified end to end on the CPU. Libraries for Linux, macOS
              and Windows on x86-64 and arm64 are built and tested in CI. GPU backends are next.
            </p>
          </div>
          <Card className="gap-4">
            <CardHeader>
              <CardTitle>Run the benchmark</CardTitle>
              <CardDescription>
                See how NanoLLM&apos;s presets run on real hardware, and add your own machine to the picture.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-3">
              <LinkButton href={BENCHMARK}>
                Open the benchmark
                <ArrowUpRight />
              </LinkButton>
              <LinkButton href={p.repo} variant="outline">
                <Github />
                GitHub
              </LinkButton>
            </CardContent>
          </Card>
        </Container>
      </section>
    </Page>
  )
}
