import { ArrowRight, ArrowUpRight, Code, Cpu, Github, StatsReport } from 'iconoir-react'

import { Container, Page } from '@/components/Layout'
import { Badge } from '@/components/ui/badge'
import { LinkButton } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { BENCHMARK, GITHUB, products } from '@/lib/products'

const principles = [
  {
    Icon: Cpu,
    title: 'No GPU required',
    text: 'Everything is tuned for the processor you already have: laptop, desktop or server.',
  },
  {
    Icon: Code,
    title: 'Open source',
    text: 'MIT licensed. Read it, run it, change it.',
  },
  {
    Icon: StatsReport,
    title: 'Measured in the open',
    text: 'A public benchmark shows how each setup runs on real hardware.',
    href: BENCHMARK,
    link: 'See the benchmark',
  },
] as const

export function Home() {
  return (
    <Page>
      <section className="relative overflow-hidden border-b">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(var(--border)_1px,transparent_1px)] [background-size:24px_24px] [mask-image:linear-gradient(to_bottom,black,transparent)]"
        />
        <Container className="relative py-20 sm:py-28">
          <Badge variant="outline">Nano*</Badge>
          <h1 className="mt-6 max-w-3xl text-4xl font-semibold tracking-tight text-balance sm:text-6xl">
            Decent AI for everyone.
            <span className="block text-muted-foreground">No GPU required.</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-muted-foreground text-pretty">
            Nano* is a family of small, fast runtimes that bring language, speech recognition and speech synthesis to
            ordinary computers: the laptops, desktops and servers people already own.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <LinkButton href="/nanollm/" size="lg">
              Explore NanoLLM
              <ArrowRight />
            </LinkButton>
            <LinkButton href={BENCHMARK} variant="outline" size="lg">
              View the benchmark
            </LinkButton>
          </div>
        </Container>
      </section>

      <section className="py-16">
        <Container>
          <h2 className="text-2xl font-semibold tracking-tight">Three solutions</h2>
          <p className="mt-2 text-muted-foreground">One idea, three products. Each runs on the CPU.</p>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {products.map(({ id, name, kicker, blurb, status, page, repo, Icon }) => (
              <Card key={id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <span className="inline-flex size-10 items-center justify-center rounded-lg border bg-background">
                      <Icon className="size-5" />
                    </span>
                    <Badge variant={status === 'Available' ? 'default' : 'secondary'}>{status}</Badge>
                  </div>
                  <CardTitle className="mt-4 text-xl">{name}</CardTitle>
                  <CardDescription className="font-medium text-foreground/70">{kicker}</CardDescription>
                </CardHeader>
                <CardContent className="flex-1 text-sm text-muted-foreground">{blurb}</CardContent>
                <CardFooter>
                  <LinkButton href={page} size="sm">
                    Learn more
                    <ArrowRight />
                  </LinkButton>
                  <LinkButton href={repo} variant="outline" size="sm" aria-label={`${name} on GitHub`}>
                    <Github />
                    GitHub
                  </LinkButton>
                </CardFooter>
              </Card>
            ))}
          </div>
        </Container>
      </section>

      <section className="border-t py-16">
        <Container>
          <div className="grid gap-8 md:grid-cols-3">
            {principles.map((p) => (
              <div key={p.title}>
                <p.Icon className="size-6" />
                <h3 className="mt-4 font-semibold">{p.title}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{p.text}</p>
                {'href' in p ? (
                  <a
                    href={p.href}
                    className="mt-3 inline-flex items-center gap-1 text-sm font-medium underline-offset-4 hover:underline"
                  >
                    {p.link}
                    <ArrowUpRight className="size-4" />
                  </a>
                ) : null}
              </div>
            ))}
          </div>
          <p className="mt-12 text-sm text-muted-foreground">
            All code lives on{' '}
            <a className="text-foreground underline underline-offset-4" href={GITHUB}>
              GitHub
            </a>
            .
          </p>
        </Container>
      </section>
    </Page>
  )
}
