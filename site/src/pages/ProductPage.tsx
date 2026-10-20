import { ArrowLeft, Github } from 'iconoir-react'
import type { ComponentType, ReactNode, SVGProps } from 'react'

import { CodeBlock } from '@/components/CodeBlock'
import { Container, Page } from '@/components/Layout'
import { Badge } from '@/components/ui/badge'
import { LinkButton } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { Product } from '@/lib/products'

export interface Highlight {
  Icon: ComponentType<SVGProps<SVGSVGElement>>
  title: string
  text: string
}

export interface ProductPageProps {
  p: Product
  lead: string
  highlights: readonly Highlight[]
  install: { title: string; code: string; note: ReactNode }
}

/** A short product page: what it is, what it does, how to install it, where the code is. */
export function ProductPage({ p, lead, highlights, install }: ProductPageProps) {
  return (
    <Page>
      <section className="border-b">
        <Container className="py-16 sm:py-24">
          <div className="flex flex-wrap items-center gap-3">
            <Badge variant="outline">{p.kicker}</Badge>
            <Badge variant={p.status === 'Available' ? 'default' : 'secondary'}>{p.status}</Badge>
          </div>
          <h1 className="mt-6 text-4xl font-semibold tracking-tight sm:text-6xl">{p.name}</h1>
          <p className="mt-6 max-w-2xl text-lg text-muted-foreground text-pretty">{lead}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <LinkButton href={p.repo} size="lg">
              <Github />
              Source on GitHub
            </LinkButton>
            <LinkButton href="/" variant="outline" size="lg">
              <ArrowLeft />
              All Nano* products
            </LinkButton>
          </div>
        </Container>
      </section>

      <section className="py-16">
        <Container>
          <h2 className="text-2xl font-semibold tracking-tight">What it does</h2>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {highlights.map(({ Icon, title, text }) => (
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
        <Container className="max-w-3xl">
          <h2 className="text-2xl font-semibold tracking-tight">{install.title}</h2>
          <div className="mt-6">
            <CodeBlock code={install.code} />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">{install.note}</p>
        </Container>
      </section>
    </Page>
  )
}
