import { Github } from 'iconoir-react'
import type { ReactNode } from 'react'

import { GITHUB, BENCHMARK, products } from '@/lib/products'
import { cn } from '@/lib/cn'

import { ThemeToggle } from './ThemeToggle'

export function Container({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn('mx-auto w-full max-w-5xl px-6', className)} {...props} />
}

function NavLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      className="rounded-md px-2 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
    >
      {children}
    </a>
  )
}

export function Header() {
  return (
    <header className="sticky top-0 z-20 border-b bg-background/80 backdrop-blur">
      <Container className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 py-3">
        <a href="/" className="text-lg font-semibold tracking-tight">
          Nano<span className="text-muted-foreground">*</span>
        </a>
        <nav
          aria-label="Main"
          className="order-last -mx-2 flex w-full items-center gap-1 overflow-x-auto sm:order-none sm:mx-0 sm:w-auto"
        >
          {products.map((p) => (
            <NavLink key={p.id} href={p.page}>
              {p.name}
            </NavLink>
          ))}
          <NavLink href={BENCHMARK}>Benchmark</NavLink>
        </nav>
        <div className="flex items-center gap-2">
          <a
            href={GITHUB}
            aria-label="Nano* on GitHub"
            title="GitHub"
            className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          >
            <Github className="size-5" />
          </a>
          <ThemeToggle />
        </div>
      </Container>
    </header>
  )
}

export function Footer() {
  return (
    <footer className="mt-24 border-t">
      <Container className="flex flex-col gap-6 py-10 text-sm text-muted-foreground sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-xs">
          <p className="font-semibold text-foreground">
            Nano<span className="text-muted-foreground">*</span>
          </p>
          <p className="mt-2">Decent AI for everyone, no GPU required.</p>
        </div>
        <div className="flex gap-12">
          <ul className="flex flex-col gap-2">
            {products.map((p) => (
              <li key={p.id}>
                <a className="hover:text-foreground" href={p.page}>
                  {p.name}
                </a>
              </li>
            ))}
          </ul>
          <ul className="flex flex-col gap-2">
            <li>
              <a className="hover:text-foreground" href={BENCHMARK}>
                Benchmark
              </a>
            </li>
            <li>
              <a className="hover:text-foreground" href={GITHUB}>
                GitHub
              </a>
            </li>
            <li>MIT licensed</li>
          </ul>
        </div>
      </Container>
    </footer>
  )
}

export function Page({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-30 focus:rounded-md focus:bg-primary focus:px-3 focus:py-2 focus:text-primary-foreground"
      >
        Skip to content
      </a>
      <Header />
      <main id="main" className="flex-1">
        {children}
      </main>
      <Footer />
    </div>
  )
}
