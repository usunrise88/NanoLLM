import { ArrowLeft } from 'iconoir-react'

import { Container, Page } from '@/components/Layout'
import { LinkButton } from '@/components/ui/button'

export function NotFound() {
  return (
    <Page>
      <Container className="py-24">
        <p className="text-sm text-muted-foreground">404</p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight">Page not found</h1>
        <p className="mt-4 text-muted-foreground">There is nothing at this address.</p>
        <LinkButton href="/" className="mt-8">
          <ArrowLeft />
          Back to Nano*
        </LinkButton>
      </Container>
    </Page>
  )
}
