import { StrictMode } from 'react'
import { renderToString } from 'react-dom/server'

import { notFound, pageFor, pages } from './pages'

export const paths = Object.keys(pages)

function escape(s: string) {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;')
}

/** The markup of a path and the <head> lines that belong to it. */
export function render(path: string, known = true) {
  const page = known ? pageFor(path) : notFound
  const html = renderToString(
    <StrictMode>
      <page.Component />
    </StrictMode>,
  )
  const head = [
    `<title>${escape(page.title)}</title>`,
    `<meta name="description" content="${escape(page.description)}" />`,
    `<meta property="og:title" content="${escape(page.title)}" />`,
    `<meta property="og:description" content="${escape(page.description)}" />`,
    `<meta property="og:type" content="website" />`,
    known ? '' : '<meta name="robots" content="noindex" />',
  ]
    .filter(Boolean)
    .join('\n    ')
  return { html, head }
}
