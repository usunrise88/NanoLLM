import type { ComponentType } from 'react'

import { Home } from './pages/Home'
import { NanoASR } from './pages/NanoASR'
import { NanoLLM } from './pages/NanoLLM'
import { NanoTTS } from './pages/NanoTTS'
import { NotFound } from './pages/NotFound'

export interface PageDef {
  title: string
  description: string
  Component: ComponentType
}

const TAGLINE = 'Decent AI for everyone, no GPU required.'

/** Every page of the site, by path. Each is prerendered to <path>/index.html. */
export const pages: Record<string, PageDef> = {
  '/': {
    title: 'Nano*: decent AI for everyone, no GPU required',
    description: `${TAGLINE} NanoLLM, NanoASR and NanoTTS: language, speech recognition and speech synthesis on the CPU.`,
    Component: Home,
  },
  '/nanollm/': {
    title: 'NanoLLM: Gemma 4 on the CPU',
    description: 'A cross-platform LLM runtime for edge computing: Gemma 4 E4B in 4 bits, an OpenAI-compatible API, no GPU.',
    Component: NanoLLM,
  },
  '/nanoasr/': {
    title: 'NanoASR: speech recognition on the CPU',
    description: 'An offline speech recognition server on sherpa-onnx: word timestamps, speakers, subtitles. One binary, on the CPU.',
    Component: NanoASR,
  },
  '/nanotts/': {
    title: 'NanoTTS: speech synthesis on the CPU',
    description: 'A speech synthesis server for small models such as Pocket TTS and Supertonic 3, with an OpenAI-compatible API. CPU only.',
    Component: NanoTTS,
  },
}

export const notFound: PageDef = {
  title: 'Page not found: Nano*',
  description: 'There is nothing at this address.',
  Component: NotFound,
}

/** The page for a path; "/nanollm" and "/nanollm/" are the same page. */
export function pageFor(path: string): PageDef {
  const p = path.endsWith('/') ? path : `${path}/`
  return pages[p] ?? notFound
}
