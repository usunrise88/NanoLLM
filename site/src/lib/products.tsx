import { ChatBubble, Microphone, SoundHigh } from 'iconoir-react'
import type { ComponentType, SVGProps } from 'react'

export const GITHUB = 'https://github.com/usunrise88'
export const BENCHMARK = '/benchmarks'

export interface Product {
  id: 'nanollm' | 'nanoasr' | 'nanotts'
  name: string
  kicker: string
  blurb: string
  status: 'Available' | 'Preview' | 'In development'
  page: string
  repo: string
  Icon: ComponentType<SVGProps<SVGSVGElement>>
}

/** The three solutions, in the order the site shows them. */
export const products: readonly Product[] = [
  {
    id: 'nanollm',
    name: 'NanoLLM',
    kicker: 'Language',
    blurb: 'Chat, tools and structured output with Gemma 4, quantized to 4 bits, served from one binary on the CPU.',
    status: 'In development',
    page: '/nanollm/',
    repo: `${GITHUB}/NanoLLM`,
    Icon: ChatBubble,
  },
  {
    id: 'nanoasr',
    name: 'NanoASR',
    kicker: 'Speech recognition',
    blurb: 'An offline speech recognition server on sherpa-onnx: word timestamps, speakers, subtitles. One binary, on the CPU.',
    status: 'Available',
    page: '/nanoasr/',
    repo: `${GITHUB}/NanoASR`,
    Icon: Microphone,
  },
  {
    id: 'nanotts',
    name: 'NanoTTS',
    kicker: 'Speech synthesis',
    blurb: 'A speech synthesis server for small models such as Pocket TTS and Supertonic 3, with an OpenAI-compatible API. CPU only.',
    status: 'Preview',
    page: '/nanotts/',
    repo: `${GITHUB}/NanoTTS`,
    Icon: SoundHigh,
  },
]

export function product(id: Product['id']): Product {
  const p = products.find((x) => x.id === id)
  if (!p) throw new Error(`unknown product ${id}`)
  return p
}
