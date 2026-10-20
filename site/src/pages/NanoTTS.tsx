import { Cpu, Globe, SoundHigh, Server, Timer } from 'iconoir-react'

import { product } from '@/lib/products'

import { ProductPage } from './ProductPage'

const highlights = [
  {
    Icon: SoundHigh,
    title: 'Small models, fast on a CPU',
    text: "Kyutai's Pocket TTS and its fine-tunes, and the Supertonic 3 family: Supertonic 3 and TeraTTS v2.",
  },
  {
    Icon: Server,
    title: 'OpenAI-compatible API',
    text: 'An HTTP API that follows the OpenAI shape, with streaming from the first audio frame.',
  },
  {
    Icon: Cpu,
    title: 'Tuned for serving',
    text: 'Multi-threaded and NUMA-aware: weights are replicated per NUMA node, and warmed voices start from a memory copy.',
  },
  {
    Icon: Globe,
    title: 'Many languages',
    text: 'English, French, German, Italian, Portuguese, Spanish, Russian, Korean and Japanese, depending on the model you pick.',
  },
  {
    Icon: Timer,
    title: 'One command',
    text: 'The installer builds the runtime image in Docker and starts the server on 127.0.0.1:8080. A launcher, nanotts, manages it.',
  },
] as const

export function NanoTTS() {
  return (
    <ProductPage
      p={product('nanotts')}
      lead="A CPU inference server for small speech synthesis models, rewritten for multi-threaded serving with an OpenAI-compatible HTTP API."
      highlights={highlights}
      install={{
        title: 'Install',
        code: 'curl -fsSL https://raw.githubusercontent.com/usunrise88/NanoTTS/main/install.sh | bash',
        note: 'Preview: it installs from source, there is no tagged release yet. Needs Linux on x86-64 (AVX2 preferred), Docker with Compose v2, git and about 12 GiB of disk.',
      }}
    />
  )
}
