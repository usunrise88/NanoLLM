import { Clock, Lock, Microphone, MultiBubble, Server, AlignLeft } from 'iconoir-react'

import { product } from '@/lib/products'

import { ProductPage } from './ProductPage'

const highlights = [
  {
    Icon: Lock,
    title: 'Offline and private',
    text: 'No audio leaves the machine. Models run locally; the network is only used to fetch model weights the first time.',
  },
  {
    Icon: Clock,
    title: 'Built for recordings',
    text: 'Calls, interviews and meetings from one minute to half an hour, mono or stereo, including 8 kHz telephony.',
  },
  {
    Icon: MultiBubble,
    title: 'Who said what',
    text: 'Word-level timestamps with confidence, speaker diarization, and SRT or VTT subtitles that carry speaker labels.',
  },
  {
    Icon: AlignLeft,
    title: 'Clean Russian text',
    text: 'Punctuation and casing, and inverse text normalisation: numbers, dates, times, currency, phone numbers and percentages.',
  },
  {
    Icon: Microphone,
    title: 'Streaming too',
    text: 'Realtime recognition over a websocket while the audio is still arriving, with partial hypotheses.',
  },
  {
    Icon: Server,
    title: 'One binary',
    text: 'The web interface is built in, four API dialects share one port, and a job queue handles webhooks. CPU by default, NVIDIA GPU with one flag.',
  },
] as const

export function NanoASR() {
  return (
    <ProductPage
      p={product('nanoasr')}
      lead="An offline speech recognition server built on sherpa-onnx. A single binary, CPU inference, word-level timestamps and an embedded web interface."
      highlights={highlights}
      install={{
        title: 'Install',
        code: 'curl -fsSL https://github.com/usunrise88/NanoASR/releases/latest/download/install.sh | bash',
        note: 'Linux, as a systemd service listening on 127.0.0.1:8080. It verifies the download, fetches the default models and prints the API keys. Windows has a PowerShell installer; see the repository.',
      }}
    />
  )
}
