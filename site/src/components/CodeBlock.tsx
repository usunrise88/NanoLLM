import { Check, Copy } from 'iconoir-react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'

/** A command to copy. Without scripts it is still text to select. */
export function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* the clipboard is blocked: the text is still selectable */
    }
  }

  return (
    <div className="relative rounded-lg border bg-muted">
      <pre className="p-4 pr-14 font-mono text-[13px] leading-relaxed break-all whitespace-pre-wrap">
        <code>{code}</code>
      </pre>
      <Button
        variant="ghost"
        size="icon"
        aria-label={copied ? 'Copied' : 'Copy the command'}
        title={copied ? 'Copied' : 'Copy'}
        onClick={copy}
        className="absolute top-2 right-2 size-8"
      >
        {copied ? <Check /> : <Copy />}
      </Button>
    </div>
  )
}
