import { useState, type ReactNode } from 'react'
import { NavArrowDown, NavArrowRight } from 'iconoir-react'

/**
 * A JSON value shown as the server sent it, indented. Read-only on purpose:
 * the effective configuration and placement plans are facts to look at, and
 * the one JSON document a person acts on (a benchmark submission) must be
 * shown exactly as it will be sent.
 */
export function JsonBlock({ value, label }: { value: unknown; label?: string }) {
  return (
    <pre
      aria-label={label}
      className="max-h-[32rem] overflow-auto rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--bg-canvas)] p-3 font-[var(--font-mono)] text-[12px] leading-5"
    >
      {typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
    </pre>
  )
}

/** A JSON block behind a disclosure, for detail most people never open. */
export function Collapsible({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[12px] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
      >
        {open ? <NavArrowDown width={12} height={12} /> : <NavArrowRight width={12} height={12} />}
        {title}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  )
}

/** A text file handed to the browser to save, without a round trip. */
export function saveText(name: string, text: string, type = 'application/json'): void {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
