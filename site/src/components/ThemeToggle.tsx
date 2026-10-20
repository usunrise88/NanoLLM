import { Toggle } from '@base-ui/react/toggle'
import { ToggleGroup } from '@base-ui/react/toggle-group'
import { HalfMoon, Laptop, SunLight } from 'iconoir-react'
import { useEffect, useState } from 'react'

import { applyTheme, isTheme, storedTheme, type Theme } from '@/lib/theme'

const options: { value: Theme; label: string; Icon: typeof Laptop }[] = [
  { value: 'system', label: 'Match the system theme', Icon: Laptop },
  { value: 'light', label: 'Light theme', Icon: SunLight },
  { value: 'dark', label: 'Dark theme', Icon: HalfMoon },
]

/**
 * System, light or dark. The page starts on the system theme (the inline script
 * in index.html has already set the class); this reads the remembered choice
 * after hydration, so the server and the client render the same first frame.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>('system')

  useEffect(() => {
    setTheme(storedTheme())
  }, [])

  // On "system" the page follows the OS while it is open.
  useEffect(() => {
    if (theme !== 'system') return
    const mq = matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => applyTheme('system')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [theme])

  return (
    <ToggleGroup
      value={[theme]}
      onValueChange={(v) => {
        const next = v[0]
        if (!isTheme(next)) return // pressing the pressed one: keep it
        setTheme(next)
        applyTheme(next)
      }}
      aria-label="Theme"
      className="inline-flex items-center rounded-lg border bg-background p-0.5"
    >
      {options.map(({ value, label, Icon }) => (
        <Toggle
          key={value}
          value={value}
          aria-label={label}
          title={label}
          className="inline-flex size-7 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring data-[pressed]:bg-accent data-[pressed]:text-foreground"
        >
          <Icon className="size-4" />
        </Toggle>
      ))}
    </ToggleGroup>
  )
}
