export type Theme = 'system' | 'light' | 'dark'

export const THEMES: readonly Theme[] = ['system', 'light', 'dark']

const KEY = 'theme'

export function isTheme(v: unknown): v is Theme {
  return v === 'system' || v === 'light' || v === 'dark'
}

/** The stored choice; storage can be blocked, and then the system decides. */
export function storedTheme(): Theme {
  try {
    const v = localStorage.getItem(KEY)
    return isTheme(v) ? v : 'system'
  } catch {
    return 'system'
  }
}

export function prefersDark(): boolean {
  return matchMedia('(prefers-color-scheme: dark)').matches
}

/** Puts the choice on <html> (the .dark class Radix Colors keys on) and remembers it. */
export function applyTheme(theme: Theme) {
  const dark = theme === 'dark' || (theme === 'system' && prefersDark())
  document.documentElement.classList.toggle('dark', dark)
  try {
    if (theme === 'system') localStorage.removeItem(KEY)
    else localStorage.setItem(KEY, theme)
  } catch {
    /* a private window: the choice lasts until the page closes */
  }
}
