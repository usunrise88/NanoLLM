/**
 * Formatting shared by every screen. A number shown two ways in two places
 * reads as two different numbers, so each has exactly one implementation.
 */

const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB']

export function bytes(n: number | undefined): string {
  if (n === undefined || !Number.isFinite(n) || n <= 0) return '0 B'
  const i = Math.min(UNITS.length - 1, Math.floor(Math.log(n) / Math.log(1024)))
  const value = n / 1024 ** i
  return `${value < 10 && i > 0 ? value.toFixed(1) : Math.round(value)} ${UNITS[i]}`
}

/** Milliseconds: two decimals under ten, whole numbers above, seconds past ten thousand. */
export function ms(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return '—'
  if (v >= 60_000) return `${(v / 1000).toFixed(0)} s`
  if (v >= 10_000) return `${(v / 1000).toFixed(1)} s`
  if (v >= 10) return `${Math.round(v)} ms`
  return `${v.toFixed(2)} ms`
}

/** Tokens per second, grouped so 275000 reads as 275,000 rather than a phone number. */
export function tokS(v: number | undefined, locale = 'en'): string {
  if (v === undefined || !Number.isFinite(v) || v <= 0) return '—'
  const digits = v >= 100 ? 0 : 1
  return `${v.toLocaleString(locale, { maximumFractionDigits: digits, minimumFractionDigits: digits })} tok/s`
}

export function percent(fraction: number | undefined): string {
  if (fraction === undefined || !Number.isFinite(fraction)) return '—'
  return `${Math.round(fraction * 100)}%`
}

export function seconds(s: number | undefined): string {
  if (s === undefined || !Number.isFinite(s)) return '—'
  if (s < 90) return `${Math.round(s)} s`
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`
}

/** A time a person can read at a glance, in their own locale. */
export function when(iso: string | undefined, locale: string): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  const elapsed = Date.now() - date.getTime()
  if (elapsed < 86_400_000 && elapsed > -60_000) {
    return date.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' })
  }
  return date.toLocaleDateString(locale, { day: 'numeric', month: 'short' })
}
