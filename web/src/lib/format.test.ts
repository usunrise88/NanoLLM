import { describe, expect, it } from 'vitest'

import { bytes, ms, percent, tokS } from './format'

describe('format', () => {
  it('bytes', () => {
    expect(bytes(0)).toBe('0 B')
    expect(bytes(4275373792)).toBe('4.0 GiB')
    expect(bytes(17179869184)).toBe('16 GiB')
  })

  it('ms', () => {
    expect(ms(0.058)).toBe('0.06 ms')
    expect(ms(312.4)).toBe('312 ms')
    expect(ms(12_500)).toBe('12.5 s')
    expect(ms(undefined)).toBe('—')
  })

  it('tokS groups thousands and hides zero', () => {
    expect(tokS(276752.7, 'en')).toBe('276,753 tok/s')
    expect(tokS(42.25, 'en')).toBe('42.3 tok/s')
    expect(tokS(0)).toBe('—')
  })

  it('percent', () => {
    expect(percent(0.5833)).toBe('58%')
    expect(percent(undefined)).toBe('—')
  })
})
