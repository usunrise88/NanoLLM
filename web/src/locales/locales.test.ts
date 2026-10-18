import { describe, expect, it } from 'vitest'

import enRaw from './en.json?raw'
import ruRaw from './ru.json?raw'

/**
 * Duplicate keys in a catalogue.
 *
 * JSON.parse keeps the last of two equal keys without a word, and the type
 * check sees only the parsed shape — so a second "stop" quietly relabelled
 * the chat's Stop button as "Stop sequences". The files are formatted one
 * key per line, which is all this needs to find them.
 */
function duplicates(raw: string): string[] {
  const stack: Set<string>[] = []
  const path: string[] = []
  const out: string[] = []
  for (const line of raw.split('\n')) {
    const key = /^\s*"([^"]+)":/.exec(line)?.[1]
    if (key !== undefined) {
      const seen = stack.at(-1)
      if (seen?.has(key)) out.push([...path, key].join('.'))
      seen?.add(key)
    }
    if (/\{\s*$/.test(line)) {
      stack.push(new Set())
      if (key !== undefined) path.push(key)
    } else if (/^\s*\},?\s*$/.test(line)) {
      stack.pop()
      path.pop()
    }
  }
  return out
}

describe('locales', () => {
  it.each([
    ['en', enRaw],
    ['ru', ruRaw],
  ])('%s has no duplicate keys', (_name, raw) => {
    expect(duplicates(raw)).toEqual([])
  })

  it('finds a duplicate when there is one', () => {
    expect(duplicates('{\n  "a": {\n    "x": "1",\n    "x": "2"\n  }\n}\n')).toEqual(['a.x'])
  })
})
