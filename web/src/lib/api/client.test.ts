import { afterEach, describe, expect, it, vi } from 'vitest'

import { authRequired, setApiKey } from '@/lib/auth'

import { ApiError, listModels, pinModel, request, submitJob, toError, unloadModel, validateJsonl } from './client'
import { stream } from './sse'
import { chunked } from './testing'

afterEach(() => {
  vi.unstubAllGlobals()
  setApiKey('')
})

describe('toError', () => {
  it('reads a native problem', async () => {
    const e = await toError(
      new Response(JSON.stringify({ type: 'about:blank', title: 'Forbidden', status: 403, detail: 'needs admin', code: 'forbidden' }), {
        status: 403,
      }),
    )
    expect(e).toMatchObject({ status: 403, code: 'forbidden', message: 'needs admin' })
    expect(e.notYet).toBe(false)
  })

  it('reads an OpenAI error envelope', async () => {
    const e = await toError(
      new Response(JSON.stringify({ error: { message: 'n > 1', type: 'invalid_request_error', param: 'n', code: 'invalid_request' } }), {
        status: 400,
      }),
    )
    expect(e).toMatchObject({ status: 400, code: 'invalid_request', message: 'n > 1', param: 'n' })
  })

  // The two ways a route can be "not yet": the planned job routes answer a
  // 501 problem, a route not mounted at all gets the mux's plain-text 404.
  it('marks 501 and a bare 404 as not yet', async () => {
    const e501 = await toError(
      new Response(JSON.stringify({ status: 501, code: 'not_implemented', detail: 'the job queue lands in v1.0' }), { status: 501 }),
    )
    const e404 = await toError(new Response('404 page not found\n', { status: 404, statusText: 'Not Found' }))
    const model404 = await toError(new Response(JSON.stringify({ status: 404, code: 'model_not_found', detail: 'x' }), { status: 404 }))
    expect(e501.notYet).toBe(true)
    expect(e404.notYet).toBe(true)
    expect(model404.notYet).toBe(false)
  })

  it('keeps Retry-After', async () => {
    const e = await toError(
      new Response(JSON.stringify({ code: 'rate_limited' }), { status: 429, headers: { 'Retry-After': '3' } }),
    )
    expect(e.retryAfter).toBe(3)
  })
})

describe('request', () => {
  it('sends the key as a bearer token', async () => {
    const fetch = vi.fn(async (_url: string, _init?: RequestInit) => Response.json({ models: [{ id: 'm', disk: 'absent' }] }))
    vi.stubGlobal('fetch', fetch)
    setApiKey('  sk-test  ')
    const models = await listModels()
    expect(models[0]?.id).toBe('m')
    expect(fetch.mock.calls[0]?.[0]).toBe('/api/v1/models')
    expect((fetch.mock.calls[0]?.[1]?.headers as Record<string, string>).Authorization).toBe('Bearer sk-test')
  })

  it('sends no Authorization header without a key', async () => {
    const fetch = vi.fn(async (_url: string, _init?: RequestInit) => Response.json({}))
    vi.stubGlobal('fetch', fetch)
    await request('/config')
    expect(fetch.mock.calls[0]?.[1]?.headers).toEqual({})
  })

  // The first 401 is how the SPA learns the server wants a key (SPEC §11).
  it('reports a 401 so the key sheet can open', async () => {
    vi.stubGlobal('fetch', async () => Response.json({ code: 'unauthorized', detail: 'a valid API key is required' }, { status: 401 }))
    expect(authRequired()).toBe(false)
    const err = await request('/models').catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).code).toBe('unauthorized')
    expect(authRequired()).toBe(true)
  })

  it('treats 204 as no body', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 204 }))
    await expect(unloadModel('m')).resolves.toBeUndefined()
  })

  it('posts the pin flag as JSON', async () => {
    const fetch = vi.fn(async (_url: string, _init?: RequestInit) => Response.json({ id: 'a/b', pinned: false }))
    vi.stubGlobal('fetch', fetch)
    await pinModel('a/b', false)
    expect(fetch.mock.calls[0]?.[0]).toBe('/api/v1/models/a%2Fb/pin')
    expect(fetch.mock.calls[0]?.[1]?.body).toBe('{"pin":false}')
  })

  it('sends a batch as NDJSON, untouched', async () => {
    const fetch = vi.fn(async (_url: string, _init?: RequestInit) => Response.json({ id: 'job_1', status: 'queued', created_at: '' }))
    vi.stubGlobal('fetch', fetch)
    const body = '{"messages":[]}\n{"messages":[]}\n'
    await submitJob(body, 'gemma')
    expect(fetch.mock.calls[0]?.[0]).toBe('/api/v1/jobs?model=gemma')
    expect(fetch.mock.calls[0]?.[1]?.body).toBe(body)
    expect((fetch.mock.calls[0]?.[1]?.headers as Record<string, string>)['Content-Type']).toBe('application/x-ndjson')
  })
})

describe('validateJsonl', () => {
  it('counts lines and skips blank ones', () => {
    expect(validateJsonl('{"a":1}\n\n{"b":2}\n')).toEqual({ count: 2 })
  })

  it('points at the first bad line by number', () => {
    const r = validateJsonl('{"a":1}\n{"b":\n{"c":3}')
    expect(r.count).toBe(1)
    expect(r.error?.line).toBe(2)
  })

  it('refuses a line that is JSON but not an object', () => {
    expect(validateJsonl('[1,2]').error).toMatchObject({ line: 1, message: 'not a JSON object' })
  })
})

describe('stream (GET, resumable)', () => {
  it('delivers events until done and does not reconnect', async () => {
    const fetch = vi.fn(async (_url: string, _init?: RequestInit) =>
      chunked('id: 1\nevent: running\ndata: {"status":"running"}\n\nid: 2\nevent: done\ndata: {"status":"succeeded"}\n\n'),
    )
    vi.stubGlobal('fetch', fetch)
    const events: string[] = []
    let done = false
    await stream<{ status: string }>('/api/v1/jobs/j/events', {
      onEvent: (e, p) => events.push(`${e}:${p.status}`),
      onDone: () => {
        done = true
      },
    })
    expect(events).toEqual(['running:running', 'done:succeeded'])
    expect(done).toBe(true)
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('resumes from the last event id after a drop', async () => {
    const fetch = vi
      .fn(async (_url: string, _init?: RequestInit) => chunked(''))
      .mockResolvedValueOnce(chunked('id: 7\nevent: running\ndata: {"status":"running"}\n\n'))
      .mockResolvedValueOnce(chunked('id: 8\nevent: done\ndata: {"status":"succeeded"}\n\n'))
    vi.stubGlobal('fetch', fetch)
    vi.useFakeTimers({ toFake: ['setTimeout'] })
    const run = stream('/x', { onEvent: () => {} })
    await vi.runAllTimersAsync()
    await run
    vi.useRealTimers()
    expect(fetch).toHaveBeenCalledTimes(2)
    expect((fetch.mock.calls[1]?.[1]?.headers as Record<string, string>)['Last-Event-ID']).toBe('7')
  })

  // A 501 will still be a 501 in two seconds: say so at once.
  it.each([404, 501])('gives up at once on a %i', async (status) => {
    const fetch = vi.fn(async () => Response.json({ status, code: 'x', detail: 'no' }, { status }))
    vi.stubGlobal('fetch', fetch)
    let error: unknown
    await stream('/x', { onEvent: () => {}, onError: (e) => (error = e) })
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(error).toBeInstanceOf(ApiError)
  })
})
