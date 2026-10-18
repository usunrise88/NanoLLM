// Test helpers shared by the unit tests of the API layer.

/** A response whose body arrives in the given pieces, as the network splits it. */
export function chunked(...pieces: (string | Uint8Array)[]): Response {
  const enc = new TextEncoder()
  return new Response(
    new ReadableStream({
      start(c) {
        for (const p of pieces) c.enqueue(typeof p === 'string' ? enc.encode(p) : p)
        c.close()
      },
    }),
  )
}
