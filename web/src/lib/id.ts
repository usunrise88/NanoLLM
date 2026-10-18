/**
 * A random id for a conversation or its session.
 *
 * crypto.randomUUID exists only in a secure context, and a server reached at
 * http://192.168.x.y is not one; getRandomValues is available everywhere.
 */
export function randomId(prefix = ''): string {
  const bytes = new Uint8Array(12)
  crypto.getRandomValues(bytes)
  let hex = ''
  for (const b of bytes) hex += b.toString(16).padStart(2, '0')
  return prefix + hex
}
