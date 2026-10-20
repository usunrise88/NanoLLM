// Writes every page of the site as static HTML: the client build's index.html is
// the template, the server build renders each path into it.
//
//   npm run build      (vite build, vite build --ssr, then this)
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const dist = join(root, 'dist')
const { render, paths } = await import(pathToFileURL(join(root, 'dist-ssr', 'entry-server.js')).href)

const template = await readFile(join(dist, 'index.html'), 'utf8')
for (const marker of ['<!--app-head-->', '<!--app-html-->']) {
  if (!template.includes(marker)) throw new Error(`index.html lost ${marker}: the template and the build disagree`)
}

const fill = (page) => template.replace('<!--app-head-->', page.head).replace('<!--app-html-->', page.html)

for (const path of paths) {
  const file = path === '/' ? join(dist, 'index.html') : join(dist, path, 'index.html')
  await mkdir(dirname(file), { recursive: true })
  await writeFile(file, fill(render(path)))
  console.log(`prerendered ${path}`)
}
await writeFile(join(dist, '404.html'), fill(render('/404', false)))
console.log('prerendered 404.html')
