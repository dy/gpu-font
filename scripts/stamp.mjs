// Pins each page to the exact stylesheets and modules it was written with. GitHub Pages lets a browser keep every file for
// ten minutes on its own, so after a deploy a page could run with another deploy's script or styles. Each stylesheet and
// the entry module get their content hash in the URL, and an import map does the same for every module the entry
// imports. Run after changing any page, stylesheet or module: node scripts/stamp.mjs
import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { posix } from 'node:path'

export const pages = ['index.html', 'catalogs.html']
const hash = async path => createHash('sha256').update(await readFile(path)).digest('hex').slice(0, 10)

// Local modules the entry reaches through static imports, the entry included.
async function graph(path, seen = new Set()) {
  if (seen.has(path)) return seen
  seen.add(path)
  for (const [, spec] of (await readFile(path, 'utf8')).matchAll(/from '(\.\.?\/[^']+)'/g)) await graph(posix.join(posix.dirname(path), spec), seen)
  return seen
}

// A page with current stamps; unchanged when they already are.
export async function stamp(html) {
  const versioned = async path => `./${path}?v=${await hash(path)}`
  for (const [tag, path] of html.matchAll(/<link rel="stylesheet" href="\.\/([^"?]+\.css)(?:\?v=\w+)?">/g)) html = html.replace(tag, `<link rel="stylesheet" href="${await versioned(path)}">`)
  const [tag, entry] = html.match(/<script type="module" src="\.\/([^"?]+\.mjs)(?:\?v=\w+)?"><\/script>/)
  html = html.replace(tag, `<script type="module" src="${await versioned(entry)}"></script>`)
  const modules = [...await graph(entry)].filter(path => path !== entry).sort()
  const map = `<script type="importmap">${JSON.stringify({ imports: Object.fromEntries(await Promise.all(modules.map(async path => [`./${path}`, await versioned(path)]))) })}</script>`
  const old = html.match(/<script type="importmap">.*?<\/script>/s)?.[0]
  // The map precedes every module script, so it goes in the head, before the first stylesheet.
  return old ? html.replace(old, map) : html.replace(/( *)<link rel="stylesheet"/, (line, indent) => `${indent}${map}\n${line}`)
}

if (import.meta.main) for (const page of pages) await writeFile(page, await stamp(await readFile(page, 'utf8')))
