// Fetches the favicons the pages show into favicons/, one PNG per host: each host a page names there (the alternatives
// it compares) and the address the ledger records for every shipped catalog. Google's favicon service renders a site's
// own icon, whatever its format, as a PNG of up to 64 px. Run after adding a catalog or an alternative.
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { pages } from './stamp.mjs'

// A catalog's icon is its source's: the host of its address in bench/foundries.json. A grouped catalog (Other) has none.
export const catalogIcon = url => url ? `favicons/${new URL(url).host}.png` : null

if (import.meta.main) {
  const read = async path => JSON.parse(await readFile(path, 'utf8'))
  const [site, ledger] = await Promise.all([read('site.json'), read('bench/foundries.json')]), byId = new Map(ledger.sources.map(s => [s.id, s]))
  const named = (await Promise.all(pages.map(page => readFile(page, 'utf8')))).flatMap(html => [...html.matchAll(/favicons\/([\w.-]+)\.png/g)].map(m => m[1]))
  const hosts = new Set([...named, ...site.catalogs.map(c => catalogIcon(byId.get(c.id)?.url)).filter(Boolean).map(path => path.slice(9, -4))])
  await mkdir('favicons', { recursive: true })
  for (const host of [...hosts].sort()) {
    const response = await fetch(`https://www.google.com/s2/favicons?domain=${host}&sz=64`)
    if (!response.ok || response.headers.get('content-type') !== 'image/png') throw new Error(`No favicon for ${host}: ${response.status}`)
    await writeFile(`favicons/${host}.png`, new Uint8Array(await response.arrayBuffer()))
    console.log(`favicons/${host}.png`)
  }
}
