// Catalogs of the free families pinned in bench/open-fonts.json, one per source, indexed in Chromium by the page's
// own builder (src/references.mjs) from the pinned files in .data/fonts-open: the Latin reference lines, rasterizer,
// preparation and int8 encoder that My fonts uses. A variable file with named instances is indexed as its static
// instances (scripts/instances.py), one face each. A file is served to the page only when its bytes match its pinned git
// blob. Writes .data/catalogs/files/<source>.json; `npm run demo:build` ships those whose terms allow it.
//
//   node scripts/catalog-files.mjs [--short] [source ...]     with the demo server running (npm run demo), after scripts/instances.py
//
// --short adds the cover line (src/references.mjs SHORT) as a row of each face and writes <source>-short.json beside the catalog: a candidate to measure, never shipped by this flag.
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { chromium } from 'playwright'
import { joinCatalogs } from '../src/references.mjs'
import { catalogFaces } from './catalog-faces.mjs'

const read = async path => JSON.parse(await readFile(path, 'utf8'))
const [inventory, registry, site] = await Promise.all(['bench/open-fonts.json', 'bench/foundries.json', 'site.json'].map(read))
const names = new Map(registry.sources.map(s => [s.id, s.name.replace(/\s*\(.*\)$/, '')])), homes = new Map(registry.sources.map(s => [s.id, s.url]))
// The static instances of the variable files, when scripts/instances.py has made them; without the manifest each file is one face at its default.
const instances = await read(`${inventory.store}/instances.json`).catch(() => { console.log('No instances.json: variable files index at their default; run node scripts/python.mjs scripts/instances.py'); return { faces: {}, families: {} } })
// Text catalogs index families with Latin letters; colour, letterless, icon and symbol-encoded families wait for a glyph mode.
const eligible = family => !family.excluded && !family.symbolEncoded && family.letters?.Latn
const short = process.argv.includes('--short'), wanted = process.argv.slice(2).filter(a => a !== '--short'), sources = Map.groupBy(inventory.families.filter(f => eligible(f) && (!wanted.length || wanted.includes(f.source))), f => f.source)
const blob = bytes => createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex')
const pins = new Map([...inventory.families.flatMap(f => f.faces.map(face => [face.path, face.blob])), ...Object.values(instances.faces).flat().map(i => [i.path, i.blob])])
const base = `http://127.0.0.1:${process.env.PORT || 4179}`

// A font can crash the renderer or the browser: faces go in batches, a crashed batch is retried one face at a time on a fresh page,
// and a face that crashes it alone is skipped and reported.
const launch = () => chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
let browser = await launch()
// A crash may take the whole browser down, not just its page, and a lost GPU process leaves the next page without WebGPU: a browser
// that cannot open a working page is relaunched.
async function session(tries = 3) {
  try { return await fresh() } catch (error) {
    if (tries === 1) throw error
    await browser.close().catch(() => {}); browser = await launch()
    return session(tries - 1)
  }
}
async function fresh() {
  if (!browser.isConnected()) browser = await launch()
  const page = await browser.newPage()
  await page.route('**/catalog-files', route => route.fulfill({ contentType: 'text/html', body: '<title>Catalog files</title>' }))
  await page.route('**/fonts-open/**', async route => {
    const path = decodeURIComponent(new URL(route.request().url()).pathname.replace(/^\/fonts-open\//, '')), bytes = await readFile(`${inventory.store}/${path}`).catch(() => null)
    if (!bytes || blob(bytes) !== pins.get(path)) return route.abort()
    route.fulfill({ body: bytes, contentType: 'font/ttf' })
  })
  await page.goto(`${base}/catalog-files`)
  await page.evaluate(async model => { const { createEncoder } = await import('/src/match.mjs'); window.encoder = await createEncoder(model) }, site.model)
  return page
}
let page = await session()
async function build(faces, source) {
  try {
    return await page.evaluate(async ({ faces, source, short }) => {
      const { buildCatalog } = await import('/src/references.mjs'), e = window.encoder
      return buildCatalog(faces, { project: e.project, preparation: e.preparation, binding: e.binding, source, short })
    }, { faces, source, short })
  } catch (error) {
    await page.close().catch(() => {}); page = await session()
    if (faces.length === 1) return { catalog: null, skipped: faces, crashed: true }
    const parts = []
    for (const face of faces) parts.push(await build([face], source))
    const built = parts.map(p => p.catalog).filter(Boolean)
    return { catalog: built.length ? built.reduce((all, next) => joinCatalogs(all, next)) : null, skipped: parts.flatMap(p => p.skipped) }
  }
}
// Twins across every catalog, from the teacher (train.style twins): a default face names the families its letters cannot be told from.
const twins = await readFile('.data/style/twins.json', 'utf8').then(text => JSON.parse(text).families, () => { console.log('No .data/style/twins.json: faces carry no twins; run node scripts/python.mjs -m train.style twins'); return {} })
const BATCH = 100
try {
  await mkdir('.data/catalogs/files', { recursive: true })
  // A source left with no family to index (Font Awesome's, whose letters only spell icons) loses its earlier catalog.
  for (const source of new Set(inventory.families.map(f => f.source))) if (!sources.has(source) && (!wanted.length || wanted.includes(source))) await rm(`.data/catalogs/files/${source}.json`, { force: true })
  for (const [source, families] of sources) {
    const faces = families.flatMap(family => catalogFaces(family, { instances, twins, home: homes.get(source) }).map(({ path, ...face }) => ({ ...face, source: `url("/fonts-open/${path.split('/').map(encodeURIComponent).join('/')}")` })))
    let catalog = null, skipped = []
    for (let i = 0; i < faces.length; i += BATCH) {
      process.stdout.write(`\r${source} ${i}/${faces.length}   `)
      const part = await build(faces.slice(i, i + BATCH), source)
      if (part.catalog) catalog = catalog ? joinCatalogs(catalog, part.catalog) : part.catalog
      skipped.push(...part.skipped)
    }
    if (!catalog) { console.log(`\n${source}: nothing indexed`); continue }
    await writeFile(`.data/catalogs/files/${source}${short ? '-short' : ''}.json`, JSON.stringify({ ...catalog, name: names.get(source) ?? source }) + '\n')
    const indexed = new Set(catalog.faces.map(f => f.familyId)).size
    console.log(`\n${source}: ${indexed} families, ${catalog.faces.length} faces${skipped.length ? `; skipped ${skipped.length}: ${skipped.map(f => f.id).join(', ')}` : ''}`)
  }
} catch (error) { process.exitCode = 1; throw error } finally {
  // The page holds a WebGPU device; released first, the browser closes. One that still hangs after 15 s is killed: every catalog is written by now.
  await page.evaluate(() => window.encoder?.destroy()).catch(() => {}); await page.close().catch(() => {})
  setTimeout(() => { console.error('The browser did not close in 15 s; killed'); spawnSync('pkill', ['-P', String(process.pid)]); process.exit() }, 15000).unref()
  await browser.close()
}
