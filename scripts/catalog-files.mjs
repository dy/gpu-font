// Catalogs of the free families pinned in bench/open-fonts.json, one per source, indexed in Chromium by the page's
// own builder (src/references.mjs) from the pinned files in .data/fonts-open: the Latin reference lines, rasterizer,
// preparation and int8 encoder that My fonts uses. A file is served to the page only when its bytes match its pinned git
// blob. Writes .data/catalogs/files/<source>.json; `npm run demo:build` ships those whose terms allow it.
//
//   node scripts/catalog-files.mjs [source ...]     with the demo server running (npm run demo)
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { joinCatalogs } from '../src/references.mjs'

const read = async path => JSON.parse(await readFile(path, 'utf8'))
const [inventory, registry, site] = await Promise.all(['bench/open-fonts.json', 'bench/foundries.json', 'site.json'].map(read))
const names = new Map(registry.sources.map(s => [s.id, s.name.replace(/\s*\(.*\)$/, '')])), homes = new Map(registry.sources.map(s => [s.id, s.url]))
const WEIGHTS = { 100: 'Thin', 200: 'ExtraLight', 300: 'Light', 400: 'Regular', 500: 'Medium', 600: 'SemiBold', 700: 'Bold', 800: 'ExtraBold', 900: 'Black' }
const styleName = (weight, italic) => { const name = WEIGHTS[Math.min(900, Math.max(100, Math.round(weight / 100) * 100))]; return italic ? (name === 'Regular' ? 'Italic' : `${name} Italic`) : name }
// Text catalogs index families with Latin letters; colour, letterless and symbol-encoded families wait for a glyph mode.
const eligible = family => !family.excluded && !family.symbolEncoded && family.letters?.Latn
const wanted = process.argv.slice(2), sources = Map.groupBy(inventory.families.filter(f => eligible(f) && (!wanted.length || wanted.includes(f.source))), f => f.source)
const blob = bytes => createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex')
const pins = new Map(inventory.families.flatMap(f => f.faces.map(face => [face.path, face.blob])))
const base = `http://127.0.0.1:${process.env.PORT || 4179}`

// A font can crash the renderer or the browser: faces go in batches, a crashed batch is retried one face at a time on a fresh page,
// and a face that crashes it alone is skipped and reported.
const launch = () => chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
let browser = await launch()
// A crash may take the whole browser down, not just its page.
async function session() {
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
    return await page.evaluate(async ({ faces, source }) => {
      const { buildCatalog } = await import('/src/references.mjs'), e = window.encoder
      return buildCatalog(faces, { project: e.project, preparation: e.preparation, binding: e.binding, source })
    }, { faces, source })
  } catch (error) {
    await page.close().catch(() => {}); page = await session()
    if (faces.length === 1) return { catalog: null, skipped: faces, crashed: true }
    const parts = []
    for (const face of faces) parts.push(await build([face], source))
    const built = parts.map(p => p.catalog).filter(Boolean)
    return { catalog: built.length ? built.reduce((all, next) => joinCatalogs(all, next)) : null, skipped: parts.flatMap(p => p.skipped) }
  }
}
const BATCH = 100
try {
  await mkdir('.data/catalogs/files', { recursive: true })
  for (const [source, families] of sources) {
    const faces = families.flatMap(family => family.faces.map(face => ({
      id: `${family.id}/${face.path.split('/').at(-1).replace(/\.[^.]+$/, '')}`, familyId: family.id, family: family.family,
      styleName: styleName(face.weight, face.italic), weight: face.weight, style: face.italic ? 'italic' : 'normal', scripts: Object.keys(family.letters).sort(),
      sourceUrl: family.sourceUrl ?? homes.get(source), ...(face.path === family.selected ? { default: true } : {}), // A family without its own page links to its source's.
      source: `url("/fonts-open/${face.path.split('/').map(encodeURIComponent).join('/')}")` })))
    let catalog = null, skipped = []
    for (let i = 0; i < faces.length; i += BATCH) {
      process.stdout.write(`\r${source} ${i}/${faces.length}   `)
      const part = await build(faces.slice(i, i + BATCH), source)
      if (part.catalog) catalog = catalog ? joinCatalogs(catalog, part.catalog) : part.catalog
      skipped.push(...part.skipped)
    }
    if (!catalog) { console.log(`\n${source}: nothing indexed`); continue }
    await writeFile(`.data/catalogs/files/${source}.json`, JSON.stringify({ ...catalog, name: names.get(source) ?? source }) + '\n')
    const indexed = new Set(catalog.faces.map(f => f.familyId)).size
    console.log(`\n${source}: ${indexed} families, ${catalog.faces.length} faces${skipped.length ? `; skipped ${skipped.length}: ${skipped.map(f => f.id).join(', ')}` : ''}`)
  }
} finally { await browser.close() }
