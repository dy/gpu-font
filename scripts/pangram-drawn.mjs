// Pangram Pangram's product pictures with the foundry's own faces drawable: what the same-text stage would do for a catalog
// whose fonts the page could load. Each picture (or, --lines, its tallest Vision line as scripts/pangram-specimens.mjs crops
// it) is read as the page reads it and ranked across every catalog to --limit rows; then the shortlist is set in the words
// Vision read and in the words the shipped reader read, with Pangram's faces drawn from .data/fonts-open and the Google
// faces from Google Fonts. An analysis, not a report: the rows go to --out (.data/pangram/drawn.json), with each picture's
// rank by the catalog alone, among Pangram's rows alone, and by the same-text stage over 30 and 100 rows.
//
//   node scripts/pangram-drawn.mjs [--covers] [--lines] [--limit 3000] [--pangram <catalog.json>] [--out .data/pangram/drawn.json]     with `npm run demo` running
//
// --pangram reads with a candidate Pangram Pangram catalog (scripts/catalog-files.mjs --short writes one) in the shipped one's place.
import { readFile, writeFile } from 'node:fs/promises'
import { extname, basename } from 'node:path'
import { chromium } from 'playwright'
import { rankOf, twinSets } from './photos.mjs'

const covers = process.argv.includes('--covers'), at = process.argv.indexOf('--limit'), LIMIT = at < 0 ? 3000 : Number(process.argv[at + 1])
const OUT = process.argv.includes('--out') ? process.argv[process.argv.indexOf('--out') + 1] : '.data/pangram/drawn.json'
const PANGRAM = process.argv.includes('--pangram') ? process.argv[process.argv.indexOf('--pangram') + 1] : null
const root = '.data/pangram', inventory = JSON.parse(await readFile('bench/open-fonts.json', 'utf8'))
const designs = new Map(), fileOf = new Map()
for (const f of inventory.families) if (f.id.startsWith('pangram-pangram')) {
  if (f.design) designs.set(f.design, [...(designs.get(f.design) ?? []), f])
  for (const face of f.faces) fileOf.set(`${f.id}/${basename(face.path).replace(/\.[^.]+$/, '')}`, `.data/fonts-open/${face.path}`)
}
const fontPath = id => id.includes('@') ? `.data/fonts-open/instances/${id}.ttf` : fileOf.get(id)
const designOf = handle => [...designs.keys()].filter(d => handle === d || handle.startsWith(d + '-')).sort((a, b) => b.length - a.length)[0]
const prior = JSON.parse(await readFile(`${root}/specimens-read.json`, 'utf8')).reads
// --lines: the tallest Vision line with a quarter of its height around, as scripts/pangram-specimens.mjs crops it.
const lines = process.argv.includes('--lines'), boxes = new Map((await readFile(`${root}/specimens-boxes.jsonl`, 'utf8')).split('\n').filter(Boolean).map(JSON.parse).map(b => [b.image, b]))
const lineBox = image => { const b = boxes.get(image), top = b?.lines?.find(l => l.box[2] - l.box[0] >= 16 && l.box[3] - l.box[1] >= 8); if (!top) return null; const [x0, y0, x1, y1] = top.box, pad = Math.round((y1 - y0) * 0.25); return [Math.max(0, x0 - pad), Math.max(0, y0 - pad), Math.min(b.width, x1 + pad), Math.min(b.height, y1 + pad)] }
const items = prior.filter(r => r.kind === (lines ? 'line' : 'whole') && (!covers || r.image.endsWith('-00.jpg')))
  .map(r => ({ ...r, path: `${root}/specimens/${r.image}`, box: lines ? lineBox(r.image) : [0, 0, 1e9, 1e9], text: prior.find(p => p.image === r.image && p.kind === 'line')?.text ?? '' }))
console.log(items.length, 'pictures')

const TYPES = { '.jpg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' }
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', '--use-angle=metal'] })
const page = await browser.newPage()
await page.route('**/photos', route => route.fulfill({ contentType: 'text/html', body: '<title>Photos</title>' }))
await page.route('**/photo/*', async route => { const item = items[Number(route.request().url().split('/').pop())]; await route.fulfill({ contentType: TYPES[extname(item.path)], body: await readFile(item.path) }) })
if (PANGRAM) await page.route('**/catalogs/pangram-pangram.json', async route => route.fulfill({ contentType: 'application/json', body: await readFile(PANGRAM) }))
await page.route('**/localfont/**', async route => {
  const id = decodeURIComponent(new URL(route.request().url()).pathname.slice('/localfont/'.length)), path = fontPath(id)
  try { await route.fulfill({ contentType: 'font/ttf', body: await readFile(path) }) } catch { await route.fulfill({ status: 404, body: '' }) }
})
page.on('console', m => { if (m.type() === 'error') console.log('page:', m.text()) })
await page.goto('http://localhost:4179/photos')
await page.evaluate(async () => {
  const { createEncoder, createReader, rerank, catalogs } = await import('/src/match.mjs'), c = await import('/src/catalog.mjs'), { prepareLine } = await import('/src/line.mjs'), { drawLine } = await import('/src/references.mjs')
  const { readings, principal } = await import('/src/clean.mjs'), encoder = await createEncoder(), reader = await createReader()
  const catalog = c.unionCatalogs(await Promise.all(Object.values(catalogs).map(async url => c.readCatalog(await (await fetch(url)).json(), encoder.binding))))
  const loaded = new Map(), drawing = document.createElement('canvas').getContext('2d', { willReadFrequently: true })
  const load = face => {
    const local = face.id.startsWith('pangram-pangram'), key = local ? face.id : `${face.family}|${face.weight}|${face.style}`
    if (!loaded.has(key)) loaded.set(key, (async () => {
      if (local) {
        const alias = `local-${loaded.size}`, font = new FontFace(alias, `url(/localfont/${encodeURIComponent(face.id)})`, { weight: String(face.weight ?? 400), style: face.style ?? 'normal' })
        document.fonts.add(await font.load()); return alias
      }
      const axes = face.style === 'italic' ? `:ital,wght@1,${face.weight ?? 400}` : (face.weight ?? 400) === 400 ? '' : `:wght@${face.weight}`
      const link = Object.assign(document.createElement('link'), { rel: 'stylesheet', href: `https://fonts.googleapis.com/css2?family=${encodeURIComponent(face.family).replaceAll('%20', '+')}${axes}&display=block` })
      await new Promise((ok, no) => { link.onload = ok; link.onerror = () => no(new Error('no stylesheet')); document.head.append(link) })
      if (!(await document.fonts.load(`${face.style ?? 'normal'} ${face.weight ?? 400} 48px "${face.family}"`)).length) throw new Error('not served')
      return face.family
    })().catch(() => null))
    return loaded.get(key)
  }
  const draw = async (row, words, size) => { const family = await load(row.face); return family ? drawLine(drawing, family, words, size, row.face) : null }
  const plain = rows => rows.map(r => ({ family: r.family, name: r.face.family, siblings: r.siblings, score: r.score, face: r.face.id, ...(r.textScore !== undefined && { textScore: r.textScore }) }))
  window.photo = async (url, [x0, y0, x1, y1], limit, text) => {
    const bitmap = await createImageBitmap(await (await fetch(url)).blob())
    x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(bitmap.width, x1); y1 = Math.min(bitmap.height, y1)
    const scale = Math.min(1, Math.sqrt(16e6 / ((x1 - x0) * (y1 - y0)))), w = Math.max(1, Math.round((x1 - x0) * scale)), h = Math.max(1, Math.round((y1 - y0) * scale))
    const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d'); ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, w, h); bitmap.close()
    const taken = readings(ctx.getImageData(0, 0, w, h)), embeddings = [], kept = []
    for (const r of taken) {
      const prepared = prepareLine(r.image, { ...encoder.preparation, deskew: true, sampler: 'windows' })
      if (prepared.status === 'ok') { embeddings.push(c.embedWindows(await encoder.project(prepared.windows))); kept.push(r) }
    }
    if (!embeddings.length) return { rows: [] }
    const embedding = embeddings[principal(kept)], judged = encoder.heads ? c.verdict(embedding, encoder.heads) : null
    const rows = c.foldTwins(c.bestOf(embeddings.map(e => c.matchCatalog(e, catalog, judged))), catalog, { script: judged?.script?.[0]?.label, limit })
    const read = reader.read(taken[0].crop)?.text ?? '', options = { embedding, encoder, draw }
    const pangram = rows.filter(r => r.family.startsWith('pangram-pangram'))
    return {
      read, embedding: Array.from(c.unit(embedding)), rows: plain(rows.slice(0, 100)), pangram: plain(pangram.slice(0, 100)), all: plain(rows).map(r => ({ family: r.family, name: r.name, siblings: r.siblings })),
      byText30: plain((await rerank(rows, text, { ...options, limit: 30 })).slice(0, 30)),
      byRead30: plain((await rerank(rows, read, { ...options, limit: 30 })).slice(0, 30)),
      byText100: plain((await rerank(rows, text, { ...options, limit: 100 })).slice(0, 100)),
    }
  }
})
const twins = await twinSets(), out = []
for (const [i, item] of items.entries()) {
  const families = designs.get(designOf(item.handle)), r = await page.evaluate(([i, box, limit, text]) => window.photo(`/photo/${i}`, box, limit, text), [i, item.box, LIMIT, item.text])
  const rank = rows => { const ks = families.map(f => rankOf(rows ?? [], f.family, twins.get(f.id))).filter(k => k !== null); return ks.length ? Math.min(...ks) : null }
  const truthFace = r.rows.concat(r.byText100 ?? []).find(x => families.some(f => f.id === x.family))
  out.push({ image: item.image, design: designOf(item.handle), text: item.text, read: r.read, embedding: r.embedding, prior: item.rank, rank: rank(r.all), pangramRank: rank(r.pangram), byText30: rank(r.byText30), byRead30: rank(r.byRead30), byText100: rank(r.byText100),
    first: r.rows.slice(0, 5).map(x => x.name), pangramFirst: r.pangram.slice(0, 5).map(x => `${x.name} (${x.face.split('@')[1] ?? ''})`), byTextFirst: r.byText30.slice(0, 5).map(x => `${x.name} ${x.textScore?.toFixed(3) ?? '-'}`), truthFace: truthFace?.face, truthTextScore: r.byText100?.find(x => families.some(f => f.id === x.family))?.textScore ?? null })
  if ((i + 1) % 25 === 0) { console.log(i + 1); await writeFile(OUT, JSON.stringify(out, null, 1)) }
}
await writeFile(OUT, JSON.stringify(out, null, 1)); await browser.close()
