// gpu-font on photographs and forum pictures, as the page runs them: every shipped catalog searched as one (the page's
// All), the crop at full resolution, rows folded as the page shows them. Needs `npm run demo` and a WebGPU adapter.
// The dataset scripts (whatfontis.mjs, dafont.mjs) supply the images; this module only matches and ranks.
import { readFile } from 'node:fs/promises'
import { extname } from 'node:path'
import { chromium } from 'playwright'

export const normalize = name => name.toLowerCase().replace(/[^a-z0-9]/g, '')

// 1-based row whose family, or a family folded into it, has the true name, or whose family id is one of `twins`, the
// families the teacher finds indistinguishable from the true one in any script, across catalogs; null when none of `rows` is.
export function rankOf(rows, truth, twins = new Set()) {
  const want = normalize(truth), i = rows.findIndex(r => [r.name, ...r.siblings].some(n => normalize(n) === want) || twins.has(r.family))
  return i < 0 ? null : i + 1
}

// The twins of each family the teacher spans, every script together: {familyId: Set of family ids}, from
// .data/style/twins.json (train.style twins), the same file the catalogs carry per face. Read from the file, the credit
// applies to any model's read alike, whether or not its catalogs were rebuilt; without the file no twin is credited.
export async function twinSets() {
  const families = await readFile('.data/style/twins.json', 'utf8').then(text => JSON.parse(text).families, () => { console.log('No .data/style/twins.json: no twin credit; run node scripts/python.mjs -m train.style twins'); return {} })
  return new Map(Object.entries(families).map(([id, scripts]) => [id, new Set(Object.values(scripts).flat())]))
}

// Accuracy of scored items ({ truth: { family }, rank }) over the items, and over families with each counted once, so a
// family with thousands of pictures does not speak for the rest. Shares are null when nothing was scored.
export function accuracy(list) {
  const share = test => list.length ? Math.round(1e4 * list.filter(test).length / list.length) / 1e4 : null
  const families = [...list.reduce((m, s) => m.set(s.truth.family, [...(m.get(s.truth.family) || []), s]), new Map()).values()]
  const mean = test => families.length ? Math.round(1e4 * families.reduce((sum, f) => sum + f.filter(test).length / f.length, 0) / families.length) / 1e4 : null
  const first = s => s.rank === 1, five = s => s.rank !== null && s.rank <= 5
  return { families: families.length, top1: share(first), top5: share(five), top20: share(s => s.rank !== null), byFamily: { top1: mean(first), top5: mean(five) } }
}

// Family names of every shipped catalog, normalized, with the catalog each belongs to.
// A family name may sit in several catalogs (Lato in Google Fonts and Adobe Fonts): `catalog` is the first that holds it in the
// shipped order, Google Fonts first, so its training role applies; `catalogs` lists them all.
export async function catalogNames() {
  const { catalogs } = await import('../src/match.mjs'), names = new Map()
  for (const [id, url] of Object.entries(catalogs))
    for (const face of JSON.parse(await readFile(url)).faces) {
      const key = normalize(face.family), known = names.get(key)
      if (known) { if (!known.catalogs.includes(id)) known.catalogs.push(id) } else names.set(key, { catalog: id, catalogs: [id], familyId: face.familyId, family: face.family })
    }
  return names
}

// `--clean <name>` on a read's command line: the cleaner to pass through, and where its report goes instead of bench/, so a
// candidate is measured without replacing the shipped read.
export function cleanFlag(name) {
  const at = process.argv.indexOf('--clean'), clean = at < 0 ? null : process.argv[at + 1]
  return { clean, report: clean ? `.data/style/${name}-${clean.replace(/\W+/g, '-')}.json` : `bench/${name}.json` }
}

const TYPES = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' }

// items: [{ path, box: [x0, y0, x1, y1] }] in image pixels, end exclusive. Returns, per item, { status, ms, rows }: up to
// `limit` rows best first, each { family, name, siblings, score, weight, style } (the matched face). `ms` runs from the
// image bytes to the ranked rows; `encoderSha256` names the model that read them.
// With `text`, each item may carry the text in its picture, the shipped line reader reads the crop, and two more rankings
// come back: `byText`, the shortlist set in the item's text (src/sametext.mjs), and `byRead`, set in the text read; `read`
// is the reader's text. Faces draw from Google Fonts, so only Google families take part; the rest keep their catalog score.
// Each crop is read as the page reads it (src/clean.mjs readings: each line as it stands and cleaned, each family at its
// better); `clean` names a candidate of scripts/clean-candidates.mjs to read each line through instead, or 'none' for each
// line as it stands alone.
export async function matchPhotos(items, { limit = 20, log = console.log, text = false, clean = null } = {}) {
  const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
  try {
    const page = await browser.newPage()
    await page.route('**/photos', route => route.fulfill({ contentType: 'text/html', body: '<title>Photos</title>' }))
    await page.route('**/photo/*', async route => {
      const item = items[Number(new URL(route.request().url()).pathname.split('/').pop())]
      await route.fulfill({ contentType: TYPES[extname(item.path).toLowerCase()] || 'application/octet-stream', body: await readFile(item.path) })
    })
    await page.goto('http://localhost:4179/photos')
    const ready = () => page.evaluate(async ([withText, cleanName]) => {
      const adapter = await navigator.gpu?.requestAdapter()
      if (!adapter) return {}
      const { createEncoder, createReader, rerank, SHORTLIST, catalogs } = await import('/src/match.mjs'), c = await import('/src/catalog.mjs'), { prepareLine } = await import('/src/line.mjs'), { drawLine } = await import('/src/references.mjs')
      const { readings, clean, turn, principal } = await import('/src/clean.mjs'), encoder = await createEncoder(), reader = withText ? await createReader() : null, cleaner = cleanName === 'none' ? image => image : cleanName ? (await import('/scripts/clean-candidates.mjs')).CLEANERS[cleanName] : clean
      const catalog = c.unionCatalogs(await Promise.all(Object.values(catalogs).map(async url => c.readCatalog(await (await fetch(url)).json(), encoder.binding))))
      // A Google face at its weight and posture, loaded once; null when Google Fonts does not serve it.
      const loaded = new Map(), drawing = document.createElement('canvas').getContext('2d', { willReadFrequently: true })
      const googleFace = face => {
        const key = `${face.family}|${face.weight}|${face.style}`
        if (!loaded.has(key)) loaded.set(key, (async () => {
          const axes = face.style === 'italic' ? `:ital,wght@1,${face.weight ?? 400}` : (face.weight ?? 400) === 400 ? '' : `:wght@${face.weight}`
          const link = Object.assign(document.createElement('link'), { rel: 'stylesheet', href: `https://fonts.googleapis.com/css2?family=${encodeURIComponent(face.family).replaceAll('%20', '+')}${axes}&display=block` })
          await new Promise((ok, no) => { link.onload = ok; link.onerror = () => no(new Error('no stylesheet')); document.head.append(link) })
          if (!(await document.fonts.load(`${face.style ?? 'normal'} ${face.weight ?? 400} 48px "${face.family}"`)).length) throw new Error('not served')
          return face.family
        })().catch(() => null))
        return loaded.get(key)
      }
      // A face its host serves (`file`, an instance at its `axes`), loaded once; the page draws it the same way (app.mjs loadFont).
      const hostedFace = face => {
        if (!loaded.has(face.id)) loaded.set(face.id, (async () => {
          const alias = `hosted-${loaded.size}`, font = new FontFace(alias, `url("${face.file}")`, { weight: String(face.weight ?? 400), style: face.style ?? 'normal', ...(face.axes ? { variationSettings: Object.entries(face.axes).map(([tag, value]) => `"${tag}" ${value}`).join(', ') } : {}) })
          document.fonts.add(await font.load()); return alias
        })().catch(() => null))
        return loaded.get(face.id)
      }
      const draw = async (row, words, size) => { const family = await (row.face.file ? hostedFace(row.face) : googleFace(row.face)); return family ? drawLine(drawing, family, words, size, row.face) : null }  // any row Google Fonts serves by that name, or whose host serves its file
      const plain = rows => rows.map(r => ({ family: r.family, name: r.face.family, siblings: r.siblings, score: r.score, weight: r.face.weight, style: r.face.style, ...(r.textScore !== undefined && { textScore: r.textScore, combined: r.combined }) }))
      window.photo = async (url, [x0, y0, x1, y1], limit, text) => {
        const start = performance.now(), bitmap = await createImageBitmap(await (await fetch(url)).blob())
        x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(bitmap.width, x1); y1 = Math.min(bitmap.height, y1)
        // A crop past the preparation's 16 M pixels is drawn smaller, as the page's resolution control would.
        const scale = Math.min(1, Math.sqrt(16e6 / ((x1 - x0) * (y1 - y0)))), w = Math.max(1, Math.round((x1 - x0) * scale)), h = Math.max(1, Math.round((y1 - y0) * scale))
        const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d')
        ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, w, h); bitmap.close()
        // As the page reads it: by the crop's readings (src/clean.mjs), each family at its best; a candidate cleaner, or
        // 'none', stands in for the page's two readings of a line with one.
        const raw = ctx.getImageData(0, 0, w, h), embeddings = [], kept = [], taken = cleaner === clean ? readings(raw) : readings(raw).filter(r => r.kind !== 'cleaned').map(r => ({ ...r, image: r.angle ? turn(cleaner(r.crop), r.angle) : cleaner(r.crop) }))
        let status = 'blank'
        for (const r of taken) {
          const prepared = prepareLine(r.image, { ...encoder.preparation, deskew: true, sampler: 'windows' })
          if (prepared.status === 'ok') { embeddings.push(c.embedWindows(await encoder.project(prepared.windows))); kept.push(r) }
          if (status !== 'ok') status = prepared.status
        }
        if (!embeddings.length) return { status, ms: performance.now() - start, rows: [] }
        const embedding = embeddings[principal(kept)], judged = encoder.heads ? c.verdict(embedding, encoder.heads) : null
        const rows = c.foldTwins(c.bestOf(embeddings.map(e => c.matchCatalog(e, catalog, judged))), catalog, { script: judged?.script?.[0]?.label, limit: reader ? Math.max(limit, SHORTLIST) : limit })
        const result = { status, ms: performance.now() - start, rows: plain(rows).slice(0, limit) }
        if (reader) {
          const started = performance.now(), read = reader.read(taken[0].crop)?.text ?? ''  /* the reader reads the first line as it is, as on the page */; result.read = read; result.readMs = performance.now() - started
          const options = { embedding, encoder, draw }
          result.byText = plain(await rerank(rows, text ?? '', options)).slice(0, limit); result.byRead = plain(await rerank(rows, read, options)).slice(0, limit)
          result.textMs = performance.now() - started - result.readMs
        }
        return result
      }
      const info = adapter.info || {}
      return { adapter: [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'WebGPU', encoderSha256: encoder.binding.encoderSha256 }
    }, [text, clean])
    const { adapter, encoderSha256 } = await ready()
    // On the CPU a match takes about ten seconds: a whole set would take hours, so stop instead.
    if (!adapter) throw new Error('No WebGPU adapter in Chromium')
    await page.evaluate(() => window.photo('/photo/0', [0, 0, 1e9, 1e9], 1))  // warm-up: kernels compiled, catalogs decoded
    // One picture can hang the page (a bitmap that never decodes, a renderer gone): a match past the patience counts as unread,
    // and the page is opened anew so the next picture starts clean.
    const results = [], PATIENCE = 60000
    const timed = promise => Promise.race([promise, new Promise(resolve => setTimeout(() => resolve({ status: 'timeout', ms: PATIENCE, rows: [] }), PATIENCE))])
    for (const [i, item] of items.entries()) {
      // A picture the page cannot take (one that does not decode, a crop with no pixels) is unread, with its reason, and the read goes on.
      const result = await timed(page.evaluate(([i, box, limit, text]) => window.photo(`/photo/${i}`, box, limit, text), [i, item.box, limit, item.text]).catch(error => ({ status: 'error', error: String(error.message ?? error).split('\n')[0], ms: 0, rows: [] })))
      if (result.status === 'error') log(`${item.path}: ${result.error}`)
      if (result.status === 'timeout') { log(`${item.path}: no answer in ${PATIENCE / 1000} s; the page is reopened`); await page.reload({ waitUntil: 'load' }).catch(() => {}); await ready() }
      results.push(result)
      if ((i + 1) % 250 === 0) log(`${i + 1} / ${items.length}`)
    }
    return { browser: browser.version(), adapter, encoderSha256, results }
  } finally { await browser.close() }
}
