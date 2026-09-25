// gpu-font on photographs and forum pictures, as the page runs them: every shipped catalog searched as one (the page's
// All), the crop at full resolution, rows folded as the page shows them. Needs `npm run demo` and a WebGPU adapter.
// The dataset scripts (whatfontis.mjs, dafont.mjs) supply the images; this module only matches and ranks.
import { readFile } from 'node:fs/promises'
import { extname } from 'node:path'
import { chromium } from 'playwright'

export const normalize = name => name.toLowerCase().replace(/[^a-z0-9]/g, '')

// 1-based row whose family, or a family folded into it, has the true name; null when none of `rows` does.
export function rankOf(rows, truth) {
  const want = normalize(truth), i = rows.findIndex(r => [r.name, ...r.siblings].some(n => normalize(n) === want))
  return i < 0 ? null : i + 1
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

const TYPES = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' }

// items: [{ path, box: [x0, y0, x1, y1] }] in image pixels, end exclusive. Returns, per item, { status, ms, rows }: up to
// `limit` rows best first, each { family, name, siblings, score, weight, style } (the matched face). `ms` runs from the
// image bytes to the ranked rows; `encoderSha256` names the model that read them.
export async function matchPhotos(items, { limit = 20, log = console.log } = {}) {
  const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
  try {
    const page = await browser.newPage()
    await page.route('**/photos', route => route.fulfill({ contentType: 'text/html', body: '<title>Photos</title>' }))
    await page.route('**/photo/*', async route => {
      const item = items[Number(new URL(route.request().url()).pathname.split('/').pop())]
      await route.fulfill({ contentType: TYPES[extname(item.path).toLowerCase()] || 'application/octet-stream', body: await readFile(item.path) })
    })
    await page.goto('http://localhost:4179/photos')
    const { adapter, encoderSha256 } = await page.evaluate(async () => {
      const adapter = await navigator.gpu?.requestAdapter()
      if (!adapter) return {}
      const { createEncoder, catalogs } = await import('/src/match.mjs'), c = await import('/src/catalog.mjs'), { prepareLine } = await import('/src/line.mjs')
      const encoder = await createEncoder()
      const catalog = c.unionCatalogs(await Promise.all(Object.values(catalogs).map(async url => c.readCatalog(await (await fetch(url)).json(), encoder.binding))))
      window.photo = async (url, [x0, y0, x1, y1], limit) => {
        const start = performance.now(), bitmap = await createImageBitmap(await (await fetch(url)).blob())
        x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(bitmap.width, x1); y1 = Math.min(bitmap.height, y1)
        const canvas = new OffscreenCanvas(x1 - x0, y1 - y0), ctx = canvas.getContext('2d')
        ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0); bitmap.close()
        const { status, windows } = prepareLine(ctx.getImageData(0, 0, x1 - x0, y1 - y0), { ...encoder.preparation, deskew: true, sampler: 'windows' })
        if (status !== 'ok') return { status, ms: performance.now() - start, rows: [] }
        const embedding = c.embedWindows(await encoder.project(windows)), judged = encoder.heads ? c.verdict(embedding, encoder.heads) : null
        const rows = c.foldTwins(c.matchCatalog(embedding, catalog, judged), catalog, { script: judged?.script?.[0]?.label, limit })
        return { status, ms: performance.now() - start, rows: rows.map(r => ({ family: r.family, name: r.face.family, siblings: r.siblings, score: r.score, weight: r.face.weight, style: r.face.style })) }
      }
      const info = adapter.info || {}
      return { adapter: [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'WebGPU', encoderSha256: encoder.binding.encoderSha256 }
    })
    // On the CPU a match takes about ten seconds: a whole set would take hours, so stop instead.
    if (!adapter) throw new Error('No WebGPU adapter in Chromium')
    await page.evaluate(() => window.photo('/photo/0', [0, 0, 1e9, 1e9], 1))  // warm-up: kernels compiled, catalogs decoded
    const results = []
    for (const [i, item] of items.entries()) {
      results.push(await page.evaluate(([i, box, limit]) => window.photo(`/photo/${i}`, box, limit), [i, item.box, limit]))
      if ((i + 1) % 250 === 0) log(`${i + 1} / ${items.length}`)
    }
    return { browser: browser.version(), adapter, encoderSha256, results }
  } finally { await browser.close() }
}
