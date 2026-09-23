// Chromium canvas renders of the frozen style benchmark: static instances, black on white, raw grayscale.
import { readFile, writeFile, open } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const planBytes = await readFile('.data/style/bench-plan.json'), plan = JSON.parse(planBytes)
const instances = JSON.parse(await readFile('.data/style/bench-instances.json'))
const byFace = new Map()
plan.queries.forEach((q, i) => { if (!byFace.has(q.faceId)) byFace.set(q.faceId, []); byFace.get(q.faceId).push(i) })
const images = new Array(plan.queries.length), file = await open('.data/style/bench-browser.u8', 'w')
const browser = await chromium.launch()
let offset = 0
try {
  const page = await browser.newPage()
  await page.route('**/style-bench', route => route.fulfill({ contentType: 'text/html', body: '<title>Style benchmark</title>' }))
  await page.goto('http://localhost:4179/style-bench')
  for (const [face, indexes] of byFace) {
    const bytes = await readFile(instances[face].path)
    if (hash(bytes) !== instances[face].sha256) throw new Error(`Changed instance: ${face}`)
    const plans = indexes.map(i => ({ text: plan.queries[i].text, size: plan.queries[i].size }))
    const rendered = await page.evaluate(async ({ base64, plans }) => {
      const font = new FontFace('bench', Uint8Array.from(atob(base64), c => c.charCodeAt(0)))
      await font.load(); document.fonts.add(font)
      try {
        return plans.map(({ text, size }) => {
          const c = document.createElement('canvas'), ctx = c.getContext('2d'); ctx.font = `${size}px bench`
          const b = ctx.measureText(text), pad = 28
          c.width = Math.ceil(b.actualBoundingBoxLeft + b.actualBoundingBoxRight) + pad * 2
          c.height = Math.ceil(b.actualBoundingBoxAscent + b.actualBoundingBoxDescent) + pad * 2
          ctx.fillStyle = 'white'; ctx.fillRect(0, 0, c.width, c.height); ctx.fillStyle = 'black'; ctx.font = `${size}px bench`
          ctx.fillText(text, pad + b.actualBoundingBoxLeft, pad + b.actualBoundingBoxAscent)
          const rgba = ctx.getImageData(0, 0, c.width, c.height).data; let raw = ''
          for (let i = 0; i < rgba.length; i += 4) raw += String.fromCharCode(rgba[i])
          return { width: c.width, height: c.height, pixels: btoa(raw) }
        })
      } finally { document.fonts.delete(font) }
    }, { base64: bytes.toString('base64'), plans })
    for (const [n, i] of indexes.entries()) {
      const raw = Buffer.from(rendered[n].pixels, 'base64')
      images[i] = { width: rendered[n].width, height: rendered[n].height, offset }; await file.write(raw); offset += raw.length
    }
  }
} finally { await browser.close(); await file.close() }
await writeFile('.data/style/bench-browser.json', JSON.stringify({ planSha256: hash(planBytes), browser: browser.version(), sha256: hash(await readFile('.data/style/bench-browser.u8')), images }) + '\n')
console.log(`Rendered ${images.length} benchmark queries in Chromium ${browser.version()}`)
