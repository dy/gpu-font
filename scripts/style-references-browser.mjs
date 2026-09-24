// Catalog reference lines rendered by Chromium canvas (the rasterizer of most screenshots), prepared by the browser code.
import { readFile, writeFile, open } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { prepareLine } from '../src/line.mjs'
import { SIZES } from '../src/references.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const read = async path => JSON.parse(await readFile(path))
const references = await read('.data/style/references.json'), faces = (await read('.data/style/faces.json')).faces
const instances = await read('.data/style/reference-instances.json')
const byFace = new Map()
references.samples.forEach((s, i) => { if (!byFace.has(s.face)) byFace.set(s.face, []); byFace.get(s.face).push(i) })
const windows = [], file = await open('.data/style/references-browser.u8', 'w'), browser = await chromium.launch()
const digest = createHash('sha256')  // of the window file as written: at three sizes it is too large to read back whole
let offset = 0, done = 0
const prepared = new Array(references.samples.length)
try {
  const page = await browser.newPage()
  await page.route('**/style-references', route => route.fulfill({ contentType: 'text/html', body: '<title>References</title>' }))
  await page.goto('http://localhost:4179/style-references')
  for (const [face, indexes] of byFace) {
    const { path, sha256 } = instances[faces[face].id], bytes = await readFile(path)
    if (hash(bytes) !== sha256) throw new Error(`Changed instance: ${faces[face].id}`)
    // Each line at each size, as the page draws My fonts (src/references.mjs drawLine).
    const images = await page.evaluate(async ({ base64, lines, sizes }) => {
      const font = new FontFace('reference', Uint8Array.from(atob(base64), c => c.charCodeAt(0)))
      await font.load(); document.fonts.add(font)
      try {
        return lines.map(text => sizes.map(size => {
          const c = document.createElement('canvas'), ctx = c.getContext('2d'); ctx.font = `${size}px reference`
          const b = ctx.measureText(text), pad = 12
          c.width = Math.max(1, Math.ceil(b.actualBoundingBoxLeft + b.actualBoundingBoxRight) + pad * 2)
          c.height = Math.max(1, Math.ceil(b.actualBoundingBoxAscent + b.actualBoundingBoxDescent) + pad * 2)
          ctx.fillStyle = 'white'; ctx.fillRect(0, 0, c.width, c.height); ctx.fillStyle = 'black'; ctx.font = `${size}px reference`
          ctx.fillText(text, pad + b.actualBoundingBoxLeft, pad + b.actualBoundingBoxAscent)
          const rgba = ctx.getImageData(0, 0, c.width, c.height).data; let raw = ''
          for (let i = 0; i < rgba.length; i += 4) raw += String.fromCharCode(rgba[i])
          return { width: c.width, height: c.height, pixels: btoa(raw) }
        }))
      } finally { document.fonts.delete(font) }
    }, { base64: bytes.toString('base64'), lines: indexes.map(i => references.samples[i].text), sizes: SIZES })
    for (const [n, i] of indexes.entries()) prepared[i] = SIZES.map((size, k) => {
      const { width, height } = images[n][k], gray = Buffer.from(images[n][k].pixels, 'base64'), data = new Uint8Array(width * height * 4)
      for (let p = 0; p < gray.length; p++) { data.fill(gray[p], p * 4, p * 4 + 3); data[p * 4 + 3] = 255 }
      return { size, windows: prepareLine({ width, height, data }, { deskew: true, sampler: 'windows' }).windows }
    })
    if (++done % 500 === 0) console.log('references', done, '/', byFace.size)
  }
  // Keep the Pillow manifest's sample order, one sample per size; windows follow it.
  const samples = []
  for (const [i, sized] of prepared.entries()) for (const { size, windows: items } of sized ?? []) {
    if (!items.length) continue
    const source = samples.length; samples.push({ ...references.samples[i], size })
    for (const w of items) {
      const raw = Buffer.from(Uint8Array.from(w.pixels, v => Math.round(v * 255)))
      windows.push({ source, offset, width: w.width, height: w.height }); await file.write(raw); digest.update(raw); offset += raw.length
    }
  }
  await file.close()
  const pins = Object.fromEntries(await Promise.all(['scripts/style-references-browser.mjs', 'src/references.mjs', 'src/line.mjs', 'src/input.mjs', 'src/prepare.mjs'].map(async p => [p, hash(await readFile(p))])))
  await writeFile('.data/style/references-browser.json', JSON.stringify({ facesSha256: references.facesSha256, pins, browser: browser.version(),
    sha256: digest.digest('hex'), samples, windows }) + '\n')
  console.log(`Chromium references ${samples.length} lines ${windows.length} windows`)
} finally { await browser.close() }
