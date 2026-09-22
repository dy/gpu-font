// Fresh demo-sized queries, independent of the training/reference text banks.
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { prepareBatch } from './corpus-prepare.mjs'

const out = '.data/detection-quality/phrases', hash = bytes => createHash('sha256').update(bytes).digest('hex')
const read = async path => JSON.parse(await readFile(path))
const fonts = (await read('bench/fonts-100.json')).fonts, inventory = await read('bench/corpus.json'), split = await read('bench/encoder-split.json')
const texts = ['Amber clouds drift', 'Hidden paths unfold', 'Crisp winter light', 'Baked figs and honey', 'aRneGQop', 'ri'], sizes = [32, 56]
const pins = Object.fromEntries(await Promise.all(['scripts/encoder-quality.mjs', 'bench/fonts-100.json', 'bench/corpus.json', 'src/line.mjs', 'src/input.mjs', 'src/prepare.mjs'].map(async p => [p, hash(await readFile(p))])))
let old
try { old = await read(`${out}.json`) } catch (error) { if (error.code !== 'ENOENT') throw error }
if (old) {
  if (JSON.stringify(old.pins) !== JSON.stringify(pins) || old.sha256 !== hash(await readFile(`${out}.u8`))) throw new Error('Changed phrase benchmark')
  console.log(`Retained ${old.samples.length} frozen phrase queries`)
} else {
  const samples = [], windows = [], chunks = [], browser = await chromium.launch(); let offset = 0
  try {
    const page = await browser.newPage()
    await page.route('**/phrase-render', route => route.fulfill({ contentType: 'text/html', body: '<title>Font queries</title>' }))
    await page.goto('http://localhost:4179/phrase-render')
    for (const font of fonts) {
      const candidates = inventory.families.filter(f => !f.excluded && (f.family === font.family || f.selected.split('/').at(-1) === font.file))
      if (candidates.length !== 1) throw new Error(`Ambiguous or missing candidate: ${font.family}`)
      const family = candidates[0]
      const bytes = await readFile(`.data/fonts/${font.id}.ttf`)
      if (hash(bytes) !== font.instanceSha256) throw new Error('Changed query font')
      const plans = texts.flatMap(text => sizes.map(size => ({ text, size })))
      const images = await page.evaluate(async ({ base64, plans }) => {
        const font = new FontFace('query', Uint8Array.from(atob(base64), c => c.charCodeAt(0)))
        await font.load(); document.fonts.add(font)
        try {
          return plans.map(({ text, size }) => {
            const c = document.createElement('canvas'), ctx = c.getContext('2d'); ctx.font = `${size}px query`
            const b = ctx.measureText(text), pad = 28
            c.width = Math.ceil(b.actualBoundingBoxLeft + b.actualBoundingBoxRight) + pad * 2
            c.height = Math.ceil(b.actualBoundingBoxAscent + b.actualBoundingBoxDescent) + pad * 2
            ctx.fillStyle = 'white'; ctx.fillRect(0, 0, c.width, c.height); ctx.fillStyle = 'black'; ctx.font = `${size}px query`
            ctx.fillText(text, pad + b.actualBoundingBoxLeft, pad + b.actualBoundingBoxAscent)
            const rgba = ctx.getImageData(0, 0, c.width, c.height).data; let raw = ''
            for (let i = 0; i < rgba.length; i += 4) raw += String.fromCharCode(rgba[i])
            return { width: c.width, height: c.height, pixels: btoa(raw) }
          })
        } finally { document.fonts.delete(font) }
      }, { base64: bytes.toString('base64'), plans })
      for (const [i, items] of prepareBatch(images).entries()) {
        if (!items.length) throw new Error(`Empty query: ${font.id}`)
        const source = samples.length, plan = plans[i]
        samples.push({ ...plan, family: family.id, split: split.families[family.id], renderer: 'chromium', script: 'Latn', length: plan.text.length > 7 ? '8+' : '2-3', condition: 'clean', role: 'confirmation', fontSha256: font.instanceSha256 })
        for (const w of items) {
          const raw = Buffer.from(w.pixels, 'base64')
          windows.push({ source, offset, width: w.width, height: w.height }); chunks.push(raw); offset += raw.length
        }
      }
    }
    const pixels = Buffer.concat(chunks)
    await mkdir('.data/detection-quality', { recursive: true }); await writeFile(`${out}.u8`, pixels)
    await writeFile(`${out}.json`, JSON.stringify({ pins, sha256: hash(pixels), samples, windows, browser: browser.version(), scope: 'Fresh phrases frozen before refinement results; synthetic clean demo rendering, 100 families, two sizes. Not independent real-world screenshots.' }) + '\n')
    console.log(`Frozen ${samples.length} phrase queries / ${windows.length} windows`)
  } finally { await browser.close() }
}
