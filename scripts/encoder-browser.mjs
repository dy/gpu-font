// Render the frozen encoder experiment plan. Final families are prepared after selection.
import { readFile, writeFile, mkdir, rename } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { prepareBatch } from './corpus-prepare.mjs'

const phase = process.argv[2] || 'development'
if (!['development', 'final'].includes(phase)) throw new Error('Invalid encoder phase')
const dir = `.data/encoder/${phase}/chromium`, plan = JSON.parse(await readFile(`.data/encoder/${phase}-plan.json`))
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
for (const [path, expected] of Object.entries(plan.pins)) if (hash(await readFile(path)) !== expected) throw new Error(`Changed plan dependency: ${path}`)
await mkdir(dir, { recursive: true })
const browser = await chromium.launch()
let cursor = 0, done = 0
try {
  await Promise.all(Array.from({ length: 4 }, async () => {
    while (cursor < plan.jobs.length) {
      const job = plan.jobs[cursor++], family = job.family, prefix = `${dir}/${family.id}`
      let old
      try { old = JSON.parse(await readFile(`${prefix}.json`)) } catch (error) { if (error.code !== 'ENOENT') throw error }
      if (old && JSON.stringify(old.pins) === JSON.stringify(plan.pins) && hash(await readFile(`${prefix}.u8`)) === old.sha256) { done++; continue }
      const bytes = await readFile(`.data/encoder/faces/${family.id}.ttf`), face = family.faces.find(f => f.path === family.selected)
      const instance = JSON.parse(await readFile(`.data/encoder/faces/${family.id}.json`))
      if (instance.source.blob !== face.blob || JSON.stringify(instance.source.axes) !== JSON.stringify(family.trainingAxes) || hash(bytes) !== instance.sha256) throw new Error('Changed normal face instance')
      const page = await browser.newPage()
      try {
        await page.route('**/encoder-render', route => route.fulfill({ contentType: 'text/html', body: '<title>Encoder specimens</title>' }))
        await page.route('**/encoder-font.ttf', route => route.fulfill({ contentType: 'font/ttf', body: bytes }))
        await page.goto('http://localhost:4179/encoder-render')
        // A static instance pins width, optical size and every other axis identically
        // to Pillow; browser defaults cannot silently change the reference face.
        await page.evaluate(async () => {
          const font = new FontFace('specimen', 'url(/encoder-font.ttf)')
          await font.load(); document.fonts.add(font)
        })
        const images = await page.evaluate(plans => plans.map(plan => {
          const font = `${plan.size}px specimen`
          const canvas = document.createElement('canvas'); let c = canvas.getContext('2d'); c.font = font
          const bounds = c.measureText(plan.text)
          canvas.width = Math.ceil(bounds.actualBoundingBoxLeft + bounds.actualBoundingBoxRight) + 12
          canvas.height = Math.ceil(bounds.actualBoundingBoxAscent + bounds.actualBoundingBoxDescent) + 12
          c = canvas.getContext('2d'); c.font = font; c.fillStyle = 'white'; c.fillRect(0, 0, canvas.width, canvas.height); c.fillStyle = 'black'
          c.fillText(plan.text, 6 + bounds.actualBoundingBoxLeft, 6 + bounds.actualBoundingBoxAscent)
          const condition = plan.condition === 'augmented' ? ['rotate', 'small', 'blur', 'edge'][Math.floor(plan.index / 2) % 4] : plan.condition
          const angle = condition === 'rotate' ? (plan.index % 2 ? -1 : 1) * Math.PI / 30 : 0, scale = condition === 'small' ? .45 : 1
          const out = document.createElement('canvas')
          out.width = Math.max(1, Math.ceil(scale * (canvas.width * Math.cos(angle) + canvas.height * Math.abs(Math.sin(angle)))))
          out.height = Math.max(1, Math.ceil(scale * (canvas.height * Math.cos(angle) + canvas.width * Math.abs(Math.sin(angle)))))
          const ctx = out.getContext('2d'); ctx.fillStyle = 'white'; ctx.fillRect(0, 0, out.width, out.height)
          ctx.translate(out.width / 2, out.height / 2); ctx.rotate(angle); ctx.scale(scale, scale)
          if (condition === 'blur') ctx.filter = 'blur(0.5px)'
          ctx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2)
          const width = condition === 'edge' ? Math.max(1, Math.round(out.width * .85)) : out.width
          const rgba = ctx.getImageData(0, 0, width, out.height).data; let pixels = ''
          for (let i = 0; i < rgba.length; i += 4) pixels += String.fromCharCode(plan.light ? 255 - rgba[i] : rgba[i])
          return { width, height: out.height, pixels: btoa(pixels) }
        }), job.plans)
        const prepared = prepareBatch(images), windows = [], chunks = []; let offset = 0
        for (const [source, items] of prepared.entries()) {
          if (!items.length) throw new Error(`Blank browser specimen: ${family.id}`)
          for (const w of items) {
            const raw = Buffer.from(w.pixels, 'base64'); windows.push({ source, offset, width: w.width, height: w.height }); chunks.push(raw); offset += raw.length
          }
        }
        const packed = Buffer.concat(chunks)
        await writeFile(`${prefix}.u8.part`, packed); await rename(`${prefix}.u8.part`, `${prefix}.u8`)
        await writeFile(`${prefix}.json.part`, JSON.stringify({ pins: plan.pins, sha256: hash(packed), samples: job.plans, windows, browser: browser.version() }))
        await rename(`${prefix}.json.part`, `${prefix}.json`)
      } finally { await page.close() }
      if (++done % 100 === 0 || done === plan.jobs.length) console.log(`Chromium ${phase} ${done}/${plan.jobs.length}`)
    }
  }))
} finally { await browser.close() }
