// Additional training-only text distributions; share the exact existing normalizer.
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { prepareBatch } from './corpus-prepare.mjs'

const dir = '.data/encoder/case-training', plan = JSON.parse(await readFile(`${dir}/plan.json`))
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
for (const [path, expected] of Object.entries(plan.pins)) if (hash(await readFile(path)) !== expected) throw new Error('Changed case-training plan')
await mkdir(`${dir}/chromium`, { recursive: true })
const browser = await chromium.launch(); let cursor = 0, done = 0
try {
  await Promise.all(Array.from({ length: 4 }, async () => {
    const page = await browser.newPage()
    await page.route('**/case-render', route => route.fulfill({ contentType: 'text/html', body: '<title>Training specimens</title>' }))
    await page.goto('http://localhost:4179/case-render')
    try {
      while (cursor < plan.jobs.length) {
        const job = plan.jobs[cursor++], prefix = `${dir}/chromium/${job.family.id}`
        const bytes = await readFile(`.data/encoder/faces/${job.family.id}.ttf`), info = JSON.parse(await readFile(`.data/encoder/faces/${job.family.id}.json`))
        if (hash(bytes) !== info.sha256 || info.source.blob !== job.family.faces.find(f => f.path === job.family.selected).blob) throw new Error('Changed font instance')
        const images = await page.evaluate(async ({ base64, plans }) => {
          const face = new FontFace('training', Uint8Array.from(atob(base64), c => c.charCodeAt(0)))
          await face.load(); document.fonts.add(face)
          try {
            return plans.map(plan => {
              const c = document.createElement('canvas'), ctx = c.getContext('2d'), font = `${plan.size}px training`
              ctx.font = font; const b = ctx.measureText(plan.text)
              c.width = Math.ceil(b.actualBoundingBoxLeft + b.actualBoundingBoxRight) + 12; c.height = Math.ceil(b.actualBoundingBoxAscent + b.actualBoundingBoxDescent) + 12
              ctx.font = font; ctx.fillStyle = 'white'; ctx.fillRect(0, 0, c.width, c.height); ctx.fillStyle = 'black'; ctx.fillText(plan.text, 6 + b.actualBoundingBoxLeft, 6 + b.actualBoundingBoxAscent)
              const condition = plan.condition === 'augmented' ? ['rotate', 'small', 'blur', 'edge'][Math.floor(plan.index / 2) % 4] : 'clean'
              const angle = condition === 'rotate' ? (plan.index % 2 ? -1 : 1) * Math.PI / 30 : 0, scale = condition === 'small' ? .45 : 1
              const out = document.createElement('canvas'); out.width = Math.max(1, Math.ceil(scale * (c.width * Math.cos(angle) + c.height * Math.abs(Math.sin(angle))))); out.height = Math.max(1, Math.ceil(scale * (c.height * Math.cos(angle) + c.width * Math.abs(Math.sin(angle)))))
              const draw = out.getContext('2d'); draw.fillStyle = 'white'; draw.fillRect(0, 0, out.width, out.height); draw.translate(out.width / 2, out.height / 2); draw.rotate(angle); draw.scale(scale, scale)
              if (condition === 'blur') draw.filter = 'blur(.5px)'
              draw.drawImage(c, -c.width / 2, -c.height / 2)
              const width = condition === 'edge' ? Math.max(1, Math.round(out.width * .85)) : out.width, rgba = draw.getImageData(0, 0, width, out.height).data; let raw = ''
              for (let i = 0; i < rgba.length; i += 4) raw += String.fromCharCode(plan.light ? 255 - rgba[i] : rgba[i])
              return { width, height: out.height, pixels: btoa(raw) }
            })
          } finally { document.fonts.delete(face) }
        }, { base64: bytes.toString('base64'), plans: job.plans })
        const windows = [], chunks = []; let offset = 0
        for (const [source, items] of prepareBatch(images).entries()) {
          if (!items.length) throw new Error('Empty training view')
          for (const w of items) { const raw = Buffer.from(w.pixels, 'base64'); windows.push({ source, offset, width: w.width, height: w.height }); chunks.push(raw); offset += raw.length }
        }
        const raw = Buffer.concat(chunks)
        await writeFile(`${prefix}.u8`, raw); await writeFile(`${prefix}.json`, JSON.stringify({ pins: plan.pins, samples: job.plans, windows, sha256: hash(raw), browser: browser.version() }))
        if (++done % 250 === 0) console.log('Case training Chromium', done, plan.jobs.length)
      }
    } finally { await page.close() }
  }))
} finally { await browser.close() }
