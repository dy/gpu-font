// The catalog benchmark's validation queries rendered by another engine: WebKit (Safari's) or Firefox, through Playwright,
// each as the frozen renderer draws them in Chromium. Writes a plan and a stage folder that `train.style_bench pack`
// reads as benchmark `catalog-<engine>`, so one model can be read on the three rasterizers with the same text.
//   node scripts/style-bench-engines.mjs webkit [--size 12]     # --size renders every query at that size instead
import { readFile, writeFile, mkdir, open } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import playwright from 'playwright'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const engine = process.argv[2], size = process.argv.includes('--size') ? Number(process.argv[process.argv.indexOf('--size') + 1]) : null
if (!['webkit', 'firefox', 'chromium'].includes(engine)) throw new Error('Pass webkit, firefox or chromium')
const name = `catalog-${engine}${size ? `-${size}` : ''}`, stage = `.data/style/stages/${name}/.data/style`
const source = JSON.parse(await readFile('.data/style/catalog-plan.json', 'utf8'))
const plan = { ...source, engine, queries: source.queries.filter(q => q.role === 'validation').map(q => size ? { ...q, size } : q) }
const planBytes = Buffer.from(JSON.stringify(plan))
await mkdir(stage, { recursive: true })
await writeFile(`.data/style/${name}-plan.json`, planBytes)
const instances = JSON.parse(await readFile('.data/style/catalog-instances.json', 'utf8'))
const byFace = new Map()
plan.queries.forEach((q, i) => { if (!byFace.has(q.faceId)) byFace.set(q.faceId, []); byFace.get(q.faceId).push(i) })
const images = new Array(plan.queries.length), file = await open(`${stage}/bench-browser.u8`, 'w'), digest = createHash('sha256')
const browser = await playwright[engine].launch()
let offset = 0, done = 0
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
      images[i] = { width: rendered[n].width, height: rendered[n].height, offset }; await file.write(raw); digest.update(raw); offset += raw.length
    }
    if (++done % 500 === 0) console.log(name, done, '/', byFace.size)
  }
} finally { await browser.close(); await file.close() }
await writeFile(`${stage}/bench-browser.json`, JSON.stringify({ planSha256: hash(planBytes), browser: `${engine} ${browser.version()}`, sha256: digest.digest('hex'), images }) + '\n')
console.log(`${name}: ${plan.queries.length} queries rendered by ${engine} ${browser.version()}`)
