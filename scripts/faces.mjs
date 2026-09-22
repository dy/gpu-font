import { readFile, writeFile, open } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { chromium } from 'playwright'
import { prepareInput } from '../src/input.mjs'

const dir = '.data/faces', hash = bytes => createHash('sha256').update(bytes).digest('hex')
const read = async path => JSON.parse(await readFile(path))
function python(module, command) {
  const p = spawnSync(process.execPath, ['scripts/python.mjs', '-m', module, command], { stdio: 'inherit' })
  if (p.error || p.status !== 0) throw p.error || new Error(`Failed ${module} ${command}`)
}
const command = process.argv[2] || 'data'
if (!['data', 'train', 'evaluate'].includes(command)) throw new Error('Expected data, train or evaluate')
if (command !== 'data') { python('train.faces', command); process.exit(0) }
python('train.faces_data', 'plans')
const plans = await read(`${dir}/plans.json`), faces = (await read('bench/font-faces.json')).faces
const browser = await chromium.launch(), raw = await open(`${dir}/browser.gray`, 'w'), samples = [], digest = createHash('sha256')
let offset = 0
try {
  const page = await browser.newPage()
  for (const face of faces) {
    const bytes = await readFile(`${dir}/${face.id}.ttf`)
    if (hash(bytes) !== face.instanceSha256) throw new Error('Changed face')
    const rows = await page.evaluate(async ({ bytes, plans }) => {
      const face = await new FontFace('pilot', new Uint8Array(bytes).buffer).load()
      document.fonts.clear(); document.fonts.add(face)
      return plans.map(plan => {
        const canvas = document.createElement('canvas'), size = plan.size * plan.dpr, pad = 8
        let c = canvas.getContext('2d'); c.font = `${size}px pilot`
        const m = c.measureText(plan.text)
        canvas.width = Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + 2 * pad
        canvas.height = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + 2 * pad
        c = canvas.getContext('2d'); c.font = `${size}px pilot`; c.fillStyle = 'white'; c.fillRect(0, 0, canvas.width, canvas.height)
        c.fillStyle = 'black'; c.fillText(plan.text, pad + m.actualBoundingBoxLeft, pad + m.actualBoundingBoxAscent)
        const rgba = c.getImageData(0, 0, canvas.width, canvas.height).data
        let pixels = ''; for (let i = 0; i < rgba.length; i += 4) pixels += String.fromCharCode(rgba[i])
        return { ...plan, width: canvas.width, height: canvas.height, pixels: btoa(pixels) }
      })
    }, { bytes: Array.from(bytes), plans })
    for (const { pixels, ...row } of rows) {
      const bytes = Buffer.from(pixels, 'base64'); await raw.writeFile(bytes); digest.update(bytes)
      samples.push({ ...row, family: face.id, offset }); offset += bytes.length
    }
  }
} finally { await raw.close(); await browser.close() }
await writeFile(`${dir}/browser.json`, JSON.stringify({ samples, rawSha256: digest.digest('hex'), facesSha256: hash(await readFile('bench/font-faces.json')), renderer: browser.version() }))
python('train.faces_data', 'render')
const source = await read(`${dir}/source.json`), input = await open(`${dir}/source.gray`), output = await open(`${dir}/prepared.u8`, 'w')
const sourceHash = createHash('sha256'), tensorHash = createHash('sha256'), windows = []
let sourceOffset = 0; offset = 0
try {
  for (const [id, sample] of source.samples.entries()) {
    const gray = Buffer.alloc(sample.width * sample.height)
    if (sample.offset !== sourceOffset || (await input.read(gray, 0, gray.length, sourceOffset)).bytesRead !== gray.length) throw new Error('Invalid source offset/length')
    sourceOffset += gray.length; sourceHash.update(gray)
    const data = new Uint8Array(gray.length * 4)
    for (let i = 0; i < gray.length; i++) { data.fill(gray[i], i * 4, i * 4 + 3); data[i * 4 + 3] = 255 }
    const result = prepareInput({ ...sample, data })
    if (result.status !== 'ok') throw new Error('Lost face sample')
    for (const w of result.windows) {
      const bytes = Uint8Array.from(w.pixels, p => Math.round(p * 255)); await output.writeFile(bytes); tensorHash.update(bytes)
      windows.push({ source: id, offset, width: w.width, height: w.height }); offset += bytes.length
    }
    if (id % 10000 === 0) console.log(`Prepare faces: ${id}/${source.samples.length}`)
  }
  if ((await input.stat()).size !== sourceOffset || sourceHash.digest('hex') !== source.rawSha256) throw new Error('Changed source pixels')
} finally { await input.close(); await output.close() }
await writeFile(`${dir}/prepared.json`, JSON.stringify({ ...source, windows, tensorSha256: tensorHash.digest('hex'), inputSha256: hash(await readFile('src/input.mjs')), normalizerSha256: hash(await readFile('src/prepare.mjs')), runnerSha256: hash(await readFile('scripts/faces.mjs')) }))
console.log(`Prepared ${source.samples.length} face crops`)
