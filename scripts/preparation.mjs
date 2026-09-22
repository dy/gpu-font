import { readFile, writeFile, mkdir, open } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { chromium } from 'playwright'
import { prepareInput } from '../src/input.mjs'
import { prepareLine } from '../src/line.mjs'

const verification = process.argv[2] === 'verification'
const dir = verification ? '.data/verification' : '.data/preparation', hash = b => createHash('sha256').update(b).digest('hex')
const read = async path => JSON.parse(await readFile(path))
function python(command) {
  const result = spawnSync(process.execPath, ['scripts/python.mjs', '-m', 'train.preparation', command, ...(verification ? ['--verification'] : [])], { stdio: 'inherit' })
  if (result.error || result.status !== 0) throw result.error || new Error(`Failed ${command}`)
}
await mkdir(dir, { recursive: true })
const command = verification ? 'data' : process.argv[2] || 'all'
if (!['all', 'data', 'prepare', 'evaluate'].includes(command)) throw new Error('Expected data, prepare or evaluate')
if (['all', 'data'].includes(command)) {
  python('plans')
  const texts = (await read(`${dir}/plans.json`)).texts
  const fonts = (await read('bench/fonts-100.json')).fonts.filter(f => f.split === 'train' || (verification && f.split === 'unknown-test'))
  const browser = await chromium.launch(), raw = await open(`${dir}/browser.gray`, 'w'), samples = []
  let offset = 0
  try {
    const page = await browser.newPage()
    for (const [index, font] of fonts.entries()) {
      const bytes = await readFile(`.data/fonts/${font.id}.ttf`)
      if (hash(bytes) !== font.instanceSha256) throw new Error(`Font changed: ${font.id}`)
      const rows = await page.evaluate(async ({ bytes, texts, index }) => {
        const face = await new FontFace('probe', new Uint8Array(bytes).buffer).load()
        document.fonts.clear(); document.fonts.add(face)
        return texts.map((text, i) => {
          const size = [18, 32, 56][(index + i) % 3], pad = 8, canvas = document.createElement('canvas')
          let c = canvas.getContext('2d'); c.font = `${size}px probe`
          const m = c.measureText(text)
          canvas.width = Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + 2 * pad
          canvas.height = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + 2 * pad
          c = canvas.getContext('2d'); c.font = `${size}px probe`; c.fillStyle = 'white'; c.fillRect(0, 0, canvas.width, canvas.height)
          const x = pad + m.actualBoundingBoxLeft, y = pad + m.actualBoundingBoxAscent
          c.fillStyle = 'black'; c.fillText(text, x, y)
          let prefix = ''
          const regions = text.split(' ').map(word => {
            const n = c.measureText(word), left = x + c.measureText(prefix).width - n.actualBoundingBoxLeft
            prefix += word + ' '
            return [[left, y - n.actualBoundingBoxAscent], [left + n.actualBoundingBoxLeft + n.actualBoundingBoxRight, y - n.actualBoundingBoxAscent], [left + n.actualBoundingBoxLeft + n.actualBoundingBoxRight, y + n.actualBoundingBoxDescent], [left, y + n.actualBoundingBoxDescent]]
          })
          const data = c.getImageData(0, 0, canvas.width, canvas.height).data
          let pixels = ''; for (let i = 0; i < data.length; i += 4) pixels += String.fromCharCode(data[i])
          return { text, size, regions, width: canvas.width, height: canvas.height, pixels: btoa(pixels) }
        })
      }, { bytes: Array.from(bytes), texts, index })
      for (const { pixels, ...row } of rows) {
        const bytes = Buffer.from(pixels, 'base64'); await raw.writeFile(bytes)
        samples.push({ ...row, family: font.id, offset }); offset += bytes.length
      }
    }
    await writeFile(`${dir}/browser.json`, JSON.stringify({ samples, browser: browser.version() }))
  } finally { await raw.close(); await browser.close() }
  python('render')
}
if (['all', 'data', 'prepare'].includes(command)) {
  const source = await read(`${dir}/source.json`), input = await readFile(`${dir}/source.gray`)
  const methods = verification ? ['current', 'deskew-windows'] : ['current', 'groups', 'deskew', 'deskew-windows', 'oracle'], reports = {}
  if (hash(input) !== source.sha256) throw new Error('Changed source pixels')
  for (const method of methods) {
    const output = await open(`${dir}/${method}.u8`, 'w'), windows = [], times = [], angles = []
    let offset = 0
    try {
      for (const [id, sample] of source.samples.entries()) {
        const data = new Uint8Array(sample.width * sample.height * 4), gray = input.subarray(sample.offset, sample.offset + sample.width * sample.height)
        if (gray.length !== data.length / 4) throw new Error('Truncated pixels')
        for (let i = 0; i < gray.length; i++) { data.fill(gray[i], i * 4, i * 4 + 3); data[i * 4 + 3] = 255 }
        const start = performance.now()
        const result = method === 'current' ? prepareInput({ ...sample, data }) : prepareLine({ ...sample, data }, { deskew: method.startsWith('deskew'), sampler: method === 'deskew-windows' ? 'windows' : 'groups', ...(method === 'oracle' ? { angle: sample.angle, regions: sample.regions } : {}) })
        times.push(performance.now() - start); angles.push(result.angle || 0)
        for (const window of result.windows) {
          const bytes = Uint8Array.from(window.pixels, p => Math.round(p * 255))
          await output.writeFile(bytes); windows.push({ source: id, offset, width: window.width, height: window.height }); offset += bytes.length
        }
      }
    } finally { await output.close() }
    const sorted = [...times].sort((a,b) => a-b)
    reports[method] = { windows, angles, sha256: hash(await readFile(`${dir}/${method}.u8`)), timing: { median: sorted[Math.floor(times.length / 2)], p95: sorted[Math.floor(times.length * .95)] } }
    console.log(`${method}: ${windows.length} windows, preparation ${reports[method].timing.median.toFixed(2)} ms median`)
  }
  await writeFile(`${dir}/prepared.json`, JSON.stringify({ ...source, methods: reports, inputSha256: hash(await readFile('src/input.mjs')), lineSha256: hash(await readFile('src/line.mjs')), normalizerSha256: hash(await readFile('src/prepare.mjs')) }))
}
if (['all', 'evaluate'].includes(command)) python('evaluate')
