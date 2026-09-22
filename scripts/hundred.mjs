import { readFile, writeFile, open, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { chromium } from 'playwright'
import { prepareInput } from '../src/input.mjs'

const dir = '.data/hundred'
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const read = async path => JSON.parse(await readFile(path))
const write = (path, data) => writeFile(path, JSON.stringify(data) + '\n')
function python(module, ...args) {
  const result = spawnSync(process.execPath, ['scripts/python.mjs', '-m', module, ...args], { stdio: 'inherit' })
  if (result.error) throw result.error
  if (result.status !== 0) throw new Error(`Failed: ${module} ${args.join(' ')}`)
}
async function render(config) {
  python('train.hundred_data', 'plans')
  const plans = await read(`${dir}/plans.json`)
  const fonts = (await read('bench/fonts-100.json')).fonts
  const browser = await chromium.launch()
  const raw = await open(`${dir}/browser.gray`, 'w')
  const digest = createHash('sha256'), samples = []
  let offset = 0
  try {
    const page = await browser.newPage()
    for (const face of fonts) {
      const family = face.id, bytes = await readFile(`.data/fonts/${family}.ttf`)
      if (hash(bytes) !== face.instanceSha256) throw new Error('Font checksum mismatch')
      const crops = await page.evaluate(async ({ bytes, plans }) => {
        const face = await new FontFace('training', new Uint8Array(bytes).buffer).load()
        document.fonts.clear(); document.fonts.add(face)
        return plans.map(plan => {
          const canvas = document.createElement('canvas'), pad = 6 * plan.dpr, size = plan.size * plan.dpr
          let ctx = canvas.getContext('2d')
          ctx.font = `${size}px training`
          const m = ctx.measureText(plan.text)
          canvas.width = Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + 2 * pad
          canvas.height = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + 2 * pad
          ctx = canvas.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height)
          ctx.fillStyle = '#000'; ctx.font = `${size}px training`
          ctx.fillText(plan.text, pad + m.actualBoundingBoxLeft, pad + m.actualBoundingBoxAscent)
          const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data
          let raw = ''
          for (let i = 0; i < rgba.length; i += 4) raw += String.fromCharCode(rgba[i])
          return { ...plan, width: canvas.width, height: canvas.height, pixels: btoa(raw) }
        })
      }, { bytes: Array.from(bytes), plans: plans.filter(p => face.split === 'train' || face.split === `unknown-${p.role}`).map(p => ({ ...p, role: face.split === 'train' ? p.role : face.split })) })
      for (const { pixels, ...sample } of crops) {
        const bytes = Buffer.from(pixels, 'base64')
        samples.push({ ...sample, family, offset }); offset += bytes.length
        await raw.writeFile(bytes); digest.update(bytes)
      }
      console.log(`Browser: ${family}`)
    }
    await write(`${dir}/browser.json`, { samples, dataConfig: config, renderer: `Chromium ${browser.version()}`, rawSha256: digest.digest('hex') })
  } finally { await raw.close(); await browser.close() }
  python('train.hundred_data', 'render')
  const source = await read(`${dir}/source.json`)
  const input = await open(`${dir}/source.gray`), output = await open(`${dir}/prepared.u8`, 'w')
  const sourceHash = createHash('sha256'), tensorHash = createHash('sha256'), windows = []
  let sourceOffset = 0, offsetOut = 0
  try {
    for (const [id, sample] of source.samples.entries()) {
      const gray = Buffer.alloc(sample.width * sample.height)
      if (sample.offset !== sourceOffset) throw new Error('Invalid source offset')
      if ((await input.read(gray, 0, gray.length, sourceOffset)).bytesRead !== gray.length) throw new Error('Truncated source pixels')
      sourceOffset += gray.length; sourceHash.update(gray)
      const rgba = new Uint8Array(gray.length * 4)
      for (let i = 0; i < gray.length; i++) { rgba.fill(gray[i], i * 4, i * 4 + 3); rgba[i * 4 + 3] = 255 }
      const prepared = prepareInput({ ...sample, data: rgba }, config.input)
      if (prepared.status !== 'ok') throw new Error(`Empty text: ${id}, ${sample.family}, ${sample.text}, ${sample.condition}`)
      for (const w of prepared.windows) {
        const pixels = Uint8Array.from(w.pixels, x => Math.round(x * 255))
        windows.push({ source: id, width: w.width, height: w.height, offset: offsetOut, rect: w.rect })
        await output.writeFile(pixels); tensorHash.update(pixels); offsetOut += pixels.length
      }
      if (id % 5000 === 0) console.log(`Prepare ${id}/${source.samples.length}`)
    }
    if ((await input.stat()).size !== sourceOffset || sourceHash.digest('hex') !== source.rawSha256) throw new Error('Source length/checksum mismatch')
  } finally { await input.close(); await output.close() }
  await write(`${dir}/prepared.json`, { ...source, windows, tensorSha256: tensorHash.digest('hex'),
    inputSha256: hash(await readFile('src/input.mjs')), preparationSha256: hash(await readFile('src/prepare.mjs')),
    runnerSha256: hash(await readFile('scripts/hundred.mjs')) })
  console.log(`Prepared ${source.samples.length} images / ${windows.length} windows / ${(offsetOut / 1048576).toFixed(1)} MiB`)
}

const command = process.argv[2] || 'all'
if (!['all', 'data', 'train', 'evaluate'].includes(command)) throw new Error('Expected data, train or evaluate')
const config = await read('bench/hundred.json')
await mkdir(dir, { recursive: true })
if (['all', 'data'].includes(command)) await render(config.data)
if (['all', 'train'].includes(command)) python('train.hundred', 'train')
if (command === 'evaluate') python('train.hundred', 'evaluate')
