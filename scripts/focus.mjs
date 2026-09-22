import { readFile, writeFile, open, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { chromium } from 'playwright'
import { prepare } from '../src/prepare.mjs'

const dir = '.data/focus'
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const read = async path => JSON.parse(await readFile(path))
const write = (path, value) => writeFile(path, JSON.stringify(value) + '\n')
function python(...args) {
  const result = spawnSync(process.execPath, ['scripts/python.mjs', '-m', 'train.focus', ...args], { stdio: 'inherit' })
  if (result.error) throw result.error
  if (result.status !== 0) throw new Error(`Focus ${args.join(' ')} failed`)
}

async function renderBrowser(config) {
  python('plans')
  const plans = await read(`${dir}/plans.json`)
  const fonts = await read('bench/fonts.json')
  const browser = await chromium.launch()
  const stream = await open(`${dir}/browser.rgba`, 'w')
  const rawHash = createHash('sha256'), samples = []
  let offset = 0
  try {
    const page = await browser.newPage()
    for (const font of fonts.fonts) {
      const bytes = await readFile(`.data/fonts/${font.id}.ttf`)
      if (hash(bytes) !== font.instanceSha256) throw new Error('Font checksum mismatch')
      const crops = await page.evaluate(async ({ bytes, plans }) => {
        const face = await new FontFace('focus', new Uint8Array(bytes).buffer).load()
        document.fonts.clear(); document.fonts.add(face)
        return plans.map(plan => {
          const canvas = document.createElement('canvas')
          let c = canvas.getContext('2d')
          c.font = `${plan.size}px focus`
          const m = c.measureText(plan.text), pad = 6
          canvas.width = Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + 2 * pad
          canvas.height = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + 2 * pad
          c = canvas.getContext('2d')
          c.fillStyle = '#fff'; c.fillRect(0, 0, canvas.width, canvas.height)
          c.fillStyle = '#000'; c.font = `${plan.size}px focus`
          c.fillText(plan.text, pad + m.actualBoundingBoxLeft, pad + m.actualBoundingBoxAscent)
          // A binary string avoids serializing each channel as a JSON number.
          const pixels = c.getImageData(0, 0, canvas.width, canvas.height).data
          let binary = ''
          for (let p = 0; p < pixels.length; p += 8192) binary += String.fromCharCode(...pixels.subarray(p, p + 8192))
          return { text: plan.text, plan: plan.plan, width: canvas.width, height: canvas.height, base64: btoa(binary) }
        })
      }, { bytes: Array.from(bytes), plans })
      for (const { base64, ...crop } of crops) {
        const rgba = Buffer.from(base64, 'base64')
        samples.push({ ...crop, family: font.id, split: font.split, offset })
        await stream.writeFile(rgba); rawHash.update(rgba); offset += rgba.length
      }
    }
    await write(`${dir}/browser.json`, { samples, renderer: `Chromium ${browser.version()}`,
      configSha256: hash(await readFile('bench/focus.json')), fontsSha256: hash(await readFile('bench/fonts.json')),
      rawSha256: rawHash.digest('hex') })
  } finally { await stream.close(); await browser.close() }
}

async function prepareData(config) {
  const manifest = await read(`${dir}/samples.json`)
  const raw = await open(`${dir}/source.rgba`), output = await open(`${dir}/prepared.f32`, 'w')
  const rawHash = createHash('sha256'), tensorHash = createHash('sha256')
  const size = config.input.width * config.input.height
  const binary = Buffer.alloc(size * 4)
  let offset = 0
  try {
    for (const sample of manifest.samples) {
      const data = Buffer.alloc(sample.width * sample.height * 4)
      if (sample.offset !== offset) throw new Error('Invalid source offset')
      const { bytesRead } = await raw.read(data, 0, data.length, offset)
      if (bytesRead !== data.length) throw new Error('Truncated source')
      offset += data.length; rawHash.update(data)
      const prepared = prepare({ ...sample, data }, config.input)
      if (prepared.status !== 'ok') throw new Error('Blank crop')
      sample.inkHeight = prepared.rect.height * prepared.scale
      for (let p = 0; p < size; p++) binary.writeFloatLE(prepared.pixels[p], p * 4)
      await output.writeFile(binary); tensorHash.update(binary)
    }
    if ((await raw.stat()).size !== offset || rawHash.digest('hex') !== manifest.rawSha256) throw new Error('Source length/checksum mismatch')
  } finally { await raw.close(); await output.close() }
  await write(`${dir}/prepared.json`, { ...manifest, input: config.input,
    preparationSha256: hash(await readFile('src/prepare.mjs')), runnerSha256: hash(await readFile('scripts/focus.mjs')),
    tensorSha256: tensorHash.digest('hex') })
  console.log(`Prepared ${manifest.samples.length} inputs through src/prepare.mjs`)
}

const config = await read('bench/focus.json')
const command = process.argv[2] || 'all'
if (!['all', 'data', 'train'].includes(command)) throw new Error('Expected all, data, or train')
if (command !== 'train') {
  await mkdir(dir, { recursive: true })
  await renderBrowser(config)
  python('render')
  await prepareData(config)
}
if (command !== 'data') for (const policy of config.policies) python('train', policy)
