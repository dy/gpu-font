// Exercise the frozen encoder with the existing convolution kernels, independently
// of the demo and its catalog ranking. Projection dimensions are native outputs.
import assert from 'node:assert/strict'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { cpus } from 'node:os'
import { gzipSync, brotliCompressSync } from 'node:zlib'
import { chromium } from 'playwright'
import { readNetwork, inferCPU } from '../src/network.mjs'
import { readCatalog, preparationHash } from '../src/catalog.mjs'

if (process.argv.length !== 2 && process.argv.length !== 5) throw new Error('Pass encoder, matching catalog, and report paths, or no arguments for the deployed model')
const [path = 'models/encoder/encoder.json', catalogPath = 'models/encoder/google-fonts.json', reportPath = 'bench/encoder-runtime.json'] = process.argv.slice(2)
const referencePath = process.argv.length === 2 ? '.data/encoder/runtime-reference.json' : `${dirname(path)}/runtime-reference.json`
const child = spawnSync(process.execPath, ['scripts/python.mjs', '-m', 'scripts.encoder_reference', path, referencePath], { stdio: 'inherit' })
if (child.error || child.status !== 0) throw child.error || new Error('Encoder reference failed')
const bytes = await readFile(path), encoder = JSON.parse(bytes), reference = JSON.parse(await readFile(referencePath))
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
assert.equal(hash(bytes), reference.encoderSha256)
assert.equal(encoder.kind, 'font-encoder'); assert.equal(encoder.dimensions, 128); assert.equal(encoder.fonts, undefined)
readCatalog(JSON.parse(await readFile(catalogPath)), { encoderSha256: hash(bytes), preparationSha256: await preparationHash(encoder.preparation) })
const artifact = encoder
const model = readNetwork(artifact), error = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))
const unit = vector => { const norm = Math.hypot(...vector); assert.ok(norm > 0); return Array.from(vector, v => v / norm) }
let cpuError = 0, gpuError = 0, embeddingError = 0
for (const c of reference.cases) {
  const output = inferCPU(model, { ...c, pixels: Float32Array.from(c.pixels) })
  cpuError = Math.max(cpuError, error(output, c.projection)); embeddingError = Math.max(embeddingError, error(unit(output), c.embedding))
}
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
let timing
const match = { text: 'Quiet rivers flow', font: '56px serif', median: 'of 9 WebGPU and 3 CPU matches after one warm-up' }
try {
  const page = await browser.newPage()
  for (const name of ['network.mjs', 'network-gpu.mjs']) await page.route(`**/src/${name}`, async route => route.fulfill({ contentType: 'text/javascript', body: await readFile(`src/${name}`, 'utf8') }))
  await page.route('**/encoder-check', route => route.fulfill({ contentType: 'text/html', body: '<title>Encoder check</title>' }))
  await page.goto('http://localhost:4179/encoder-check')
  const result = await page.evaluate(async ({ artifact, cases }) => {
    const { readNetwork } = await import('/src/network.mjs'), { createNetworkGPU } = await import('/src/network-gpu.mjs')
    const start = performance.now(), gpu = await createNetworkGPU(readNetwork(artifact)), initializationMs = performance.now() - start
    const outputs = [], times = []
    try {
      for (const index of [0, 0, 1, 2, 3, 0]) {
        const c = cases[index]; outputs.push({ index, values: Array.from(await gpu.infer({ ...c, pixels: Float32Array.from(c.pixels) })) })
      }
      const c = cases[1], input = { ...c, pixels: Float32Array.from(c.pixels) }
      for (let i = 0; i < 22; i++) {
        const start = performance.now(); await gpu.infer(input)
        if (i > 1) times.push(performance.now() - start)
      }
    } finally { gpu.destroy() }
    times.sort((a,b) => a-b)
    return { outputs, timing: { adapter: gpu.info, initializationMs, warmWindowP50Ms: times[9], warmWindowP95Ms: times[18], samples: 20, width: 128, height: 48 } }
  }, { artifact, cases: reference.cases })
  timing = result.timing
  for (const { index, values } of result.outputs) {
    gpuError = Math.max(gpuError, error(values, reference.cases[index].projection))
    embeddingError = Math.max(embeddingError, error(unit(values), reference.cases[index].embedding))
  }
  // One whole match as the page runs it (prepare, infer, rank the catalog), on WebGPU and with WebGPU hidden (the CPU).
  for (const backend of ['webgpu', 'cpu']) {
    const matchPage = await browser.newPage()
    if (backend === 'cpu') await matchPage.addInitScript(() => Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined }))
    await matchPage.route('**/encoder-check', route => route.fulfill({ contentType: 'text/html', body: '<title>Encoder check</title>' }))
    await matchPage.route('**/timing-model.json', async route => route.fulfill({ contentType: 'application/json', body: await readFile(path) }))
    await matchPage.route('**/timing-catalog.json', async route => route.fulfill({ contentType: 'application/json', body: await readFile(catalogPath) }))
    await matchPage.goto('http://localhost:4179/encoder-check')
    const times = await matchPage.evaluate(async runs => {
      const { createMatcher } = await import('/src/match.mjs')
      const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 96
      const c = canvas.getContext('2d'); c.fillStyle = '#fff'; c.fillRect(0, 0, 640, 96); c.fillStyle = '#111'; c.font = '56px serif'; c.textBaseline = 'middle'; c.fillText('Quiet rivers flow', 16, 48)
      const image = c.getImageData(0, 0, 640, 96), matcher = await createMatcher('/timing-catalog.json', '/timing-model.json'), times = []
      try {
        await matcher.match(image)  // warm: kernels compiled, catalog decoded
        for (let i = 0; i < runs; i++) { const start = performance.now(); await matcher.match(image); times.push(performance.now() - start) }
      } finally { matcher.destroy() }
      return times.sort((a, b) => a - b)
    }, backend === 'cpu' ? 3 : 9)
    match[`${backend}Ms`] = times[times.length >> 1]
    await matchPage.close()
  }
} finally { await browser.close() }
assert.ok(cpuError < 1e-4, `CPU projection error ${cpuError}`)
assert.ok(gpuError < 1e-4, `GPU projection error ${gpuError}`)
assert.ok(embeddingError < 1e-4, `Normalized embedding error ${embeddingError}`)
assert.ok(match.webgpuMs < match.cpuMs / 10, `A WebGPU match (${match.webgpuMs} ms) this close to the CPU (${match.cpuMs} ms) ran on the CPU`)
const payload = []
for (const file of [path, catalogPath,
  'src/network.mjs', 'src/network-gpu.mjs', 'src/catalog.mjs', 'src/prepare.mjs', 'src/input.mjs', 'src/line.mjs']) {
  const bytes = await readFile(file)
  payload.push({ path: file, sha256: hash(bytes), bytes: bytes.length, gzipBytes: gzipSync(bytes, { level: 9 }).length, brotliBytes: brotliCompressSync(bytes).length })
}
const report = { encoderSha256: reference.encoderSha256, dimensions: encoder.dimensions, hardware: cpus()[0]?.model, cpuError, gpuError, embeddingError, timing, match, payload,
  componentBytes: payload.reduce((sum, item) => sum + item.bytes, 0),
  scope: 'Existing CPU/Metal convolution kernels with native encoder outputs. Single-window projection timing excludes image preparation and catalog search; match timing is one whole match in Chromium through createMatcher. Bytes include encoder, catalog and existing inference/preparation modules; include catalog parsing/ranking; exclude demo and optional font previews.' }
await mkdir(dirname(reportPath), { recursive: true })
await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n')
console.log(report)
