// Exercise the frozen encoder with the existing convolution kernels, independently
// of the demo classifier. Feature channel names are only an adapter for this check.
import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { cpus } from 'node:os'
import { gzipSync, brotliCompressSync } from 'node:zlib'
import { chromium } from 'playwright'
import { readNetwork, inferCPU } from '../src/network.mjs'

const path = 'models/encoder/encoder.json', referencePath = '.data/encoder/runtime-reference.json'
const child = spawnSync(process.execPath, ['scripts/python.mjs', '-m', 'scripts.encoder_reference', path, referencePath], { stdio: 'inherit' })
if (child.error || child.status !== 0) throw child.error || new Error('Encoder reference failed')
const bytes = await readFile(path), encoder = JSON.parse(bytes), reference = JSON.parse(await readFile(referencePath))
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
assert.equal(hash(bytes), reference.encoderSha256)
assert.equal(encoder.kind, 'font-encoder'); assert.equal(encoder.dimensions, 128); assert.equal(encoder.fonts, undefined)
const artifact = { ...encoder, fonts: Array.from({ length: encoder.dimensions }, (_, i) => `feature-${i}`) }
const model = readNetwork(artifact), error = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))
const unit = vector => { const norm = Math.hypot(...vector); assert.ok(norm > 0); return Array.from(vector, v => v / norm) }
let cpuError = 0, gpuError = 0, embeddingError = 0
for (const c of reference.cases) {
  const output = inferCPU(model, { ...c, pixels: Float32Array.from(c.pixels) })
  cpuError = Math.max(cpuError, error(output, c.projection)); embeddingError = Math.max(embeddingError, error(unit(output), c.embedding))
}
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
let timing
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
} finally { await browser.close() }
assert.ok(cpuError < 1e-4, `CPU projection error ${cpuError}`)
assert.ok(gpuError < 1e-4, `GPU projection error ${gpuError}`)
assert.ok(embeddingError < 1e-4, `Normalized embedding error ${embeddingError}`)
const payload = []
for (const path of ['models/encoder/encoder.json', 'models/encoder/google-fonts.json',
  'src/network.mjs', 'src/network-gpu.mjs', 'src/prepare.mjs', 'src/input.mjs', 'src/line.mjs']) {
  const bytes = await readFile(path)
  payload.push({ path, sha256: hash(bytes), bytes: bytes.length, gzipBytes: gzipSync(bytes, { level: 9 }).length, brotliBytes: brotliCompressSync(bytes).length })
}
const report = { encoderSha256: reference.encoderSha256, dimensions: encoder.dimensions, hardware: cpus()[0]?.model, cpuError, gpuError, embeddingError, timing, payload,
  componentBytes: payload.reduce((sum, item) => sum + item.bytes, 0),
  scope: 'Existing CPU/Metal convolution kernels with a feature-channel adapter. Single-window projection timing excludes image preparation and catalog search. Bytes include encoder, catalog and existing inference/preparation modules; exclude the not-yet-released catalog API, demo and optional font previews.' }
await writeFile('bench/encoder-runtime.json', JSON.stringify(report, null, 2) + '\n')
console.log(report)
