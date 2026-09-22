import assert from 'node:assert/strict'
import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { readNetwork, inferCPU } from '../src/network.mjs'

const path = process.argv[2]
const runtimePath = process.argv[3] || 'src/network-gpu.mjs'
if (!path) throw new Error('Expected a model JSON path')
await mkdir('.data/artifact-checks', { recursive: true })
const filename = '.data/artifact-checks/reference.json'
const child = spawnSync(process.execPath, ['scripts/python.mjs', '-m', 'scripts.artifact_reference', path, filename], { stdio: 'inherit' })
if (child.error || child.status !== 0) throw child.error || new Error('Reference generation failed')
const bytes = await readFile(path), artifact = JSON.parse(bytes), reference = JSON.parse(await readFile(filename))
assert.equal(createHash('sha256').update(bytes).digest('hex'), reference.modelSha256)
const model = readNetwork(artifact), cases = reference.cases
const error = (a,b) => Math.max(...a.map((value,i) => Math.abs(value-b[i])))
let cpuError = 0, gpuError = 0, timing
for (const c of cases) cpuError = Math.max(cpuError, error(inferCPU(model, { ...c, pixels: Float32Array.from(c.pixels) }), c.logits))
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
try {
  const page = await browser.newPage()
  // Test current runtime source, even while another session edits the deployed demo.
  for (const name of ['network.mjs', 'network-gpu.mjs']) await page.route(`**/src/${name}`, async route => route.fulfill({ contentType: 'text/javascript', body: await readFile(name === 'network-gpu.mjs' ? runtimePath : `src/${name}`, 'utf8') }))
  // Only load the origin; do not initialize the deployed demo's other network.
  await page.route('**/artifact-check', route => route.fulfill({ contentType: 'text/html', body: '<title>Artifact check</title>' }))
  await page.goto('http://localhost:4179/artifact-check')
  const outputs = await page.evaluate(async ({ artifact, cases }) => {
    const { readNetwork } = await import('/src/network.mjs'), { createNetworkGPU } = await import('/src/network-gpu.mjs')
    const start = performance.now(), gpu = await createNetworkGPU(readNetwork(artifact)), results = []
    const initializationMs = performance.now() - start, times = []
    try {
      for (const index of [0,0,1,2,3,0]) {
        const c = cases[index]; results.push({ index, logits: Array.from(await gpu.infer({ ...c, pixels: Float32Array.from(c.pixels) })) })
      }
      const input = { ...cases[1], pixels: Float32Array.from(cases[1].pixels) }
      for (let i = 0; i < 22; i++) {
        const start = performance.now(); await gpu.infer(input)
        if (i > 1) times.push(performance.now() - start)
      }
    } finally { gpu.destroy() }
    times.sort((a,b) => a-b)
    return { results, timing: { adapter: gpu.info, initializationMs, warmWindowP50Ms: times[Math.ceil(times.length*.5)-1], warmWindowP95Ms: times[Math.ceil(times.length*.95)-1], samples: times.length, width:128, height:48 } }
  }, { artifact, cases })
  for (const result of outputs.results) gpuError = Math.max(gpuError, error(result.logits, cases[result.index].logits))
  timing = outputs.timing
} finally { await browser.close() }
assert.ok(cpuError < 1e-4, `CPU error ${cpuError}`); assert.ok(gpuError < 1e-4, `GPU error ${gpuError}`)
const report = { modelSha256: reference.modelSha256, runtimeSha256:createHash('sha256').update(await readFile(runtimePath)).digest('hex'), classes: model.fonts.length, cpuError, gpuError, timing, cases: 4, sequence: 'A → A → maximum → odd → small → A' }
await writeFile(`.data/artifact-checks/${reference.modelSha256}.json`, JSON.stringify(report, null, 2) + '\n')
console.log(report)
