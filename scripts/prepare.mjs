import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { performance } from 'node:perf_hooks'
import { prepare } from '../src/prepare.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const configBytes = await readFile('bench/pilot.json')
const config = JSON.parse(configBytes)
const fontsBytes = await readFile('bench/fonts.json'), fontsHash = hash(fontsBytes)
const imported = JSON.parse(fontsBytes).fonts
if (imported.length !== config.fonts.length || config.fonts.some((font, i) => ['id', 'file', 'split'].some(key => font[key] !== imported[i][key]))) throw new Error('Font selection changed; run data:import first')
const sourcePaths = process.argv.slice(2)
if (!sourcePaths.length) sourcePaths.push('.data/pillow/samples.json', '.data/browser/samples.json')
const sources = []
for (const path of sourcePaths) {
  const bytes = await readFile(path), manifest = JSON.parse(bytes)
  if (manifest.configSha256 !== hash(configBytes) || manifest.fontsSha256 !== fontsHash) throw new Error(`Stale fixtures: ${path}`)
  sources.push({ path, sha256: hash(bytes), renderer: manifest.renderer, samples: manifest.samples })
}
const samples = sources.flatMap(s => s.samples)
if (!samples.length) throw new Error('No samples to prepare')
const count = config.input.width * config.input.height
const binary = Buffer.alloc(samples.length * count * 4)
const times = []
for (const [index, sample] of samples.entries()) {
  const data = await readFile(sample.raw)
  const start = performance.now()
  const result = prepare({ ...sample, data }, config.input)
  times.push(performance.now() - start)
  if (result.status !== 'ok') throw new Error(`${sample.id}: ${result.status}`)
  for (let p = 0; p < count; p++) binary.writeFloatLE(result.pixels[p], (index * count + p) * 4)
}
times.sort((a, b) => a - b)
await writeFile('.data/prepared.f32', binary)
const manifest = { version: 1, ...config.input, dtype: 'float32-le', configSha256: hash(configBytes), fontsSha256: fontsHash,
  preparationSha256: hash(await readFile('src/prepare.mjs')), tensorSha256: hash(binary),
  sources: sources.map(({ samples, ...source }) => source),
  preparationMs: { median: times[Math.floor(times.length / 2)], p95: times[Math.floor(times.length * .95)] }, samples }
await writeFile('.data/prepared.json', JSON.stringify(manifest, null, 2) + '\n')
console.log(JSON.stringify({ samples: samples.length, shape: [samples.length, config.input.height, config.input.width], preparationMs: manifest.preparationMs }, null, 2))
