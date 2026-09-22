import { readFile, writeFile, mkdir, cp, rm, readdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { readNetwork } from '../src/network.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const experiment = process.argv.includes('--ten') ? 'ten' : 'hundred'
const bytes = await readFile(`.data/${experiment}/model.json`)
const artifact = JSON.parse(bytes), model = readNetwork(artifact)
if (artifact.preparation.sha256 !== hash(await readFile('src/input.mjs')) || artifact.preparation.normalizerSha256 !== hash(await readFile('src/prepare.mjs'))) throw new Error('Model uses a different normalizer; retrain before building')
const manifest = JSON.parse(await readFile(experiment === 'ten' ? 'bench/fonts.json' : 'bench/fonts-100.json'))
const trained = JSON.parse(await readFile(`bench/${experiment}-training.json`))
if (trained.modelSha256 !== hash(bytes)) throw new Error('Training report does not match model')
if (trained.calibration && (!(trained.calibration.temperature > 0) || !Number.isFinite(trained.calibration.temperature) || !Number.isFinite(trained.calibration.threshold) || trained.calibration.threshold < 0)) throw new Error('Invalid calibration')
const validation = process.argv.includes('--validation')
const tested = validation ? { ...trained.validationInt8, modelSha256: trained.modelSha256, split: 'validation' } : { ...JSON.parse(await readFile(`bench/${experiment}-test.json`)), split: 'test' }
if (tested.modelSha256 !== hash(bytes)) throw new Error('Test report does not match model')
if (experiment === 'hundred') {
  const calibration = await readFile('.data/hundred/calibration.json')
  if (JSON.stringify(JSON.parse(calibration)) !== JSON.stringify(trained.calibration) || (!validation && tested.calibrationSha256 !== hash(calibration))) throw new Error('Calibration does not match the measured model')
}
const catalog = [], assets = []
for (const id of model.fonts) {
  const font = manifest.fonts.find(f => f.id === id)
  if (!font) throw new Error(`Missing font metadata: ${id}`)
  const fontBytes = await readFile(`.data/fonts/${id}.ttf`)
  if (hash(fontBytes) !== font.instanceSha256) throw new Error(`Font checksum failed: ${id}`)
  assets.push([id, fontBytes])
  catalog.push({ id, name: font.family, split: font.split, file: `assets/fonts/${id}.ttf` })
}
// The display face is also included when it is outside the candidate set.
const display = await readFile('.data/fonts/source-serif-4.ttf')
if (hash(display) !== manifest.fonts.find(f => f.id === 'source-serif-4').instanceSha256) throw new Error('Display font checksum failed')
assets.push(['source-serif-4', display])
await mkdir('dist/assets/fonts', { recursive: true })
await cp('demo', 'dist', { recursive: true })
await mkdir('dist/src', { recursive: true })
for (const name of ['prepare', 'input', 'network', 'network-gpu']) await cp(`src/${name}.mjs`, `dist/src/${name}.mjs`)
for (const name of ['descriptor', 'rank', 'model', 'gpu']) await rm(`dist/src/${name}.mjs`, { force: true })
await cp('tokens.css', 'dist/tokens.css')
await writeFile('dist/assets/model.json', bytes)
for (const [id, bytes] of assets) await writeFile(`dist/assets/fonts/${id}.ttf`, bytes)
const shipped = new Set([...model.fonts, 'source-serif-4'])
for (const name of await readdir('dist/assets/fonts')) {
  if (name.endsWith('.ttf') && !shipped.has(name.slice(0, -4))) await rm(`dist/assets/fonts/${name}`)
}
await writeFile('dist/assets/catalog.json', JSON.stringify({ fonts: catalog, modelSha256: hash(bytes), metrics: tested, parameters: trained.parameters, calibration: trained.calibration ? { temperature: trained.calibration.temperature, threshold: trained.calibration.threshold } : null }))
await writeFile('dist/assets/font-notices.json', JSON.stringify({ ...manifest, fonts: manifest.fonts.filter(f => shipped.has(f.id)) }, null, 2))
console.log(`Built dist/: ${catalog.length} fonts, local model, CPU/WebGPU inference. Run npm run demo.`)
