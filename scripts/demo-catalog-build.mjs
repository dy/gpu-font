// Writes site.json, the page's manifest. The page reads the model and catalogs straight from the repository and loads
// every font from Google Fonts, so nothing heavy is built, copied or committed.
import { readFile, writeFile, mkdir, access, rm } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { readNetwork } from '../src/network.mjs'
import { readCatalog, preparationHash } from '../src/catalog.mjs'
import { sourceCatalogs } from './catalog-sources.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const read = async path => JSON.parse(await readFile(path, 'utf8'))
const modelPath = 'models/encoder/encoder.json'
const bytes = await readFile(modelPath), artifact = JSON.parse(bytes), model = readNetwork(artifact)
const binding = { encoderSha256: hash(bytes), preparationSha256: await preparationHash(artifact.preparation) }
for (const [name, key] of [['input', 'sha256'], ['prepare', 'normalizerSha256'], ['line', 'lineSha256']]) {
  if (hash(await readFile(`src/${name}.mjs`)) !== artifact.preparation[key]) throw new Error(`Encoder preparation changed: ${name}`)
}
const options = [], families = new Set(), previews = {}, inventory = await read('bench/corpus.json')
const byId = new Map(inventory.families.map(f => [f.id, f]))
async function addCatalog(id, name, file) {
  const bytes = await readFile(file), data = JSON.parse(bytes), decoded = readCatalog(data, binding)
  options.push({ id, name, file, sha256: hash(bytes), families: decoded.families, faces: data.faces.length })
  for (const face of data.faces) families.add(face.family.toLowerCase())
  return data
}
const google = await addCatalog('google-fonts', 'Google Fonts', 'models/encoder/google-fonts.json')
// Previews render each family in its own face from Google Fonts; families without Latin show their own letters, read-only.
for (const familyId of new Set(google.faces.map(f => f.familyId))) {
  const family = byId.get(familyId)
  if (!family) throw new Error(`Unknown catalog family: ${familyId}`)
  previews[familyId] = { sampleText: [...(family.alphabets.Latn || Object.values(family.alphabets)[0])].slice(0, 20).join(''), latin: !!family.alphabets.Latn }
}
// Source samples, loaded by name from Google Fonts; a few names differ between the repository and the Google Fonts API.
const renamed = { 'Rounded Mplus 1c': 'M PLUS Rounded 1c' }
const fonts = (await read('bench/fonts-100.json')).fonts.map(font => ({ id: font.id, name: font.family, family: renamed[font.family] ?? font.family }))
const sources = []
for (const version of ['v1', 'v2']) {
  const compiled = `.data/catalogs/preview-pilot-${version}`
  if (!await access(`${compiled}/report.json`).then(() => true, () => false)) continue
  const report = await read(`${compiled}/report.json`), snapshot = await readFile(`${compiled}/inputs.json`)
  if (hash(snapshot) !== report.inputsSha256) throw new Error('Changed preview input snapshot; recompile the catalog')
  const inputs = JSON.parse(snapshot), records = inputs.samples
  if (inputs.manifestSha256 !== report.sourceManifestSha256 || inputs.encoderSha256 !== binding.encoderSha256) throw new Error('Incompatible preview snapshot')
  const product = report.catalogs.find(c => c.recipe === 'all')  // every capture, one row each; bench/preview-swap.json measures it
  if (!product) continue
  const path = `${compiled}/${product.path}`
  if (hash(await readFile(path)) !== product.sha256) throw new Error('Changed preview catalog')
  const data = await read(path)
  sources.push({ catalog: data, records })
}
// Captured specimens stay local and never ship. Only the derived catalogs (names, links, vectors) are
// committed; local captures refresh them, and a fresh clone builds from the committed copies.
// A source ships only when bench/foundries.json records its terms as permitting collection or stating nothing
// against it; banned, restricted, unverified and unlisted sources are derived into .data and stay there.
const derived = 'models/encoder/catalogs', withheld = '.data/catalogs/withheld'
const ledger = new Map((await read('bench/foundries.json')).sources.map(source => [source.id, source]))
const shippable = ({ id }) => ['permitted', 'none-found'].includes(ledger.get(id)?.terms?.status)
const compiled = sourceCatalogs(sources, binding).filter(source => source.id !== 'google-fonts') // Google is indexed in full above.
let shipped = compiled.filter(shippable)
if (compiled.length) {
  await mkdir(derived, { recursive: true }); await mkdir(withheld, { recursive: true })
  for (const source of compiled) await writeFile(`${shippable(source) ? derived : withheld}/${source.id}.json`, JSON.stringify(source.data) + '\n')
  for (const source of compiled.filter(source => !shippable(source))) await rm(`${derived}/${source.id}.json`, { force: true })
  await writeFile(`${derived}/index.json`, JSON.stringify(shipped.map(({ id, name }) => ({ id, name })), null, 2) + '\n')
} else if (await access(`${derived}/index.json`).then(() => true, () => false)) {
  const listed = (await read(`${derived}/index.json`)).filter(shippable)
  shipped = await Promise.all(listed.map(async ({ id, name }) => ({ id, name, data: await read(`${derived}/${id}.json`) })))
}
for (const source of compiled.filter(source => !shippable(source))) console.log(`Withheld ${source.name}: terms ${ledger.get(source.id)?.terms?.status ?? 'not in bench/foundries.json'}; kept in ${withheld}.`)
for (const source of shipped) await addCatalog(source.id, source.name, `${derived}/${source.id}.json`)
const measured = await read('bench/encoder-test.json')
let metrics = measured.encoderSha256 === binding.encoderSha256 && measured.catalogSha256 === options[0].sha256 ? measured.results.groups['split/test'] : null
for (const file of ['bench/encoder-recovery-quality.json', 'bench/encoder-quality.json']) {
  if (metrics || !await access(file).then(() => true, () => false)) continue
  const quality = (await read(file)).reports.after
  if (quality.encoderSha256 === binding.encoderSha256 && quality.catalogSha256 === options[0].sha256) metrics = quality.historical['split/test']
}
if (!metrics && await access('bench/style-quality.json').then(() => true, () => false)) {
  const quality = (await read('bench/style-quality.json')).demo
  if (quality?.encoderSha256 === binding.encoderSha256 && quality.catalogSha256 === options[0].sha256) metrics = quality.metrics
}
if (!metrics) throw new Error('Changed encoder/catalog evaluation')
// Catalog swap (scripts/preview_swap.py): collected fonts the encoder never saw, found in their own catalogs and among Google's.
const swap = await read('bench/preview-swap.json')
if (swap.encoderSha256 !== binding.encoderSha256 || swap.googleCatalogSha256 !== options[0].sha256) throw new Error('Changed encoder/catalog swap benchmark')
metrics = { ...metrics, swapTop5: swap.results.shipped['all sources'].top5, swapMergedTop5: swap.results.shipped['with Google Fonts'].top5 }
// Held-out case and script figures (train.style breakdown): every reference, then capitals, lowercase or other scripts against fewer.
const breakdown = await read('bench/style-breakdown.json'), { groups, cross } = breakdown
if (breakdown.encoderSha256 !== binding.encoderSha256) throw new Error('Changed encoder breakdown')
metrics = { ...metrics, latinTop5: groups['script/Latn'].twin5, otherScriptTop5: groups['case/native'].twin5, hanziTop5: groups['slice/hanzi'].twin5,
  capitalsFromLowercaseTop5: cross.capitalsFromLowercase.twin5, lowercaseFromCapitalsTop5: cross.lowercaseFromCapitals.twin5,
  otherScriptsFromLatinTop5: cross.otherScriptsFromLatin.twin5, hanziFromLatinTop5: cross.hanziFromLatin.twin5 }
await writeFile('site.json', JSON.stringify({ ...binding, modelSha256: binding.encoderSha256, model: modelPath, fonts, catalogs: options, families: families.size, previews,
  parameters: model.layers.reduce((sum, l) => sum + l.weights.length + l.bias.length, 0), metrics }) + '\n')
console.log(`Wrote site.json: ${options.map(o => `${o.name} (${o.families})`).join(', ')}. Fonts load from Google Fonts.`)
