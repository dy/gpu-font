// Local demo assets. Specimen fonts/images are optional previews, not encoder payload.
import { readFile, writeFile, mkdir, cp, access, rm, copyFile, link } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { readNetwork } from '../src/network.mjs'
import { readCatalog, preparationHash } from '../src/catalog.mjs'
import { sourceCatalogs } from './catalog-sources.mjs'
import { copySite } from './site.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const read = async path => JSON.parse(await readFile(path, 'utf8'))
const bytes = await readFile('models/encoder/encoder.json'), artifact = JSON.parse(bytes), model = readNetwork(artifact)
const binding = { encoderSha256: hash(bytes), preparationSha256: await preparationHash(artifact.preparation) }
for (const [name, key] of [['input', 'sha256'], ['prepare', 'normalizerSha256'], ['line', 'lineSha256']]) {
  if (hash(await readFile(`src/${name}.mjs`)) !== artifact.preparation[key]) throw new Error(`Encoder preparation changed: ${name}`)
}
// A clean catalogs folder: no captured specimen from an earlier build may linger.
await rm('dist/assets/catalogs', { recursive: true, force: true })
await mkdir('dist/assets/catalogs', { recursive: true })
await copySite()
await mkdir('dist/src', { recursive: true })
for (const name of ['prepare', 'input', 'line', 'network', 'network-gpu', 'catalog']) await cp(`src/${name}.mjs`, `dist/src/${name}.mjs`)
await writeFile('dist/assets/model.json', bytes)
const options = [], families = new Set(), previews = {}, inventory = await read('bench/corpus.json'), notices = []
const byId = new Map(inventory.families.map(f => [f.id, f]))
async function addCatalog(id, name, bytes) {
  const data = JSON.parse(bytes), decoded = readCatalog(data, binding)
  const file = `assets/catalogs/${id}.json`
  await writeFile(`dist/${file}`, bytes)
  options.push({ id, name, file, sha256: hash(bytes), families: decoded.families, faces: data.faces.length })
  for (const face of data.faces) families.add(face.family.toLowerCase())
  return data
}
const google = await addCatalog('google-fonts', 'Google Fonts', await readFile('models/encoder/google-fonts.json'))
// A clean fonts folder: every face previews in its own static instance, the file its Chromium references were rendered
// from, so a bold or italic match shows bold or italic. The verified instances are immutable, so hard links replace
// 4 GB of copies; the folder is rebuilt from scratch, so nothing writes through them.
await rm('dist/assets/google-fonts', { recursive: true, force: true })
await mkdir('dist/assets/google-fonts', { recursive: true })
const defaults = google.faces.some(f => f.default) ? google.faces.filter(f => f.default) : google.faces
if (new Set(defaults.map(f => f.familyId)).size !== defaults.length || defaults.length !== new Set(google.faces.map(f => f.familyId)).size) throw new Error('Every catalog family needs exactly one default face')
for (const face of defaults) {
  const id = face.familyId, family = byId.get(id), licenses = []
  for (const license of family.licenses) {
    const text = await readFile(`.data/google/${inventory.commit}/${license.path}`, 'utf8')
    const blob = createHash('sha1').update(`blob ${Buffer.byteLength(text)}\0`).update(text).digest('hex')
    if (blob !== license.blob) throw new Error(`Changed license: ${id}`)
    licenses.push({ ...license, text })
  }
  notices.push({ id, family: face.family, sourcePath: face.sourcePath, licenses })
}
const instances = await read('.data/style/reference-instances.json'), sourceFaces = new Map((await read('.data/style/faces.json')).faces.map(f => [f.id, f]))
for (const face of google.faces) {
  const instance = instances[face.id], source = sourceFaces.get(face.id), family = byId.get(face.familyId)
  // The instance was cut from a file at the pinned commit; a default face also matches the catalog's own record of it.
  if (!instance || !family.faces.some(x => x.path === source?.path && x.blob === source.blob) || (face.default && source.blob !== face.sourceBlob)) throw new Error(`Unknown preview face: ${face.id}`)
  const file = `assets/google-fonts/${face.id.replaceAll('/', '--')}.ttf`
  await link(instance.path, `dist/${file}`).catch(() => copyFile(instance.path, `dist/${file}`))
  if (hash(await readFile(`dist/${file}`)) !== instance.sha256) throw new Error(`Changed preview font: ${face.id}`)
  previews[face.id] = { file, sampleText: [...(family.alphabets.Latn || Object.values(family.alphabets)[0])].slice(0, 20).join(''), latin: !!family.alphabets.Latn }
}
// Keep source samples independent of the search catalog.
const manifest = await read('bench/fonts-100.json'), fonts = []
await mkdir('dist/assets/fonts', { recursive: true })
for (const font of manifest.fonts) {
  const bytes = await readFile(`.data/fonts/${font.id}.ttf`)
  if (hash(bytes) !== font.instanceSha256) throw new Error(`Changed source sample: ${font.id}`)
  const file = `assets/fonts/${font.id}.ttf`
  await writeFile(`dist/${file}`, bytes); fonts.push({ id: font.id, name: font.family, file })
}
await cp('.data/fonts/source-serif-4.ttf', 'dist/assets/fonts/source-serif-4.ttf')
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
for (const source of shipped) await addCatalog(source.id, source.name, Buffer.from(JSON.stringify(source.data) + '\n'))
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
await writeFile('dist/assets/catalog.json', JSON.stringify({ ...binding, modelSha256: binding.encoderSha256, fonts, catalogs: options, families: families.size, previews,
  parameters: model.layers.reduce((sum, l) => sum + l.weights.length + l.bias.length, 0), metrics }))
await writeFile('dist/assets/font-notices.json', JSON.stringify({ sourceCommit: inventory.commit, google: notices, samples: manifest, previewArchives: 'Non-Google catalogs ship names, links and embedding vectors only; their specimen captures stay local and are not distributed.' }))
console.log(`Built shared-encoder demo: ${options.map(o => `${o.name} (${o.families})`).join(', ')}. Specimen assets load only when shown.`)
