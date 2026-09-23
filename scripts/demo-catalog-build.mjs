// Local demo assets. Specimen fonts/images are optional previews, not encoder payload.
import { readFile, writeFile, mkdir, cp, access } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { readNetwork } from '../src/network.mjs'
import { readCatalog, preparationHash } from '../src/catalog.mjs'
import { decodePng, encodePng, cropView } from './preview-lib.mjs'
import { sourceCatalogs } from './catalog-sources.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const read = async path => JSON.parse(await readFile(path, 'utf8'))
const bytes = await readFile('models/encoder/encoder.json'), artifact = JSON.parse(bytes), model = readNetwork(artifact)
const binding = { encoderSha256: hash(bytes), preparationSha256: await preparationHash(artifact.preparation) }
for (const [name, key] of [['input', 'sha256'], ['prepare', 'normalizerSha256'], ['line', 'lineSha256']]) {
  if (hash(await readFile(`src/${name}.mjs`)) !== artifact.preparation[key]) throw new Error(`Encoder preparation changed: ${name}`)
}
await mkdir('dist/assets/catalogs', { recursive: true })
await cp('demo', 'dist', { recursive: true }); await cp('tokens.css', 'dist/tokens.css')
await mkdir('dist/src', { recursive: true })
for (const name of ['prepare', 'input', 'line', 'network', 'network-gpu', 'catalog']) await cp(`src/${name}.mjs`, `dist/src/${name}.mjs`)
await writeFile('dist/assets/model.json', bytes)
const options = [], previews = {}, inventory = await read('bench/corpus.json'), notices = []
async function addCatalog(id, name, bytes) {
  const data = JSON.parse(bytes), decoded = readCatalog(data, binding)
  const file = `assets/catalogs/${id}.json`
  await writeFile(`dist/${file}`, bytes)
  options.push({ id, name, file, sha256: hash(bytes), families: decoded.families, faces: data.faces.length })
  return data
}
const google = await addCatalog('google-fonts', 'Google Fonts', await readFile('models/encoder/google-fonts.json'))
await mkdir('dist/assets/google-fonts', { recursive: true })
for (const face of google.faces) {
  const id = face.familyId, path = `.data/encoder/faces/${id}.ttf`, info = await read(`.data/encoder/faces/${id}.json`)
  const font = await readFile(path)
  if (info.source.blob !== face.sourceBlob || info.sha256 !== hash(font)) throw new Error(`Changed preview font: ${id}`)
  const file = `assets/google-fonts/${id}.ttf`
  await writeFile(`dist/${file}`, font)
  const family = inventory.families.find(f => f.id === id), licenses = []
  for (const license of family.licenses) {
    const text = await readFile(`.data/google/${inventory.commit}/${license.path}`, 'utf8')
    const blob = createHash('sha1').update(`blob ${Buffer.byteLength(text)}\0`).update(text).digest('hex')
    if (blob !== license.blob) throw new Error(`Changed license: ${id}`)
    licenses.push({ ...license, text })
  }
  notices.push({ id, family: face.family, sourcePath: face.sourcePath, licenses })
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
  const archive = `.data/previews/pilot-${version}`, compiled = `.data/catalogs/preview-pilot-${version}`
  if (!await access(`${compiled}/report.json`).then(() => true, () => false)) continue
  const report = await read(`${compiled}/report.json`), snapshot = await readFile(`${compiled}/inputs.json`)
  if (hash(snapshot) !== report.inputsSha256) throw new Error('Changed preview input snapshot; recompile the catalog')
  const inputs = JSON.parse(snapshot), records = inputs.samples
  if (inputs.manifestSha256 !== report.sourceManifestSha256 || inputs.encoderSha256 !== binding.encoderSha256) throw new Error('Incompatible preview snapshot')
  const product = report.catalogs.find(c => c.recipe === 'words')
  if (!product) continue
  const path = `${compiled}/${product.path}`
  if (hash(await readFile(path)) !== product.sha256) throw new Error('Changed preview catalog')
  const data = await read(path)
  sources.push({ catalog: data, records })
  for (const face of data.faces) {
    if (previews[face.id]) continue
    const record = records.find(r => r.id === face.referenceIds[0])
    const bytes = await readFile(`${archive}/${record.image.path}`)
    if (hash(bytes) !== record.image.sha256) throw new Error('Changed reference image')
    const image = cropView(decodePng(bytes), record.region), file = `assets/catalogs/${hash(Buffer.from(face.id))}.png`
    await writeFile(`dist/${file}`, encodePng(image))
    previews[face.id] = { image: file, text: record.text, width: image.width, height: image.height }
  }
}
for (const source of sourceCatalogs(sources, binding)) {
  if (source.id === 'google-fonts') continue // The complete Google corpus is already indexed.
  await addCatalog(source.id, source.name, Buffer.from(JSON.stringify(source.data) + '\n'))
}
const measured = await read('bench/encoder-test.json')
let metrics = measured.encoderSha256 === binding.encoderSha256 && measured.catalogSha256 === options[0].sha256 ? measured.results.groups['split/test'] : null
for (const file of ['bench/encoder-recovery-quality.json', 'bench/encoder-quality.json']) {
  if (metrics || !await access(file).then(() => true, () => false)) continue
  const quality = (await read(file)).reports.after
  if (quality.encoderSha256 === binding.encoderSha256 && quality.catalogSha256 === options[0].sha256) metrics = quality.historical['split/test']
}
if (!metrics) throw new Error('Changed encoder/catalog evaluation')
await writeFile('dist/assets/catalog.json', JSON.stringify({ ...binding, modelSha256: binding.encoderSha256, fonts, catalogs: options, previews,
  parameters: model.layers.reduce((sum, l) => sum + l.weights.length + l.bias.length, 0), metrics }))
await writeFile('dist/assets/font-notices.json', JSON.stringify({ sourceCommit: inventory.commit, google: notices, samples: manifest, previewArchives: 'Local research captures; see .data/previews/pilot-v*/report.md. Not a redistribution grant.' }))
console.log(`Built shared-encoder demo: ${options.map(o => `${o.name} (${o.families})`).join(', ')}. Specimen assets load only when shown.`)
