// Offline audit of a preview archive: schema, identity, files, hashes, decoded
// sizes, region bounds, plus the coverage counts the handoff asks for.
import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { validateArchive, readJsonl } from './preview-lib.mjs'

const dir = process.argv[2] ?? '.data/previews/pilot-v1'
const { records, errors, warnings, stray, manifestSha256 } = await validateArchive(dir)
if (errors.length) {
  for (const error of errors) console.error(`  error: ${error}`)
  process.exit(1)
}

const bytesOf = async relative => (await stat(path.join(dir, relative))).size
const unique = key => new Set(records.map(record => (typeof key === 'function' ? key(record) : record[key])))
const count = values => values.reduce((tally, value) => tally.set(value, (tally.get(value) ?? 0) + 1), new Map())

const images = new Set(records.map(record => record.image.path))
const evidence = new Set(records.map(record => record.evidence))
let bytes = 0
for (const file of [...images, ...evidence]) bytes += await bytesOf(file)

const byRecipe = count(records.map(record => record.recipeId))
const byAcquisition = count(records.map(record => record.acquisition))
const bySource = count(records.map(record => record.source?.key ?? 'unknown'))
const scripts = count(records.flatMap(record => record.scripts))
const controlled = records.filter(record => record.recipeId !== 'provided-v1')
const recipesPerFace = new Map()
for (const record of controlled) recipesPerFace.set(record.faceId, (recipesPerFace.get(record.faceId) ?? new Set()).add(record.recipeId))
const partial = [...recipesPerFace].filter(([, set]) => set.size < 5)

const attempts = await readJsonl(path.join(dir, 'attempts.jsonl'))
const say = (label, value) => console.log(`${label.padEnd(30)} ${value}`)

console.log(`Archive ${dir}`)
say('manifest.jsonl sha256', manifestSha256)
say('accepted records', records.length)
say('families', unique('familyId').size)
say('genuine faces', unique('faceId').size)
say('images / evidence files', `${images.size} / ${evidence.size}`)
say('image + evidence bytes', bytes.toLocaleString('en-US'))
say('scripts', [...scripts].map(([tag, n]) => `${tag}:${n}`).join(' '))
say('acquisition', [...byAcquisition].map(([kind, n]) => `${kind}:${n}`).join(' '))
say('sources', [...bySource].map(([key, n]) => `${key}:${n}`).join(' '))
say('recipes', [...byRecipe].map(([id, n]) => `${id}:${n}`).join(' '))
say('faces with all five recipes', `${recipesPerFace.size - partial.length} of ${recipesPerFace.size}`)
say('logged attempts', attempts.length)
for (const [faceId, set] of partial) console.log(`  partial coverage ${faceId}: ${[...set].join(', ')}`)
for (const warning of warnings) console.log(`  warning: ${warning}`)
for (const file of stray) console.log(`  unreferenced file: ${file}`)
for (const error of errors) console.error(`  error: ${error}`)
console.log(errors.length ? `FAILED with ${errors.length} error(s)` : 'All accepted records validate offline')
process.exit(errors.length ? 1 : 0)
