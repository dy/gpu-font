// The foundry and source registry: what each source offers, its licence, what its
// terms allow, and where our work with it stands. `bench/foundries.json` holds the
// decisions and the dated log; this script refreshes the indexed counts from the
// artifacts themselves, so the published figures are never typed by hand.
//
//   node scripts/foundries.mjs           refresh counts and print the table
//   node scripts/foundries.mjs --check   validate only
import { readdir, readFile, writeFile } from 'node:fs/promises'

const REGISTRY = 'bench/foundries.json'
const CATALOGS = 'models/encoder/catalogs'
export const KINDS = ['library', 'foundry', 'retailer', 'aggregator', 'platform', 'icons', 'music', 'math']
export const ACCESS = ['files', 'previews', 'none']
export const TERMS = ['permitted', 'none-found', 'restricted', 'ban', 'not-verified']
export const UNITS = ['families', 'fonts', 'icons']
export const WORK = ['indexed', 'partial', 'inventoried', 'planned', 'paused', 'permission-to-request', 'awaiting-permission', 'declined', 'excluded']

/** Structural check; returns a list of problems. */
export function validateRegistry(registry) {
  const problems = [], ids = new Set()
  if (!Array.isArray(registry?.sources)) return ['sources must be an array']
  for (const source of registry.sources) {
    const at = source?.id ?? '(no id)'
    if (!/^[a-z0-9][a-z0-9-]*$/.test(source?.id ?? '')) problems.push(`${at}: id must be a lowercase slug`)
    if (ids.has(source.id)) problems.push(`${at}: duplicate id`)
    ids.add(source.id)
    if (!source.name) problems.push(`${at}: name is required`)
    if (!/^https:\/\//.test(source.url ?? '')) problems.push(`${at}: url must be https`)
    if (!KINDS.includes(source.kind)) problems.push(`${at}: kind must be one of ${KINDS.join(', ')}`)
    if (!ACCESS.includes(source.access)) problems.push(`${at}: access must be one of ${ACCESS.join(', ')}`)
    if (!TERMS.includes(source.terms?.status)) problems.push(`${at}: terms.status must be one of ${TERMS.join(', ')}`)
    if (['ban', 'restricted'].includes(source.terms?.status) && !(source.terms.quote && /^https:\/\//.test(source.terms.url ?? '')))
      problems.push(`${at}: a ${source.terms.status} verdict needs the quoted clause and its URL`)
    if (!WORK.includes(source.work?.status)) problems.push(`${at}: work.status must be one of ${WORK.join(', ')}`)
    // Nothing is indexed against a written ban unless permission is on record.
    if (source.terms?.status === 'ban' && ['indexed', 'partial', 'planned'].includes(source.work?.status) && !source.work.permission)
      problems.push(`${at}: work cannot proceed against a ban without recorded permission`)
    if (!Array.isArray(source.log) || !source.log.length || source.log.some(entry => !/^\d{4}-\d{2}-\d{2}$/.test(entry.date ?? '') || !entry.event))
      problems.push(`${at}: log needs dated events`)
    // An approximate size, in the unit the source itself states, with where it was read.
    if ('familiesAvailable' in source && !(Number.isInteger(source.familiesAvailable) && source.familiesAvailable > 0 && typeof source.availableNote === 'string' && source.availableNote))
      problems.push(`${at}: familiesAvailable needs a positive count and an availableNote saying where it was read`)
    if ('availableUnit' in source && !UNITS.includes(source.availableUnit)) problems.push(`${at}: availableUnit must be one of ${UNITS.join(', ')}`)
  }
  return problems
}

/** Family counts per source, read from the artifacts: families in each shipped catalogue
 *  (models/encoder/catalogs, plus the Google corpus) and in the open-font inventory. */
export async function familyCounts() {
  const indexed = new Map(), inventoried = new Map()
  const tally = (map, id, key) => (map.get(id) ?? map.set(id, new Set()).get(id)).add(key)
  const read = async file => { try { return JSON.parse(await readFile(file, 'utf8')) } catch { return null } }
  // The corpus is counted by its own entries, matching the published 2,004.
  for (const family of (await read('bench/corpus.json'))?.families ?? []) if (!family.excluded) tally(indexed, 'google-fonts', family.id)
  for (const file of (await readdir(CATALOGS)).filter(name => name.endsWith('.json') && name !== 'index.json'))
    // A grouped catalogue (Other) tags each face with the source it came from; that source is the one indexed.
    for (const face of (await read(`${CATALOGS}/${file}`))?.faces ?? []) tally(indexed, face.sourceId ?? file.slice(0, -5), face.familyId)
  for (const family of (await read('bench/open-fonts.json'))?.families ?? []) if (!family.excluded) tally(inventoried, family.source, family.family)
  const sizes = map => new Map([...map].map(([id, set]) => [id, set.size]))
  return { indexed: sizes(indexed), inventoried: sizes(inventoried) }
}

async function main() {
  const registry = JSON.parse(await readFile(REGISTRY, 'utf8'))
  const problems = validateRegistry(registry)
  if (problems.length) { for (const problem of problems) console.error(`  ${problem}`); process.exit(1) }
  if (process.argv.includes('--check')) return console.log(`${registry.sources.length} sources valid`)
  const { indexed, inventoried } = await familyCounts()
  // A shipped catalogue is what makes a source indexed; private captures never enter this public registry.
  for (const source of registry.sources) {
    delete source.work.familiesIndexed; delete source.work.familiesInventoried; delete source.work.familiesHeldLocally
    if (indexed.has(source.id) && ['planned', 'inventoried', 'partial'].includes(source.work.status)) source.work.status = 'indexed'
    if (['indexed', 'partial'].includes(source.work.status) && indexed.has(source.id)) source.work.familiesIndexed = indexed.get(source.id)
    else if (source.work.status === 'inventoried' && inventoried.has(source.id)) source.work.familiesInventoried = inventoried.get(source.id)
  }
  registry.updated = new Date().toISOString().slice(0, 10)
  await writeFile(REGISTRY, JSON.stringify(registry, null, 2) + '\n')
  for (const source of registry.sources)
    console.log(`${source.name.slice(0, 30).padEnd(31)} ${source.access.padEnd(9)} ${source.terms.status.padEnd(13)} ${source.work.status.padEnd(22)} ${source.work.familiesIndexed ?? (source.work.familiesInventoried ? `${source.work.familiesInventoried} inventoried` : '-')}`)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
