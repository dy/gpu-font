import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { validateRegistry, familyCounts, settle } from '../scripts/foundries.mjs'

const entry = (patch = {}) => ({
  id: 'example', name: 'Example Foundry', url: 'https://example.invalid/', kind: 'foundry', access: 'previews',
  licence: 'commercial', terms: { status: 'none-found', checked: '2026-09-23' }, work: { status: 'excluded' },
  log: [{ date: '2026-09-23', event: 'Surveyed' }], ...patch,
})

test('the committed foundry registry is valid', async () => {
  const registry = JSON.parse(await readFile('bench/foundries.json', 'utf8'))
  assert.deepEqual(validateRegistry(registry), [])
  assert.ok(registry.sources.length > 40)
})

test('a ban needs its quoted clause, and blocks work without recorded permission', () => {
  assert.deepEqual(validateRegistry({ sources: [entry()] }), [])
  const unquoted = validateRegistry({ sources: [entry({ terms: { status: 'ban' } })] })
  assert.match(unquoted.join(), /quoted clause/)
  const banned = { status: 'ban', quote: 'no robots', url: 'https://example.invalid/terms' }
  assert.match(validateRegistry({ sources: [entry({ terms: banned, work: { status: 'planned' } })] }).join(), /without recorded permission/)
  assert.deepEqual(validateRegistry({ sources: [entry({ terms: banned, work: { status: 'indexed', permission: 'email 2026-10-01' } })] }), [])
  assert.deepEqual(validateRegistry({ sources: [entry({ terms: banned, work: { status: 'permission-to-request' } })] }), [])
})

test('ids, vocabularies, URLs and the log are checked', () => {
  const problems = validateRegistry({ sources: [
    entry(), entry(),
    entry({ id: 'Bad Id', url: 'http://x', kind: 'shop', access: 'maybe', work: { status: 'soon' }, log: [] }),
  ] })
  for (const pattern of [/duplicate id/, /lowercase slug/, /must be https/, /kind must be/, /access must be/, /work\.status must be/, /dated events/])
    assert.ok(problems.some(problem => pattern.test(problem)), `${pattern} in ${problems.join('; ')}`)
  assert.deepEqual(validateRegistry({}), ['sources must be an array'])
})

test('an approximate size carries its unit and where it was read', () => {
  const sized = patch => validateRegistry({ sources: [entry(patch)] })
  assert.deepEqual(sized({ familiesAvailable: 5616, availableNote: 'Adobe Fonts browse page' }), [])
  assert.deepEqual(sized({ familiesAvailable: 2000, availableUnit: 'icons', availableNote: 'icons.getbootstrap.com' }), [])
  assert.equal(sized({ familiesAvailable: 5616 }).length, 1)
  assert.equal(sized({ familiesAvailable: 0, availableNote: 'x' }).length, 1)
  assert.equal(sized({ familiesAvailable: 12, availableUnit: 'glyphs', availableNote: 'x' }).length, 1)
})

test('a source folded into Other is counted under its own id, from the sourceId each of its faces carries', async () => {
  const { indexed } = await familyCounts(), other = JSON.parse(await readFile('models/encoder/catalogs/other.json', 'utf8'))
  const bySource = Map.groupBy(other.faces, f => f.sourceId)
  assert.ok(bySource.size > 1 && !bySource.has(undefined), 'Every face in Other names its source')
  for (const [source, faces] of bySource) assert.equal(indexed.get(source), new Set(faces.map(f => f.familyId)).size, source)
  assert.equal(indexed.has('other'), false, 'Other is a group, never a source')
})

test('a shipped catalogue makes a source indexed; an inventory alone makes a planned one inventoried', () => {
  assert.deepEqual(settle({ status: 'planned' }, undefined, 73), { status: 'inventoried', familiesInventoried: 73 })
  assert.deepEqual(settle({ status: 'planned' }, 60, 73), { status: 'indexed', familiesIndexed: 60 })
  assert.deepEqual(settle({ status: 'inventoried', familiesInventoried: 9 }, 12, 14), { status: 'indexed', familiesIndexed: 12 })
  assert.deepEqual(settle({ status: 'planned' }, undefined, undefined), { status: 'planned' })  // nothing collected yet
  assert.deepEqual(settle({ status: 'excluded', reason: 'terms' }, undefined, 5), { status: 'excluded', reason: 'terms' })  // a verdict is never overridden
  assert.deepEqual(settle({ status: 'indexed', familiesIndexed: 4 }, undefined, undefined), { status: 'indexed' })  // a stale count is dropped
})
