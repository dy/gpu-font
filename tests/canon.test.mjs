import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { matcher } from '../scripts/canon.mjs'

const family = (name, status, source = 'test') => ({ name, status, source })
const row = (classic, name, relation) => ({ classic, family: name, relation, evidence: 'https://example.invalid/' })

// Fonts In Use pages are fontsinuse.com/typefaces/<number>: anything else means the wrong Wikidata property was read.
test('every Fonts In Use identifier in the canon is a typeface page number', async () => {
  const { typefaces } = JSON.parse(await readFile('bench/canon.json', 'utf8'))
  const ids = typefaces.filter(face => 'fontsInUse' in face).map(face => face.fontsInUse)
  assert.ok(ids.length > 100, `${ids.length} identifiers`)
  assert.deepEqual(ids.filter(id => !/^[1-9]\d*$/.test(id)), [])
  assert.equal(typefaces.find(face => face.name === 'Helvetica')?.fontsInUse, '44')
})

test('an open clone on file outranks the original held from a preview-only source', () => {
  const lookup = matcher([family('Palatino', 'held-locally', 'previews'), family('P052', 'inventoried', 'urw-base35')], [row('Palatino', 'P052', 'clone')])
  assert.deepEqual(lookup('Palatino'), { status: 'inventoried', source: 'urw-base35', via: 'P052', match: 'clone',
    equivalents: [{ family: 'P052', relation: 'clone', status: 'inventoried', evidence: 'https://example.invalid/' }] })
})

test('at equal status the name itself wins over a twin', () => {
  const lookup = matcher([family('Nimbus Roman', 'inventoried'), family('Nimbus Roman No. 9 L', 'inventoried')], [row('Nimbus Roman No. 9 L', 'Nimbus Roman', 'same-design')])
  assert.equal(lookup('Nimbus Roman No. 9 L').match, 'exact')
})

test('metric twins and inspirations are listed but never count as coverage', () => {
  const lookup = matcher([family('Carlito', 'indexed'), family('Jost', 'indexed')], [row('Calibri', 'Carlito', 'metric-compatible'), row('Futura', 'Jost', 'inspired')])
  assert.equal(lookup('Calibri').status, 'missing')
  assert.equal(lookup('Calibri').equivalents[0].relation, 'metric-compatible')
  assert.equal(lookup('Futura').status, 'missing')
})

test('a documented relation overrides the whole-word name heuristic', () => {
  const known = [family('Libre Caslon Text', 'indexed'), family('Adobe Caslon', 'held-locally')]
  assert.equal(matcher(known, [])('Caslon').via, 'Libre Caslon Text')
  const lookup = matcher(known, [row('Caslon', 'Libre Caslon Text', 'inspired')])
  assert.deepEqual([lookup('Caslon').via, lookup('Caslon').match], ['Adobe Caslon', 'related'])
})

test('collective names and suffixes resolve: "Croscore fonts" reaches Arimo', () => {
  const lookup = matcher([family('Arimo', 'indexed')], [row('Croscore fonts', 'Arimo', 'same-design')])
  assert.equal(lookup('Croscore fonts').via, 'Arimo')
})

test('the committed canon never counts private captures', async () => {
  const canon = JSON.parse(await readFile('bench/canon.json', 'utf8'))
  const held = canon.typefaces.filter(entry => entry.coverage.status === 'held-locally').map(entry => entry.name)
  assert.deepEqual(held, [])
})

test('every committed equivalent names a known relation and cites a page', async () => {
  const data = JSON.parse(await readFile('bench/equivalents.json', 'utf8'))
  const pairs = new Set()
  for (const entry of data.equivalents) {
    assert.ok(entry.relation in data.relations, `${entry.classic}: unknown relation ${entry.relation}`)
    assert.match(entry.evidence, /^https:\/\//, `${entry.classic}: evidence must be an https page`)
    const pair = `${entry.classic}|${entry.family}`
    assert.ok(!pairs.has(pair), `duplicate ${pair}`)
    pairs.add(pair)
  }
})
