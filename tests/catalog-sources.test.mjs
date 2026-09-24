import test from 'node:test'
import assert from 'node:assert/strict'
import { sourceCatalogs } from '../scripts/catalog-sources.mjs'
import { readCatalog, rankCatalog, pack } from '../src/catalog.mjs'

const binding = { encoderSha256: 'encoder', preparationSha256: 'preparation' }
function batch(id, source, dimension) {
  const vector = Buffer.alloc(128); vector[dimension] = 127
  return { catalog: { version: 2, kind: 'font-catalog', ...binding, dimensions: 128, referencesPerFace: 1, sourceManifestSha256: id,
    faces: [{ id, familyId: id, family: id, referenceIds: [id + '-words'] }],
    vectors: { encoding: 'int8-base64', shape: [1, 128], data: vector.toString('base64'), owners: [0], scales: [1 / 127] } },
  records: [{ id: id + '-words', faceId: id, source: { key: source, name: source } }] }
}
test('source collections merge batches, remap owners and deduplicate identical faces without losing scores', () => {
  const a = batch('a', 'fontshare', 0), b = batch('b', 'fontshare', 1), c = batch('c', 'myfonts', 2), original = structuredClone([a, b, c])
  const result = sourceCatalogs([b, c, a, a], binding)
  assert.deepEqual(result.map(c => [c.id, c.name, c.data.faces.map(f => f.id)]), [['fontshare', 'Fontshare', ['a', 'b']], ['myfonts', 'MyFonts', ['c']]])
  assert.deepEqual(result, sourceCatalogs([c, a, b], binding))
  for (const [dimension, family] of [[0, 'a'], [0, 'a'], [1, 'b'], [0, 'a']]) {
    const query = new Float32Array(128); query[dimension] = 1
    assert.deepEqual(rankCatalog(query, readCatalog(result[0].data, binding)).map(m => [m.family, m.score]), [[family, 1], [family === 'a' ? 'b' : 'a', 0]])
  }
  assert.deepEqual([a, b, c], original); assert.deepEqual(sourceCatalogs([], binding), [])
  assert.equal(readCatalog(sourceCatalogs([a], binding)[0].data, binding).families, 1)
})
test('source grouping rejects conflicting identities, unknown sources and incompatible encoder bytes', () => {
  const a = batch('a', 'fontshare', 0)
  assert.throws(() => sourceCatalogs([a, batch('a', 'fontshare', 1)], binding), /Conflicting source face/)
  for (const change of [b => b.records = [], b => b.records[0].faceId = 'wrong', b => b.records[0].source.key = '../fontshare', b => b.catalog.encoderSha256 = 'old']) {
    const bad = structuredClone(a); change(bad); assert.throws(() => sourceCatalogs([bad], binding))
  }
  const bad = structuredClone(a)
  bad.catalog.faces[0].referenceIds.push('other'); bad.records.push({ id: 'other', faceId: 'a', source: { key: 'myfonts' } })
  assert.throws(() => sourceCatalogs([bad], binding), /Conflicting reference sources/)
})
test('4-bit batches merge row by row and keep their encoding; one source packed two ways rejects', () => {
  const four = (id, source, dimension) => {
    const b = batch(id, source, dimension), values = new Int8Array(128); values[dimension] = 7
    b.catalog.vectors = { ...b.catalog.vectors, encoding: 'int4-base64', data: Buffer.from(pack(values, 4)).toString('base64'), scales: [1 / 7] }
    return b
  }
  const [merged] = sourceCatalogs([four('b', 'fontshare', 1), four('a', 'fontshare', 0)], binding), [eight] = sourceCatalogs([batch('b', 'fontshare', 1), batch('a', 'fontshare', 0)], binding)
  assert.equal(merged.data.vectors.encoding, 'int4-base64')
  assert.equal(Buffer.from(merged.data.vectors.data, 'base64').length, 2 * 64)
  const query = new Float32Array(128); query[1] = 1
  assert.deepEqual(rankCatalog(query, readCatalog(merged.data, binding)), rankCatalog(query, readCatalog(eight.data, binding)))
  assert.throws(() => sourceCatalogs([four('a', 'fontshare', 0), batch('b', 'fontshare', 1)], binding), /differently/)
})
