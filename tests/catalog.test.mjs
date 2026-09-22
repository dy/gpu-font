import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { readCatalog, rankCatalog, embedWindows, preparationHash, sha256 } from '../src/catalog.mjs'
import { readNetwork, inferCPU } from '../src/network.mjs'

const binding = { encoderSha256: 'encoder', preparationSha256: 'preparation' }
const vector = (d, value = 127) => Array.from({ length: 128 }, (_, i) => i === d ? value : 0)
function catalog() {
  return { version: 2, kind: 'font-catalog', ...binding, dimensions: 128, referencesPerFace: 1,
    faces: [{ id: 'a/regular', familyId: 'a', family: 'A' }, { id: 'a/bold', familyId: 'a', family: 'A' }, { id: 'b', familyId: 'b', family: 'B' }],
    vectors: { encoding: 'int8-base64', shape: [3, 128], data: Buffer.from([...vector(0), ...vector(1), ...vector(0, -127)]).toString('base64'), scales: [1 / 127, 1 / 127, 1 / 127], owners: [0, 1, 2] } }
}
test('catalogs retain genuine faces, signed cosine scores and deterministic family ranking through A → A → B → A', () => {
  const a = catalog(), b = structuredClone(a); b.faces[2] = { id: 'c', familyId: 'c', family: 'C' }
  for (const data of [a, a, b, a]) {
    const decoded = readCatalog(data, binding), matches = rankCatalog(vector(1, 1), decoded)
    assert.equal(decoded.families, 2); assert.equal(matches.length, 2)
    assert.equal(matches[0].face.id, 'a/bold'); assert.equal(matches[0].score, 1)
    assert.equal(matches[1].family, data.faces[2].familyId); assert.equal(matches[1].score, 0)
    const signed = rankCatalog(vector(0, 1), decoded)
    assert.equal(signed[0].face.id, 'a/regular'); assert.equal(signed[1].score, -1)
  }
  const reversed = structuredClone(a); reversed.faces.reverse(); reversed.vectors.owners = [2, 1, 0]
  assert.deepEqual(rankCatalog(vector(0, 1), readCatalog(reversed, binding)), rankCatalog(vector(0, 1), readCatalog(a, binding)))
  const smallest = structuredClone(a); smallest.faces.length = 1
  smallest.vectors = { encoding: 'int8-base64', shape: [1, 128], data: Buffer.from(vector(0)).toString('base64'), scales: [1 / 127], owners: [0] }
  assert.deepEqual(rankCatalog(vector(0, 1), readCatalog(smallest, binding)).map(m => [m.face.id, m.score]), [['a/regular', 1]])
})
test('catalog parser rejects empty, incompatible, truncated, trailing, zero and overflowing vectors', () => {
  for (const bad of [null, {}, [], { ...catalog(), faces: [] }]) assert.throws(() => readCatalog(bad, binding))
  for (const key of Object.keys(binding)) assert.throws(() => readCatalog(catalog(), { ...binding, [key]: 'other' }), /different/)
  const changes = [
    a => a.version = 3, a => a.version = 1, a => a.dimensions = 64, a => a.referencesPerFace = 2,
    a => a.faces[1].id = a.faces[0].id, a => a.faces[1].family = 'different',
    a => a.vectors.owners = [0, 0, 2], a => a.vectors.owners = [0, 1, 3],
    ...[0, NaN, Infinity, 1e40].map(s => a => a.vectors.scales[0] = s),
    a => a.vectors.data = '???', a => a.vectors.data = '', a => a.vectors.data = Buffer.alloc(384).toString('base64'),
    ...[-1, 1].map(delta => a => a.vectors.data = Buffer.alloc(384 + delta, 1).toString('base64')),
  ]
  for (const change of changes) { const a = catalog(); change(a); assert.throws(() => readCatalog(a, binding)) }
  for (const q of [null, [], vector(0, NaN), vector(0, 0), [1]]) assert.throws(() => rankCatalog(q, readCatalog(catalog(), binding)))
})
test('source embedding averages normalized windows, retaining direction rather than projection magnitude', () => {
  const a = vector(0, 100), b = vector(1, 1), embedding = embedWindows([a, b])
  assert.ok(Math.abs(embedding[0] - Math.SQRT1_2) < 1e-7)
  assert.equal(embedding[0], embedding[1]); assert.ok(embedding.slice(2).every(v => v === 0))
  assert.deepEqual(embedWindows([a]), Float32Array.from(vector(0, 1)))
  for (const p of [null, [], [a, a, a, a], [[]], [vector(0, 0)], [vector(0, NaN)], [a, vector(0, -100)]]) assert.throws(() => embedWindows(p))
})
test('frozen Google catalog binds native encoder outputs and Python preparation hash without synthetic font labels', async () => {
  const bytes = await readFile('models/encoder/encoder.json'), artifact = JSON.parse(bytes), data = JSON.parse(await readFile('models/encoder/google-fonts.json'))
  const model = readNetwork(artifact)
  assert.equal(model.fonts, null); assert.equal(model.outputs, 128)
  const binding = { encoderSha256: await sha256(bytes), preparationSha256: await preparationHash(artifact.preparation) }
  const catalog = readCatalog(data, binding)
  assert.equal(catalog.families, 2004)
  const input = { width: 1, height: 1, pixels: new Float32Array([0]) }
  const a = inferCPU(model, input); assert.equal(a.length, 128)
  assert.deepEqual(inferCPU(model, input), a)
  assert.throws(() => readNetwork({ ...artifact, fonts: ['fake'] }), /encoder/)
  for (const change of [{ dimensions: 64 }, { normalization: 'none' }]) assert.throws(() => readNetwork({ ...artifact, ...change }))
})
