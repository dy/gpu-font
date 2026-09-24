import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { base64, readCatalog, rankCatalog, embedWindows, preparationHash, sha256, readHeads, verdict, matchCatalog, foldTwins } from '../src/catalog.mjs'
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
    a => a.version = 4, a => a.version = 1, a => a.dimensions = 64, a => a.referencesPerFace = 2,
    a => a.faces[1].id = a.faces[0].id, a => a.faces[1].family = 'different',
    a => a.vectors.owners = [0, 0, 2], a => a.vectors.owners = [0, 1, 3],
    ...[0, NaN, Infinity, 1e40].map(s => a => a.vectors.scales[0] = s),
    a => a.vectors.data = '???', a => a.vectors.data = '', a => a.vectors.data = Buffer.alloc(384).toString('base64'),
    ...[-1, 1].map(delta => a => a.vectors.data = Buffer.alloc(384 + delta, 1).toString('base64')),
  ]
  for (const change of changes) { const a = catalog(); change(a); assert.throws(() => readCatalog(a, binding)) }
  for (const q of [null, [], vector(0, NaN), vector(0, 0), [1]]) assert.throws(() => rankCatalog(q, readCatalog(catalog(), binding)))
})
test('decoded rows are the unit-normalized float32 dequantization, with overflow judged as stored', () => {
  const row = (entries, n = 128) => Array.from({ length: n }, (_, d) => entries[d] ?? 0)
  const data = catalog(); data.version = 3; delete data.referencesPerFace
  data.vectors = { encoding: 'int8-base64', shape: [3, 128], owners: [0, 1, 2], scales: [.5, 1e-3, 2e36],
    data: Buffer.from(Int8Array.from([...row({ 0: 3, 1: 4 }), ...row({ 0: -127, 127: 127 }), ...row({ 5: 127 })]).buffer).toString('base64') }
  const { vectors } = readCatalog(data, binding)
  const expected = [row({ 0: .6, 1: .8 }), row({ 0: -Math.SQRT1_2, 127: Math.SQRT1_2 }), row({ 5: 1 })].flat()
  assert.ok(vectors.every((v, i) => Math.abs(v - expected[i]) < 1e-7))
  for (let r = 0; r < 3; r++) assert.ok(Math.abs(Math.hypot(...vectors.subarray(r * 128, (r + 1) * 128)) - 1) < 1e-6)
  data.vectors.scales[2] = 3e36  // 127 × 3e36 overflows float32: the stored row would be infinite
  assert.throws(() => readCatalog(data, binding), /Invalid embedding/)
})
test('variable references keep distinct scripts and require every face to own a vector', () => {
  const data = catalog(); data.version = 3; delete data.referencesPerFace
  data.faces.splice(1, 1); data.vectors.owners = [0, 0, 1]
  const before = structuredClone(data)
  for (const query of [vector(0, 1), vector(1, 1), vector(0, -1), vector(0, 1)]) {
    const result = rankCatalog(query, readCatalog(data, binding))
    assert.equal(result[0].family, query[0] < 0 ? 'b' : 'a'); assert.equal(result[0].score, 1)
  }
  assert.deepEqual(data, before)
  for (const change of [a => a.referencesPerFace = 1, a => a.vectors.owners = [0, 0, 0], a => a.vectors.shape = [0, 128], a => a.vectors.shape = [640001, 128], a => a.vectors.shape = [3.5, 128], a => a.vectors.shape = null]) {
    const bad = structuredClone(data); change(bad); assert.throws(() => readCatalog(bad, binding))
  }
})
test('typed verdicts reproduce the head arithmetic; the script filter never returns a face lacking the script', () => {
  const zero = () => Array(128).fill(0), row = (d, v) => { const r = zero(); r[d] = v; return r }
  const artifact = { heads: {
    weight: { weights: [row(0, 1)], bias: [0], scale: 300, offset: 400 }, italic: { weights: [row(1, 2)], bias: [0] },
    script: { weights: [row(0, 1), row(1, 1)], bias: [0, 0], labels: ['Latn', 'Hani'] },
    category: { weights: [row(0, 1), zero()], bias: [0, 0], labels: ['SANS_SERIF', 'SERIF'] },
    fine: { weights: [row(0, 3)], bias: [0], labels: ['/Sans/Geometric'] } } }
  const v = verdict(vector(0, 1), readHeads(artifact))
  assert.equal(v.weight, 700); assert.equal(v.italic, .5)
  assert.equal(v.script[0].label, 'Latn'); assert.ok(Math.abs(v.script[0].p - Math.E / (Math.E + 1)) < 1e-6)
  assert.ok(Math.abs(v.fine[0].p - 1 / (1 + Math.exp(-3))) < 1e-6); assert.equal(v.category[0].label, 'SANS_SERIF')
  assert.equal(readHeads({}), null)
  for (const change of [h => h.weight.scale = NaN, h => h.script.labels = ['Latn'], h => h.script.labels = ['Latn', 'Latn'], h => h.italic.weights = [row(0, NaN)], h => h.fine.weights[0].length = 64,
    h => delete h.italic, h => h.category.bias = [0, Infinity], h => { for (const key of Object.keys(h)) delete h[key] }]) {
    const bad = structuredClone(artifact); change(bad.heads); assert.throws(() => readHeads(bad))
  }
  const data = catalog(); data.faces[0].scripts = ['Latn']; data.faces[1].scripts = ['Latn']; data.faces[2].scripts = ['Latn', 'Hani']
  const decoded = readCatalog(data, binding)
  assert.deepEqual(rankCatalog(vector(1, 1), decoded, { script: 'Hani' }).map(m => m.family), ['b'])
  assert.deepEqual(rankCatalog(vector(1, 1), decoded, { script: 'Latn' }).map(m => m.family), ['a', 'b'])
  assert.deepEqual(matchCatalog(vector(1, 1), decoded, { script: [{ label: 'Hani', p: .9 }] }).map(m => m.family), ['b'])
  assert.deepEqual(matchCatalog(vector(1, 1), decoded, { script: [{ label: 'Arab', p: .9 }] }).map(m => m.family), ['a', 'b'])  // no Arabic font: style across scripts
  assert.deepEqual(matchCatalog(vector(1, 1), decoded).map(m => m.family), ['a', 'b'])
  delete data.faces[0].scripts  // undeclared coverage stays eligible
  assert.deepEqual(rankCatalog(vector(0, 1), readCatalog(data, binding), { script: 'Hani' }).map(m => m.face.id), ['a/regular', 'b'])
})
test('identical designs fold into one row per script, never by chains, under their base name', () => {
  const face = (familyId, family, twins) => ({ id: familyId + '/400', familyId, family, ...(twins ? { twins } : {}) })
  const plex = ['anuphan', 'plexkr', 'plex', 'plexar'], latin = id => ({ Latn: plex.filter(f => f !== id) })
  const faces = [face('anuphan', 'Anuphan', latin('anuphan')), face('plexkr', 'IBM Plex Sans KR', latin('plexkr')), face('plex', 'IBM Plex Sans', latin('plex')),
    face('plexar', 'IBM Plex Sans Arabic', { ...latin('plexar'), Arab: ['kufi'] }), face('kufi', 'Kufi', { Arab: ['plexar'] }),
    face('a', 'A', { Latn: ['b'] }), face('b', 'B', { Latn: ['a', 'c'] }), face('c', 'C', { Latn: ['b'] }), face('d', 'D')]
  const ranked = [['anuphan', .96], ['plexkr', .95], ['plex', .94], ['a', .9], ['plexar', .89], ['kufi', .85], ['b', .8], ['c', .7], ['d', .6]].map(([id, score]) => ({ family: id, score, face: faces.find(f => f.familyId === id) }))
  const before = structuredClone(ranked), folded = foldTwins(ranked, { faces }, { script: 'Latn' })
  assert.deepEqual(folded.map(m => [m.family, m.score, m.siblings]), [['plex', .96, ['Anuphan', 'IBM Plex Sans KR', 'IBM Plex Sans Arabic']], ['a', .9, ['B']], ['kufi', .85, []], ['c', .7, []], ['d', .6, []]])
  assert.equal(folded[0].face.family, 'IBM Plex Sans')  // the base's own face, the group's best score
  assert.deepEqual(ranked, before)
  // Twins are judged in the query's script: families sharing Latin letters stay apart for Arabic text.
  assert.deepEqual(foldTwins(ranked, { faces }, { script: 'Arab' }).map(m => [m.family, m.siblings]).filter(([, s]) => s.length), [['plexar', ['Kufi']]])
  for (const script of [undefined, 'Hani']) assert.deepEqual(foldTwins(ranked, { faces }, { script }).map(m => m.family), ranked.map(m => m.family))
  assert.deepEqual(foldTwins([], { faces }, { script: 'Latn' }), [])
  // A limit keeps the first groups whole: a sibling ranked after the limit still joins its group.
  assert.deepEqual(foldTwins(ranked, { faces }, { script: 'Latn', limit: 2 }).map(m => [m.family, m.siblings]), [['plex', ['Anuphan', 'IBM Plex Sans KR', 'IBM Plex Sans Arabic']], ['a', ['B']]])
})
test('the twin index is kept per catalog and per script: A → A, A → B → A, scripts interleaved', () => {
  const face = (familyId, family, twins) => ({ id: familyId + '/400', familyId, family, ...(twins ? { twins } : {}) })
  const a = { faces: [face('x', 'X', { Latn: ['y'] }), face('y', 'Y', { Latn: ['x'], Arab: ['z'] }), face('z', 'Z', { Arab: ['y'] })] }
  const b = { faces: [face('x', 'X', { Latn: ['z'] }), face('y', 'Y'), face('z', 'Z', { Latn: ['x'] })] }
  const ranked = a.faces.map((f, i) => ({ family: f.familyId, score: 1 - i / 10, face: f }))
  const fold = (catalog, script) => foldTwins(ranked, catalog, { script }).map(m => [m.family, m.siblings])
  for (let n = 0; n < 2; n++) {
    assert.deepEqual(fold(a, 'Latn'), [['x', ['Y']], ['z', []]])
    assert.deepEqual(fold(a, 'Arab'), [['x', []], ['y', ['Z']]])
    assert.deepEqual(fold(b, 'Latn'), [['x', ['Z']], ['y', []]])
    assert.deepEqual(fold(a, undefined), [['x', []], ['y', []], ['z', []]])
  }
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

test('base64 accepts exactly padded standard base64, as the grouped pattern it replaced', () => {
  const grouped = text => /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(text)
  for (const text of ['', 'AAAA', 'AA==', 'AAA=', '+/+/', Buffer.from([255, 254, 1]).toString('base64')]) assert.equal(base64(text), true, text)
  for (const text of ['A', 'AAA', 'AAAAA', 'A===', '====', 'AA=A', 'AA==AAAA', 'AAAA\n', 'AA A', '-_AA', 'AAA\u00e9']) assert.equal(base64(text), false, text)
  let seed = 1
  const next = () => (seed = seed * 16807 % 2147483647) / 2147483647
  for (let i = 0; i < 20000; i++) {
    const text = Array.from({ length: Math.floor(next() * 13) }, () => 'AZaz09+/=-\n'[Math.floor(next() * 11)]).join('')
    assert.equal(base64(text), grouped(text), JSON.stringify(text))
  }
})
