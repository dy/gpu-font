import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { base64, readCatalog, rankCatalog, embedWindows, preparationHash, sha256, readHeads, verdict, matchCatalog, foldTwins, pack, unpack, writeStyle, readStyle } from '../src/catalog.mjs'
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
  const artifact = { dimensions: 128, heads: {
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
test('kinds of one family with identical letters fold into one row per script, never by chains, under their base name', () => {
  const face = (familyId, family, twins) => ({ id: familyId + '/400', familyId, family, ...(twins ? { twins } : {}) })
  const plex = ['anuphan', 'plexkr', 'plex', 'plexar'], latin = id => ({ Latn: plex.filter(f => f !== id) })
  const faces = [face('anuphan', 'Anuphan', latin('anuphan')), face('plexkr', 'IBM Plex Sans KR', latin('plexkr')), face('plex', 'IBM Plex Sans', latin('plex')),
    face('plexar', 'IBM Plex Sans Arabic', { ...latin('plexar'), Arab: ['kufi'] }), face('kufi', 'IBM Plex Kufi', { Arab: ['plexar'] }),
    face('ekmukta', 'Ek Mukta', { Latn: ['mukta'] }), face('mukta', 'Mukta', { Latn: ['ekmukta', 'vaani'] }), face('vaani', 'Mukta Vaani', { Latn: ['mukta'] }), face('d', 'D')]
  const ranked = [['anuphan', .96], ['plexkr', .95], ['plex', .94], ['ekmukta', .9], ['plexar', .89], ['kufi', .85], ['mukta', .8], ['vaani', .7], ['d', .6]].map(([id, score]) => ({ family: id, score, face: faces.find(f => f.familyId === id) }))
  const before = structuredClone(ranked), folded = foldTwins(ranked, { faces }, { script: 'Latn' })
  // Anuphan draws IBM Plex Sans's Latin letters but is a font of its own: its own row, not a twin under another name.
  assert.deepEqual(folded.map(m => [m.family, m.score, m.siblings]), [['anuphan', .96, []], ['plex', .95, ['IBM Plex Sans KR', 'IBM Plex Sans Arabic']], ['mukta', .9, ['Ek Mukta']], ['kufi', .85, []], ['vaani', .7, []], ['d', .6, []]])
  assert.equal(folded[1].face.family, 'IBM Plex Sans')  // the base's own face, the group's best score
  assert.deepEqual(ranked, before)
  // Twins are judged in the query's script: families sharing Latin letters stay apart for Arabic text.
  assert.deepEqual(foldTwins(ranked, { faces }, { script: 'Arab' }).map(m => [m.family, m.siblings]).filter(([, s]) => s.length), [['plexar', ['IBM Plex Kufi']]])
  for (const script of [undefined, 'Hani']) assert.deepEqual(foldTwins(ranked, { faces }, { script }).map(m => m.family), ranked.map(m => m.family))
  assert.deepEqual(foldTwins([], { faces }, { script: 'Latn' }), [])
  // A limit keeps the first groups whole: a sibling ranked after the limit still joins its group.
  assert.deepEqual(foldTwins(ranked, { faces }, { script: 'Latn', limit: 2 }).map(m => [m.family, m.siblings]), [['anuphan', []], ['plex', ['IBM Plex Sans KR', 'IBM Plex Sans Arabic']]])
  // A shared generic word is no shared name: Open Sans and Noto Sans draw the same Latin letters yet stay two rows,
  // while a name holding the other's first word folds whichever ranks first.
  const pair = (x, y) => { const fs = [face('x', x, { Latn: ['y'] }), face('y', y, { Latn: ['x'] })]
    return foldTwins(fs.map((f, i) => ({ family: f.familyId, score: 1 - i / 10, face: f })), { faces: fs }, { script: 'Latn' }).map(m => [m.face.family, m.siblings]) }
  assert.deepEqual(pair('Open Sans', 'Noto Sans'), [['Open Sans', []], ['Noto Sans', []]])
  for (const names of [['Mukta', 'Ek Mukta'], ['Ek Mukta', 'Mukta']]) assert.deepEqual(pair(...names), [['Mukta', ['Ek Mukta']]])
})
test('the twin index is kept per catalog and per script: A → A, A → B → A, scripts interleaved', () => {
  const face = (familyId, family, twins) => ({ id: familyId + '/400', familyId, family, ...(twins ? { twins } : {}) })
  const a = { faces: [face('x', 'Sans', { Latn: ['y'] }), face('y', 'Sans Y', { Latn: ['x'], Arab: ['z'] }), face('z', 'Sans Z', { Arab: ['y'] })] }
  const b = { faces: [face('x', 'Sans', { Latn: ['z'] }), face('y', 'Sans Y'), face('z', 'Sans Z', { Latn: ['x'] })] }
  const ranked = a.faces.map((f, i) => ({ family: f.familyId, score: 1 - i / 10, face: f }))
  const fold = (catalog, script) => foldTwins(ranked, catalog, { script }).map(m => [m.family, m.siblings])
  for (let n = 0; n < 2; n++) {
    assert.deepEqual(fold(a, 'Latn'), [['x', ['Sans Y']], ['z', []]])
    assert.deepEqual(fold(a, 'Arab'), [['x', []], ['y', ['Sans Z']]])
    assert.deepEqual(fold(b, 'Latn'), [['x', ['Sans Z']], ['y', []]])
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
  assert.equal(model.fonts, null); assert.equal(model.outputs, artifact.dimensions); assert.equal(data.dimensions, artifact.dimensions)
  const binding = { encoderSha256: await sha256(bytes), preparationSha256: await preparationHash(artifact.preparation) }
  const catalog = readCatalog(data, binding)
  assert.equal(catalog.families, 2004)
  const input = { width: 1, height: 1, pixels: new Float32Array([0]) }
  const a = inferCPU(model, input); assert.equal(a.length, artifact.dimensions)
  assert.deepEqual(inferCPU(model, input), a)
  assert.throws(() => readNetwork({ ...artifact, fonts: ['fake'] }), /encoder/)
  for (const change of [{ dimensions: artifact.dimensions * 2 }, { normalization: 'none' }]) assert.throws(() => readNetwork({ ...artifact, ...change }))
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

test('unionCatalogs searches read catalogs as one, owners shifted past earlier faces', async () => {
  const { unionCatalogs, rankCatalog } = await import('../src/catalog.mjs')
  const axis = i => { const v = new Float32Array(128); v[i] = 1; return v }
  const read = (faces, rows, owners) => ({ faces, vectors: Float32Array.from(rows.flatMap(r => [...r])), owners, families: new Set(faces.map(f => f.familyId)).size, dimensions: 128 })
  const a = read([{ id: 'a/400', familyId: 'a', family: 'A' }], [axis(0)], [0])
  const b = read([{ id: 'b/400', familyId: 'b', family: 'B' }, { id: 'b/700', familyId: 'b', family: 'B' }], [axis(1), axis(2)], [0, 1])
  const all = unionCatalogs([a, b])
  assert.deepEqual([all.faces.map(f => f.id), all.owners, all.vectors.length, all.families], [['a/400', 'b/400', 'b/700'], [0, 1, 2], 3 * 128, 2])
  // Each part's best face wins in the union exactly as it does alone.
  for (const [query, id] of [[axis(0), 'a/400'], [axis(1), 'b/400'], [axis(2), 'b/700']]) assert.equal(rankCatalog(query, all)[0].face.id, id)
  // A family in two catalogs counts once; nothing joins to an empty catalog.
  assert.equal(unionCatalogs([a, a]).families, 1)
  assert.deepEqual(unionCatalogs([]), { faces: [], vectors: new Float32Array(0), owners: [], families: 0, dimensions: undefined })
})

test('subsetCatalog keeps the accepted faces with their own rows, owners renumbered, and ranks them as the whole would', async () => {
  const { subsetCatalog } = await import('../src/catalog.mjs')
  const axis = i => { const v = new Float32Array(128); v[i] = 1; return v }
  const faces = [{ id: 'a/400', familyId: 'a', family: 'A', sourceId: 'x' }, { id: 'b/400', familyId: 'b', family: 'B', sourceId: 'y' }, { id: 'a/700', familyId: 'a', family: 'A', sourceId: 'x' }]
  const whole = { faces, vectors: Float32Array.from([axis(0), axis(1), axis(2), axis(3)].flatMap(r => [...r])), owners: [0, 1, 2, 2], families: 2, dimensions: 128 }
  const before = structuredClone(whole), x = subsetCatalog(whole, f => f.sourceId === 'x')
  assert.deepEqual(whole, before, 'The whole catalog is left as it was')
  assert.deepEqual([x.faces.map(f => f.id), x.owners, x.families], [['a/400', 'a/700'], [0, 1, 1], 1])
  assert.deepEqual(x.vectors, Float32Array.from([axis(0), axis(2), axis(3)].flatMap(r => [...r])))
  for (const [query, id] of [[axis(0), 'a/400'], [axis(3), 'a/700']]) assert.equal(rankCatalog(query, x)[0].face.id, id)
  assert.deepEqual(rankCatalog(axis(1), x).map(m => m.family), ['a'], 'Faces left out never rank')
  assert.deepEqual(subsetCatalog(whole, () => false), { faces: [], vectors: new Float32Array(0), owners: [], families: 0, dimensions: 128 })
  assert.deepEqual(subsetCatalog(whole, () => true), whole, 'Keeping every face is the same catalog')
  assert.deepEqual(subsetCatalog(x, f => f.id === 'a/700'), subsetCatalog(whole, f => f.id === 'a/700'), 'A subset of a subset is the subset of the whole')
})

test('a style link: 180 URL-safe characters that decode to the same direction and re-encode to themselves', async () => {
  const { writeStyle, readStyle, unit } = await import('../src/catalog.mjs')
  const model = '62e556d8b0a34aa18618cc9ddaa34a632f49117480f08b6e6cdcd998103b0605'
  // Signed bytes in base64url: 127, -17, -64 are 7f ef c0, standard "f+/A", URL-safe "f-_A".
  assert.equal(writeStyle(vector(0, 3), model), `62e556d8.fwAA${'A'.repeat(167)}`)
  assert.equal(writeStyle(vector(0, -1), model), `62e556d8.gQAA${'A'.repeat(167)}`)
  assert.equal(writeStyle([127, -17, -64, ...Array(125).fill(0)], model), `62e556d8.f-_A${'A'.repeat(167)}`)
  let seed = 7
  const next = () => (seed = seed * 16807 % 2147483647) / 2147483647 - .5
  for (let i = 0; i < 200; i++) {
    const v = unit(Array.from({ length: 128 }, next)), text = writeStyle(v, model), read = readStyle(text)
    assert.match(text, /^62e556d8\.[A-Za-z0-9_-]{171}$/)
    assert.equal(read.model, '62e556d8')
    assert.ok(read.embedding.reduce((sum, x, d) => sum + x * v[d], 0) > .9999, 'Int8 keeps the direction')
    assert.equal(writeStyle(read.embedding, model), text, 'A link opened and written again is the same link')
  }
  for (const bad of [undefined, null, '', '62e556d8', `62e556d8${'A'.repeat(171)}`, `62e556d8.${'A'.repeat(170)}`, `62e556d8.${'A'.repeat(172)}`, `62E556D8.fwAA${'A'.repeat(167)}`,
    `62e556d8.f+AA${'A'.repeat(167)}`, `62e556d8.fwA=${'A'.repeat(167)}`, ` 62e556d8.fwAA${'A'.repeat(167)}`, `62e556d8.fwAA${'A'.repeat(167)}\n`]) assert.throws(() => readStyle(bad), /Invalid style/, JSON.stringify(bad))
  assert.throws(() => readStyle(`62e556d8.${'A'.repeat(171)}`), /Zero embedding/)
  for (const bad of [vector(0, 0), vector(0, NaN), [1]]) assert.throws(() => writeStyle(bad, model), /embedding/)
})

test('a 64-number embedding: catalogs, rows, heads and style links carry their own dimension', async () => {
  const { packVectors } = await import('../src/references.mjs')
  const binding = { encoderSha256: 'e'.repeat(64), preparationSha256: 'p'.repeat(64) }
  const axis = (d, n = 64) => { const v = new Float32Array(n); v[d] = 1; return Array.from(v) }
  const catalog = readCatalog({ version: 3, kind: 'font-catalog', ...binding, dimensions: 64, faces: [{ id: 'a/400', familyId: 'a', family: 'A' }, { id: 'b/400', familyId: 'b', family: 'B' }],
    vectors: { ...packVectors([axis(0), axis(1)]), owners: [0, 1] } }, binding)
  assert.equal(catalog.dimensions, 64); assert.equal(catalog.vectors.length, 128)
  assert.deepEqual(rankCatalog(axis(1), catalog).map(m => m.family), ['b', 'a'])
  assert.throws(() => rankCatalog(axis(1, 128), catalog), /dimensions/)
  assert.equal(embedWindows([axis(2), axis(2)]).length, 64)
  const heads = readHeads({ dimensions: 64, heads: { weight: { weights: [axis(0)], bias: [0], scale: 300, offset: 400 }, italic: { weights: [axis(1)], bias: [0] },
    script: { weights: [axis(2), axis(3)], bias: [0, 0], labels: ['Latn', 'Cyrl'] }, category: { weights: [axis(4)], bias: [0], labels: ['SERIF'] }, fine: { weights: [axis(5)], bias: [0], labels: ['didone'] } } })
  assert.equal(verdict(axis(3), heads).script[0].label, 'Cyrl')
  const text = writeStyle(axis(7), 'a'.repeat(64)); assert.equal(text.length, 9 + 86); assert.deepEqual(Array.from(readStyle(text).embedding), axis(7))
  assert.throws(() => readHeads({ dimensions: 128, heads: heads }), /head/)
  // Rows of 64 join rows of 64, never rows of 128.
  const other = { version: 3, kind: 'font-catalog', ...binding, dimensions: 128, faces: [{ id: 'c/400', familyId: 'c', family: 'C' }], vectors: { ...packVectors([axis(0, 128)]), owners: [0] } }
  const { joinCatalogs } = await import('../src/references.mjs')
  assert.throws(() => joinCatalogs(other, { version: 3, kind: 'font-catalog', ...binding, dimensions: 64, faces: [{ id: 'a/400', familyId: 'a', family: 'A' }], vectors: { ...packVectors([axis(0)]), owners: [0] } }), /differently/)
})

test('bit packing: the Python exporter\'s streams, every width\'s extremes, partial last bytes, and nothing', () => {
  // Nine values at each width's extremes, packed by train/ten_model.py pack_bits: the two languages share one bit order.
  for (const [bits, stream] of [[4, 'eRBvOg0='], [6, '4QcEvycOPQ=='], [8, 'gX8AAf9+ggP9']]) {
    const top = 2 ** (bits - 1) - 1, values = [-top, top, 0, 1, -1, top - 1, 1 - top, 3, -3], bytes = Buffer.from(stream, 'base64')
    assert.equal(bytes.length, Math.ceil(9 * bits / 8))
    assert.deepEqual(Array.from(unpack(bytes, bits, 9)), values)
    assert.deepEqual(Buffer.from(pack(values, bits)).toString('base64'), stream)
  }
  // Every width round-trips any count; at 8 bits the stream is the int8 bytes themselves.
  for (let bits = 2; bits <= 8; bits++) for (const count of [1, 7, 8, 9, 257]) {
    const top = 2 ** (bits - 1) - 1, values = Array.from({ length: count }, (_, i) => (i * 7919) % (2 * top + 1) - top)
    assert.deepEqual(Array.from(unpack(pack(values, bits), bits, count)), values, `${bits} bits, ${count} values`)
  }
  const bytes = Int8Array.from([-127, 5, 127, -1])
  assert.deepEqual(pack(bytes, 8), new Uint8Array(bytes.buffer))
  assert.deepEqual([pack([], 4).length, unpack(new Uint8Array(0), 4, 0).length], [0, 0])
})

test('a catalog holds up to 100,000 faces: a captured catalogue of one face a family is far past the Google set\'s 8,132', () => {
  const many = n => {
    const faces = Array.from({ length: n }, (_, i) => ({ id: `f${i}`, familyId: `f${i}`, family: `F${i}` })), data = Buffer.alloc(n * 128)
    for (let i = 0; i < n; i++) data[i * 128 + (i % 128)] = 127
    return { ...catalog(), faces, vectors: { encoding: 'int8-base64', shape: [n, 128], data: data.toString('base64'), scales: Array(n).fill(1 / 127), owners: Array.from({ length: n }, (_, i) => i) } }
  }
  assert.equal(readCatalog(many(10001), binding).families, 10001)
  assert.throws(() => readCatalog(many(100001), binding), /unique catalog faces/)
})
