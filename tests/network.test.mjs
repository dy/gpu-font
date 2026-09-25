import test from 'node:test'
import assert from 'node:assert/strict'
import { readNetwork, inferCPU, rankWindows, architecture, channels } from '../src/network.mjs'
import { createNetworkGPU } from '../src/network-gpu.mjs'
import { pack } from '../src/catalog.mjs'

function artifact(fonts = ['inter', 'roboto']) {
  const layers = Array.from({ length: 5 }, (_, i) => {
    const shape = i === 4 ? [fonts.length, 64] : [channels[i + 1], channels[i], 3, 3]
    return { shape, scale: Array(shape[0]).fill(1), weights: Buffer.alloc(shape.reduce((a, b) => a * b)).toString('base64'), bias: Array(shape[0]).fill(0) }
  })
  layers[4].bias = fonts.map((_, i) => 1 - i * 2)
  return { version: 1, architecture, fonts, preparation: { width: 128, height: 48, windows: 3 }, layers }
}

test('variable-size inference retains independent exact logits through A → A → B → A', () => {
  const model = readNetwork(artifact()), a = { width: 1, height: 1, pixels: new Float32Array([0]) }, b = { width: 7, height: 3, pixels: new Float32Array(21).fill(.5) }
  const result = inferCPU(model, a)
  assert.deepEqual([...result], [1, -1])
  for (const input of [a, b, a]) assert.deepEqual([...inferCPU(model, input)], [1, -1])
  assert.deepEqual([...result], [1, -1])
  for (const input of [null, {}, { ...a, pixels: new Float32Array() }, { ...a, pixels: new Float32Array(2) }, { ...a, width: 129 }, { ...a, height: 0 }, ...[NaN, Infinity, -1, 2].map(x => ({ ...a, pixels: new Float32Array([x]) }))]) assert.throws(() => inferCPU(model, input))
  assert.deepEqual([...inferCPU(model, a)], [1, -1])
})

test('packed weights preserve signed values and reject either side of the final byte', () => {
  const a = artifact(['inter']), input = { width: 1, height: 1, pixels: new Float32Array([0]) }
  const packed = Buffer.from(a.layers[0].weights, 'base64')
  packed[0] = 128; packed[packed.length - 1] = 127
  a.layers[0].weights = packed.toString('base64'); a.layers[0].scale.fill(.5)
  const exact = Buffer.from(a.layers[4].weights, 'base64')
  const valid = () => {
    const model = readNetwork(a)
    assert.equal(model.layers[0].weights[0], -64)
    assert.equal(model.layers[0].weights.at(-1), 63.5)
    assert.deepEqual([...inferCPU(model, input)], [1])
    assert.deepEqual(rankWindows([inferCPU(model, input)], model.fonts), [{ family: 'inter', score: 1 }])
    return model
  }
  const retained = valid()
  valid()
  for (const bytes of [Buffer.alloc(0), exact.subarray(0, -1), Buffer.concat([exact, Buffer.from([0])])]) {
    a.layers[4].weights = bytes.toString('base64')
    assert.throws(() => readNetwork(a), /weight count/)
    a.layers[4].weights = exact.toString('base64'); valid()
  }
  assert.deepEqual([...inferCPU(retained, input)], [1])
})

test('weights packed at fewer bits decode as the same integers at 8; a byte short or long, or a width outside 2–8, rejects', () => {
  // Layer 0 is 16×1×3×3: 144 values, all within 6 bits.
  const eight = artifact(['inter']), values = Array.from({ length: 144 }, (_, i) => (i * 37) % 63 - 31)
  eight.layers[0].weights = Buffer.from(pack(values, 8)).toString('base64'); eight.layers[0].scale.fill(.25)
  const six = structuredClone(eight), exact = Buffer.from(pack(values, 6)), with0 = layer => ({ ...six, layers: [{ ...six.layers[0], ...layer }, ...six.layers.slice(1)] })
  Object.assign(six.layers[0], { bits: 6, weights: exact.toString('base64') })
  assert.equal(exact.length, 108)
  assert.deepEqual(readNetwork(six).layers[0].weights, readNetwork(eight).layers[0].weights)
  for (const bytes of [exact.subarray(0, -1), Buffer.concat([exact, Buffer.from([0])])]) assert.throws(() => readNetwork(with0({ weights: bytes.toString('base64') })), /weight count/)
  for (const bits of [1, 9, 6.5, '6']) assert.throws(() => readNetwork(with0({ bits })), /packed weights/)
})

test('a codebook layer decodes each index to a shared value times its row scale; a wrong-sized codebook rejects', () => {
  // Layer 0 (16×1×3×3, 144 values) at 4 bits: indices stored signed, -8..7, into 16 shared values.
  const a = artifact(['inter']), codebook = Array.from({ length: 16 }, (_, i) => (i - 8) / 8), indices = Array.from({ length: 144 }, (_, i) => i % 16 - 8)
  a.layers[0] = { ...a.layers[0], bits: 4, codebook, weights: Buffer.from(pack(indices, 4)).toString('base64'), scale: Array(16).fill(2) }
  const weights = readNetwork(a).layers[0].weights
  assert.deepEqual(Array.from(weights.slice(0, 18)), indices.slice(0, 18).map(i => 2 * codebook[i + 8]))
  assert.throws(() => readNetwork({ ...a, layers: [{ ...a.layers[0], codebook: codebook.slice(1) }, ...a.layers.slice(1)] }), /codebook/)
})

test('a rejection threshold ships as a score in (-1, 1) or not at all', () => {
  const a = artifact()
  assert.equal(readNetwork(a).threshold, null)
  assert.equal(readNetwork({ ...a, rejection: { threshold: .62, presentCoverage: .9 } }).threshold, .62)
  for (const threshold of [1, -1, NaN, '0.6']) assert.throws(() => readNetwork({ ...a, rejection: { threshold } }), /rejection/)
})

test('int8 model rejects corrupt schema, tensors, labels, and floating-point overflow', () => {
  const changes = [a => { a.fonts[1] = a.fonts[0] }, a => { a.layers[0].shape[0]++ }, a => { a.layers[0].scale[0] = NaN }, a => { a.layers[0].bias[0] = Infinity }, a => { a.layers[0].bias[0] = 1e200 }, a => { a.layers[0].weights = '!!!' }, a => { a.layers[0].weights = '' }, a => { a.layers.pop() }, a => { a.preparation.height++ }]
  for (const change of changes) { const a = artifact(); change(a); assert.throws(() => readNetwork(a)) }
  assert.throws(() => readNetwork(null))
  for (const dilations of [[], [1, 1, 2], [1, 1, 2, 3], [1, 1, NaN, 2], '1122']) assert.throws(() => readNetwork({ ...artifact(), dilations }))
})

test('window aggregation averages probabilities with deterministic ties and no fake confidence', async () => {
  assert.deepEqual(rankWindows([new Float32Array([0, 0]), new Float32Array([0, 0])], ['roboto', 'inter']), [{ family: 'inter', score: .5 }, { family: 'roboto', score: .5 }])
  const scores = rankWindows([new Float32Array([1000, 998])], ['inter', 'roboto'])
  assert.ok(Math.abs(scores[0].score - 1 / (1 + Math.exp(-2))) < 1e-12)
  assert.throws(() => rankWindows([], ['inter']))
  assert.throws(() => rankWindows([new Float32Array([NaN])], ['inter']))
  await assert.rejects(createNetworkGPU(readNetwork(artifact()), null), /unavailable/)
  await assert.rejects(createNetworkGPU(readNetwork(artifact()), { requestAdapter: async () => null }), /adapter/)
})

test('10,000-class head retains the final class and rejects labels beyond the supported bound', () => {
  const names = Array.from({ length: 10000 }, (_, i) => `font-${i}`), a = artifact(names)
  a.layers[4].bias[9999] = 10
  const model = readNetwork(a), values = inferCPU(model, { width: 1, height: 1, pixels: new Float32Array([0]) })
  assert.equal(values.length, 10000); assert.equal(values[9999], 10)
  assert.equal(rankWindows([values], names)[0].family, 'font-9999')
  assert.throws(() => readNetwork(artifact([...names, 'extra'])), /font labels/)
  assert.throws(() => readNetwork(artifact([])), /font labels/)
})
test('temperature scales logits before window averaging and rejects invalid calibration', () => {
  const names = ['a', 'b'], logits = [new Float32Array([8, 0])]
  const raw = rankWindows(logits, names), calibrated = rankWindows(logits, names, 4)
  assert.ok(calibrated[0].score < raw[0].score)
  assert.ok(Math.abs(calibrated[0].score - 1 / (1 + Math.exp(-2))) < 1e-12)
  assert.ok(Math.abs(calibrated.reduce((sum, p) => sum + p.score, 0) - 1) < 1e-12)
  for (const t of [0, -1, NaN, Infinity, null]) assert.throws(() => rankWindows(logits, names, t), /temperature/)
})

test('the loader retains the model preparation method and rejects unknown methods', () => {
  const a = artifact()
  assert.equal(readNetwork(a).preparation.method, 'windows')
  a.preparation.method = 'deskew-windows'
  const model = readNetwork(a)
  assert.equal(model.preparation.method, 'deskew-windows')
  a.preparation.method = 'unsupported'
  assert.throws(() => readNetwork(a), /preparation method/)
  assert.equal(model.preparation.method, 'deskew-windows')
})

test('the optional context convolution preserves bounded variable-size classification', () => {
  const names = Array.from({ length: 100 }, (_, i) => `font-${i}`), a = artifact(names)
  a.architecture = 'font-conv16-32-48-64-64-v2'
  a.layers.splice(4, 0, { shape: [64, 64, 3, 3], scale: Array(64).fill(1), bias: Array(64).fill(0), weights: Buffer.alloc(64 * 64 * 9).toString('base64') })
  a.dilations = [1, 1, 2, 2, 1]
  const model = readNetwork(a)
  assert.equal(model.strides.length, 5)
  for (const [width, height] of [[1, 1], [7, 3], [128, 48], [1, 1]]) {
    assert.deepEqual([...inferCPU(model, { width, height, pixels: new Float32Array(width * height) })], a.layers[5].bias)
  }
  assert.throws(() => readNetwork({ ...a, dilations: [1, 1, 2, 2] }), /dilations/)
  assert.throws(() => readNetwork({ ...a, layers: a.layers.slice(0, -1) }), /layers/)
})
