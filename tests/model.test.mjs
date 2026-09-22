import test from 'node:test'
import assert from 'node:assert/strict'
import { readModel, encodeCPU, channels, architecture } from '../src/model.mjs'
import { createGPU } from '../src/gpu.mjs'

function artifact() {
  const tensors = {}
  for (let i = 0; i < 4; i++) {
    const shape = [channels[i + 1], channels[i], 3, 3]
    const values = Array(shape.reduce((a, b) => a * b)).fill(0)
    values[4] = 1 // Center weight on channel 0 carries the input through each stride.
    tensors[`features.${i * 2}.weight`] = { shape, values }
    tensors[`features.${i * 2}.bias`] = { shape: [channels[i + 1]], values: Array(channels[i + 1]).fill(0) }
  }
  tensors['project.weight'] = { shape: [32, 48], values: Array(32 * 48).fill(0) }
  tensors['project.weight'].values[0] = 1
  tensors['project.bias'] = { shape: [32], values: Array(32).fill(0) }
  tensors['project.bias'].values[1] = 1
  return { version: 1, architecture, preparation: { width: 128, height: 32, padding: 2 }, tensors,
    catalog: [{ family: 'test', vector: Array.from({ length: 32 }, (_, i) => +(i === 0)) }] }
}
test('CPU convolutions invert, stride, pool and project with independent A → A → B → A outputs', () => {
  const model = readModel(artifact()), white = new Float32Array(4096).fill(1), black = new Float32Array(4096)
  const a = encodeCPU(model, white), snapshot = a.slice()
  assert.deepEqual(Array.from(a), Array.from({ length: 32 }, (_, i) => +(i === 1)))
  assert.deepEqual(encodeCPU(model, white), a)
  const b = encodeCPU(model, black)
  assert.ok(Math.abs(b[0] - Math.SQRT1_2) < 1e-6); assert.equal(b[0], b[1])
  assert.ok(b.slice(2).every(v => v === 0))
  const again = encodeCPU(model, white); again[0] = 20
  assert.deepEqual(a, snapshot); assert.ok(white.every(v => v === 1))
})
test('model rejects incompatible schema, malformed tensors and duplicate/invalid prototypes', () => {
  for (const mutate of [a => a.version = 2, a => a.architecture = 'other', a => a.preparation.width = 64,
    a => delete a.tensors['project.bias'], a => a.tensors['project.bias'].shape = [1],
    a => a.tensors['project.bias'].values.pop(), a => a.tensors['project.bias'].values[0] = Infinity,
    a => a.tensors['project.bias'].values[0] = 1e100, a => a.catalog = [],
    a => a.catalog.push(a.catalog[0]), a => a.catalog[0].vector.fill(0), a => a.catalog[0].vector[0] = NaN]) {
    const a = artifact(); mutate(a); assert.throws(() => readModel(a))
  }
  for (const a of [null, undefined, {}]) assert.throws(() => readModel(a))
})
test('inference rejects empty, truncated, oversized, non-finite and out-of-range pixels and recovers', () => {
  const model = readModel(artifact()), a = new Float32Array(4096)
  const expected = encodeCPU(model, a)
  for (const pixels of [null, [], new Float32Array(0), new Float32Array(4095), new Float32Array(4097), new Float32Array(4096).fill(NaN), new Float32Array(4096).fill(-1), new Float32Array(4096).fill(2)]) {
    assert.throws(() => encodeCPU(model, pixels))
    assert.deepEqual(encodeCPU(model, a), expected)
  }
})
test('GPU initialization explicitly rejects absent GPU and missing adapter', async () => {
  const model = readModel(artifact())
  await assert.rejects(createGPU(model, null), /unavailable/)
  await assert.rejects(createGPU(model, { requestAdapter: async () => null }), /No WebGPU adapter/)
  await assert.rejects(createGPU(model, { requestAdapter: async () => { throw new Error('Adapter failed') } }), /Adapter failed/)
  await assert.rejects(createGPU(model, { requestAdapter: async () => ({ requestDevice: async () => { throw new Error('Device failed') } }) }), /Device failed/)
})
