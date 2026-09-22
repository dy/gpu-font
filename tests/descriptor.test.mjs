import test from 'node:test'
import assert from 'node:assert/strict'
import { descriptor } from '../src/descriptor.mjs'
import { rank, unit } from '../src/rank.mjs'

test('cosine retrieval is scale-invariant with stable ties and signed scores', () => {
  const catalog = [{ family: 'b', vector: [2, 0] }, { family: 'a', vector: [1, 0] }, { family: 'c', vector: [-3, 0] }]
  assert.deepEqual(rank([5, 0], catalog), [{ family: 'a', score: 1 }, { family: 'b', score: 1 }, { family: 'c', score: -1 }])
  assert.deepEqual(rank([1], []), [])
  assert.throws(() => rank([1], catalog))
  for (const v of [[], [0, 0], [NaN, 1], [Infinity]]) assert.throws(() => unit(v))
})
test('descriptor preserves directional evidence and rejects blank/invalid data', () => {
  const vertical = Float32Array.from({ length: 64 }, (_, i) => i % 8 === 4 ? 0 : 1)
  const horizontal = Float32Array.from({ length: 64 }, (_, i) => Math.floor(i / 8) === 4 ? 0 : 1)
  const v = descriptor(vertical, 8, 8), h = descriptor(horizontal, 8, 8)
  assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-6)
  assert.equal(rank(v, [{ family: 'vertical', vector: v }, { family: 'horizontal', vector: h }])[0].family, 'vertical')
  assert.throws(() => descriptor(new Float32Array(64).fill(1), 8, 8))
  assert.throws(() => descriptor(new Float32Array(64).fill(NaN), 8, 8))
  assert.throws(() => descriptor([], 0, 0))
})
