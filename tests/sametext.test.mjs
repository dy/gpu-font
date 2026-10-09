import test from 'node:test'
import assert from 'node:assert/strict'
import { rerank, MIN_TEXT, SHORTLIST } from '../src/sametext.mjs'
import { SIZES } from '../src/references.mjs'

// A drawn line: a white image with a black bar, solid or with a white gap in its middle third, the face's "letterforms".
function drawing(width, gap, height = 48) {
  const data = new Uint8Array(width * height * 4).fill(255)
  for (let y = 12; y < 36; y++) for (let x = 8; x < width - 8; x++) if (!gap || x < width / 3 || x > width * 2 / 3) data.fill(0, (y * width + x) * 4, (y * width + x) * 4 + 3)
  return { width, height, data }
}
// An encoder whose embedding is the window's ink fraction, in sixteen numbers: a gapped bar embeds away from a solid one.
const encoder = { preparation: { width: 128, height: 48, windows: 3 }, project: async windows => windows.map(w => { const ink = 1 - w.pixels.reduce((a, b) => a + b) / w.pixels.length; return Float32Array.from({ length: 16 }, (_, i) => i === 0 ? ink : i === 1 ? 1 - ink : 0) }) }
const rows = [{ family: 'a', score: .8, gap: false }, { family: 'b', score: .82, gap: true }, { family: 'c', score: .7, gap: null }, { family: 'd', score: .6, gap: true }]
const draw = (row, text, size) => row.gap === null ? null : drawing(120 + size, row.gap)
const calls = []
const counting = (row, text, size) => { calls.push([row.family, text, size]); return draw(row, text, size) }

test('a short or missing text ranks nothing: the rows come back as they were', async () => {
  for (const text of ['', ' ', 'a', undefined, 7]) assert.equal(await rerank(rows, text, { embedding: new Float32Array(16), encoder, draw }), rows)
  assert.deepEqual(await rerank([], 'Quiet rivers', { embedding: new Float32Array(16), encoder, draw }), [])
  assert.equal(MIN_TEXT, 2); assert.equal(SHORTLIST, 30)
})

test('the shortlist re-sorts by catalog score plus the cosine with the text drawn in each face; an undrawable face counts its catalog score twice', async () => {
  // The crop is a gapped bar: faces b and d, which draw one, get a text score near 1; a, solid, a lower one; c cannot be drawn.
  const crop = await encoder.project([{ width: 128, height: 48, pixels: Float32Array.from({ length: 128 * 48 }, (_, i) => { const x = i % 128; return x < 43 || x > 85 ? 0 : 1 }) }])
  const result = await rerank(rows, '  Quiet rivers ', { embedding: crop[0], encoder, draw: counting })
  assert.deepEqual(result.map(r => r.family), ['b', 'a', 'd', 'c'])  // b: .82 + ~.96; a: .8 + ~.91; d: .6 + ~.96; c: .7 twice
  for (const r of result) assert.ok(r.combined <= 2 && r.combined >= -2)
  const byFamily = Object.fromEntries(result.map(r => [r.family, r]))
  assert.equal(byFamily.c.textScore, null); assert.equal(byFamily.c.combined, 1.4)                 // undrawable: 0.7 twice
  assert.ok(byFamily.b.textScore > .9 && byFamily.b.textScore > byFamily.a.textScore); assert.equal(byFamily.b.combined, .82 + byFamily.b.textScore)
  assert.ok(byFamily.d.combined > byFamily.c.combined)
  assert.equal(byFamily.b.score, .82)  // the catalog score itself is kept
  // Every drawable face is drawn at each reference size with the trimmed text; drawing stops at the first null.
  assert.deepEqual(calls.filter(c => c[0] === 'b').map(c => c[2]), SIZES); assert.ok(calls.every(c => c[1] === 'Quiet rivers'))
  assert.equal(calls.filter(c => c[0] === 'c').length, 1)
})

test('rows past the shortlist keep their place after it', async () => {
  const many = Array.from({ length: 35 }, (_, i) => ({ family: `f${i}`, score: 1 - i / 100, gap: i % 2 === 1 }))
  const result = await rerank(many, 'rivers', { embedding: Float32Array.from({ length: 16 }, (_, i) => i === 0 ? 1 : 0), encoder, draw, limit: 30 })
  assert.equal(result.length, 35); assert.deepEqual(result.slice(30), many.slice(30)); assert.ok(result.slice(0, 30).every(r => 'combined' in r))
  assert.ok(result.slice(30).every(r => !('combined' in r)))
})
