import { test } from 'node:test'
import assert from 'node:assert/strict'
import { inkMap, luminanceMap, flatten, otsu, binary, bands, isolate, CLEANERS } from '../scripts/clean-candidates.mjs'

// A w×h picture: `paint(x, y)` gives [r, g, b].
const picture = (w, h, paint) => { const data = new Uint8ClampedArray(w * h * 4); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set([...paint(x, y), 255], 4 * (y * w + x)); return { width: w, height: h, data } }
const letter = (x, y) => x >= 8 && x < 12 && y >= 4 && y < 20  // a stem four pixels wide

test('the ink axis separates a red word from a blue wall that luminance cannot, and reads from the background to the ink', () => {
  const red = [200, 40, 40], blue = [40, 40, 210]  // the same luminance within a few levels
  const image = picture(24, 24, (x, y) => letter(x, y) ? red : blue), { map } = inkMap(image)
  assert.ok(map[10 * 24 + 10] > .95 && map[2 * 24 + 2] < .05)
  const lum = luminanceMap(image).map
  assert.ok(Math.abs(lum[10 * 24 + 10] - lum[2 * 24 + 2]) < .1)
})

test('luminance ink takes the side with more of it: dark letters on light, or light letters on dark', () => {
  const dark = picture(24, 24, (x, y) => letter(x, y) ? [0, 0, 0] : [255, 255, 255]), light = picture(24, 24, (x, y) => letter(x, y) ? [255, 255, 255] : [0, 0, 0])
  for (const image of [dark, light]) { const { map } = luminanceMap(image); assert.ok(map[10 * 24 + 10] > .95 && map[2 * 24 + 2] < .05) }
})

test('flattening takes a gradient out from under the letters and leaves them', () => {
  const image = picture(40, 24, (x, y) => letter(x, y) ? [0, 0, 0] : [255 - 4 * x, 255 - 4 * x, 255 - 4 * x])  // a wall darkening to the right
  const raw = luminanceMap(image), flat = flatten(raw, 6)
  assert.ok(raw.map[12 * 40 + 30] > .1, 'the dark side of the wall reads as ink before')
  assert.ok(flat.map[12 * 40 + 30] < .02 && flat.map[10 * 40 + 10] > .6, 'after flattening only the stem is ink')
})

test('Otsu finds the gap between two populations, and the binary picture is the ink black on white, lone pixels voted out', () => {
  const map = Float32Array.from({ length: 100 }, (_, i) => i < 70 ? .1 + (i % 7) / 100 : .8 + (i % 5) / 100)
  const t = otsu(map); assert.ok(map.every((v, i) => i < 70 ? v <= t : v > t), `threshold ${t} parts the two`)
  const speckled = Float32Array.from({ length: 100 }, (_, i) => (i % 10 >= 3 && i % 10 < 7 && i >= 20 && i < 80) || i === 1 ? 1 : 0)
  const image = binary({ map: speckled, width: 10, height: 10 }, .5, true)
  assert.equal(image.data[4 * 1], 255, 'the lone pixel is gone'); assert.equal(image.data[4 * 45], 0, 'the block stays'); assert.equal(image.data[4 * 0], 255)
})

test('every cleaner returns the crop as an image of its size', () => {
  const image = picture(24, 24, (x, y) => letter(x, y) ? [200, 40, 40] : [40, 40, 210])
  for (const [name, clean] of Object.entries(CLEANERS)) { const out = clean(image); assert.deepEqual([out.width, out.height, out.data.length], [24, 24, 24 * 24 * 4], name); if (!['isolated', 'gated'].includes(name)) assert.equal(out.data[4 * (10 * 24 + 10)], 0, name) }  // isolating keeps the crop's own colours; a two-tone crop passes the gate as it is
})

test('bands are the rows with ink, tallest first, a descender\'s gap joined to its line and the far-apart lines kept apart', () => {
  const w = 40, h = 60, map = new Float32Array(w * h)
  const paint = (top, bottom, left, right) => { for (let y = top; y <= bottom; y++) for (let x = left; x <= right; x++) map[y * w + x] = 1 }
  paint(4, 15, 2, 30); paint(17, 18, 10, 12)  // a tall line and a descender two rows under it
  paint(30, 35, 5, 20)                         // a short line far below
  const found = bands({ map, width: w, height: h })
  assert.deepEqual(found, [{ top: 4, bottom: 18, left: 2, right: 30 }, { top: 30, bottom: 35, left: 5, right: 20 }])
  const columns = new Float32Array(w * h); for (let y = 40; y <= 49; y++) for (let x = 0; x <= 39; x++) if (x <= 8 || x >= 25) columns[y * w + x] = 1
  assert.deepEqual(bands({ map: columns, width: w, height: h }), [{ top: 40, bottom: 49, left: 0, right: 8 }, { top: 40, bottom: 49, left: 25, right: 39 }])  // a gap of 16 columns parts a 10-row line
  const words = new Float32Array(w * h); for (let y = 40; y <= 49; y++) for (let x = 0; x <= 39; x++) if (x <= 8 || x >= 15) words[y * w + x] = 1
  assert.deepEqual(bands({ map: words, width: w, height: h }), [{ top: 40, bottom: 49, left: 0, right: 39 }])  // a word space does not
  assert.deepEqual(bands({ map: new Float32Array(w * h), width: w, height: h }), [])
})

test('isolating keeps the tallest line and paints the slivers of the lines above and below the background; a one-line crop is untouched', () => {
  const w = 60, h = 50, white = [255, 255, 255]
  const crop = picture(w, h, (x, y) => (y >= 15 && y < 35 && x >= 5 && x < 55 && x % 6 < 4) || (y < 3 && x >= 10 && x < 40) || (y >= 47 && x >= 20 && x < 30) ? [0, 0, 0] : white)
  const kept = isolate(crop), at = (x, y) => kept.data[4 * (y * w + x)]
  assert.equal(at(12, 1), 255, 'the sliver above is gone'); assert.equal(at(22, 48), 255, 'the sliver below is gone'); assert.equal(at(6, 20), 0, 'the line stays')
  const one = picture(w, h, (x, y) => y >= 15 && y < 35 && x >= 5 && x < 55 && x % 6 < 4 ? [0, 0, 0] : white)
  assert.equal(isolate(one), one)
})
