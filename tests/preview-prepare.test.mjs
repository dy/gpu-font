import test from 'node:test'
import assert from 'node:assert/strict'
import { prepareRegion } from '../scripts/preview-prepare.mjs'
import { cropView } from '../scripts/preview-lib.mjs'
import { prepareLine } from '../src/line.mjs'

test('archive preparation uses only the declared region and matches browser tensor bytes for all PNG channel layouts', () => {
  const width = 48, height = 24, region = { x: 8, y: 4, width: 32, height: 16 }
  const expected = new Uint8Array(region.width * region.height * 4).fill(255)
  for (let y = 4; y < 12; y++) for (let x = 6; x < 26; x += 4) expected.fill(17, (y * 32 + x) * 4, (y * 32 + x) * 4 + 3)
  const browser = prepareLine({ width: 32, height: 16, data: expected }, { deskew: true, sampler: 'windows' })
  for (const channels of [1, 2, 3, 4]) {
    const pixels = new Uint8Array(width * height * channels).fill(0) // distractor outside region
    for (let y = 0; y < 16; y++) for (let x = 0; x < 32; x++) {
      const at = ((y + 4) * width + x + 8) * channels
      pixels.fill(expected[(y * 32 + x) * 4], at, at + channels)
      if (channels === 2 || channels === 4) pixels[at + channels - 1] = 255
    }
    const image = { width, height, channels, pixels }, snapshot = pixels.slice()
    for (let i = 0; i < 2; i++) {
      const actual = prepareRegion(image, region)
      assert.deepEqual(actual.map(w => [w.width, w.height, [...w.pixels]]), browser.windows.map(w => [w.width, w.height, [...w.pixels].map(v => Math.round(v * 255))]))
      assert.deepEqual(pixels, snapshot)
    }
    const single = cropView(image, { x: 14, y: 8, width: 1, height: 1 })
    assert.equal(single.pixels[0], 17); assert.equal(single.pixels.length, channels)
  }
})
test('reference cropping refuses empty, missing, out-of-bounds and blank regions', () => {
  const image = { width: 2, height: 2, channels: 4, pixels: new Uint8Array(16).fill(255) }, rect = { x: 0, y: 0, width: 2, height: 2 }
  for (const r of [null, {}, { ...rect, width: 0 }, { ...rect, x: -1 }, { ...rect, y: 1 }, { ...rect, width: 1.5 }]) assert.throws(() => prepareRegion(image, r), /region/)
  for (const i of [null, {}, { ...image, pixels: [] }, { ...image, channels: 5 }]) assert.throws(() => prepareRegion(i, rect), /image/)
  assert.throws(() => prepareRegion(image, rect), /blank/)
  assert.throws(() => prepareRegion(image, { x: 1, y: 1, width: 1, height: 1 }), /blank/)
})
