import { test } from 'node:test'
import assert from 'node:assert/strict'
import { prepareReadings } from '../scripts/photo-prepare.mjs'

test('an RGB picture is read as the page reads a crop: coloured letters on a coloured wall give the line as it stands and cleaned, each with windows; a two-tone one only as it stands', () => {
  const w = 120, h = 40, rgb = paint => { const b = Buffer.alloc(w * h * 3); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) b.set(paint(x, y), 3 * (y * w + x)); return { width: w, height: h, pixels: b.toString('base64') } }
  const bars = (x, y) => y >= 10 && y < 30 && x >= 10 && x < 110 && x % 12 < 7
  let seed = 3; const noise = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31
  const wall = prepareReadings([rgb((x, y) => bars(x, y) ? [20, 20, 20] : [100 + 150 * noise(), 120, 140])])[0]
  assert.deepEqual(wall.map(r => [r.kind, r.line]), [['as it is', 0], ['cleaned', 0]])
  for (const r of wall) { assert.ok(r.windows.length); for (const win of r.windows) assert.equal(Buffer.from(win.pixels, 'base64').length, win.width * win.height) }
  const plain = prepareReadings([rgb((x, y) => bars(x, y) ? [0, 0, 0] : [255, 255, 255])])[0]
  assert.deepEqual(plain.map(r => r.kind), ['as it is'])
  assert.throws(() => prepareReadings([{ ...rgb(() => [0, 0, 0]), width: w - 1 }]), 'trailing pixels')
  assert.throws(() => prepareReadings([{ width: w, height: h, pixels: '!' }]))
  assert.deepEqual(prepareReadings([]), [])
})
