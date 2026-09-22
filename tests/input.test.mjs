import test from 'node:test'
import assert from 'node:assert/strict'
import { prepareInput } from '../src/input.mjs'

function fixture(width, height, rects) {
  const data = new Uint8Array(width * height * 4).fill(255)
  for (const [x, y, w, h] of rects) for (let r = y; r < y + h; r++) for (let c = x; c < x + w; c++) data.fill(0, (r * width + c) * 4, (r * width + c) * 4 + 3)
  return { width, height, data }
}
function edges(input) {
  const { width: w, height: h, pixels: p } = input
  assert.ok([...p.slice(0, w)].some(v => v < 1))
  assert.ok([...p.slice((h - 1) * w)].some(v => v < 1))
  assert.ok(Array.from({ length: h }, (_, y) => p[y * w]).some(v => v < 1))
  assert.ok(Array.from({ length: h }, (_, y) => p[y * w + w - 1]).some(v => v < 1))
}

test('actual model inputs have tight bounds, adaptive dimensions, and byte-exact grayscale', () => {
  const thin = fixture(120, 80, [[56, 10, 8, 60]])
  const wide = fixture(300, 80, [[10, 10, 10, 60], [35, 10, 200, 8], [230, 10, 10, 60]])
  const a = prepareInput(thin), b = prepareInput(wide)
  assert.equal(a.windows.length, 1)
  assert.ok(a.windows[0].width < 20)
  assert.ok(b.windows.length >= 2 && b.windows.length <= 3)
  assert.ok(b.windows[0].width > a.windows[0].width)
  for (const { windows } of [a, b]) for (const input of windows) {
    assert.equal(input.pixels.length, input.width * input.height)
    assert.ok(input.width <= 128 && input.height <= 48)
    for (const v of input.pixels) assert.equal(v, Math.fround(Math.round(v * 255) / 255))
    edges(input)
  }
  const saved = a.windows[0].pixels.slice()
  assert.deepEqual(prepareInput(thin), a)
  prepareInput(wide)
  assert.deepEqual(prepareInput(thin), a)
  a.windows[0].pixels.fill(.5)
  assert.deepEqual(prepareInput(thin).windows[0].pixels, saved)
})

test('blank inputs produce no pretend model image and malformed inputs reject', () => {
  assert.deepEqual(prepareInput(fixture(1, 1, [])), { status: 'blank', windows: [] })
  const alpha = fixture(10, 10, [[0, 0, 10, 10]])
  for (let i = 3; i < alpha.data.length; i += 4) alpha.data[i] = 0
  assert.equal(prepareInput(alpha).windows.length, 0)
  for (const image of [null, {}, { width: 0, height: 1, data: new Uint8Array() }, { width: 1, height: 1, data: new Uint8Array(3) }]) assert.throws(() => prepareInput(image))
  for (const options of [{ width: 0 }, { height: 129 }, { windows: 0 }, { windows: 4 }, { windows: 1.5 }]) assert.throws(() => prepareInput(fixture(1, 1, []), options))
})

test('smallest two-tone image preserves its grayscale ramp through A → blank → A', () => {
  const image = fixture(2, 1, [[0, 0, 1, 1]])
  const result = prepareInput(image)
  assert.equal(result.status, 'ok')
  assert.equal(result.windows.length, 1)
  const input = result.windows[0], row = input.pixels.slice(0, input.width)
  assert.equal(row[0], 0)
  assert.ok(row.at(-1) > 0 && row.at(-1) < 1)
  for (let x = 1; x < row.length; x++) assert.ok(row[x] >= row[x - 1])
  for (let y = 1; y < input.height; y++) assert.deepEqual(input.pixels.slice(y * input.width, (y + 1) * input.width), row)
  edges(input)
  assert.deepEqual(prepareInput(image), result)
  assert.deepEqual(prepareInput(fixture(1, 1, [])), { status: 'blank', windows: [] })
  assert.deepEqual(prepareInput(image), result)
})

test('extra plain background changes source coordinates but not the model pixels', () => {
  const tight = fixture(92, 42, [[8, 8, 5, 25], [20, 12, 30, 6], [63, 8, 5, 25]])
  const tall = fixture(106, 160, [[15, 67, 5, 25], [27, 71, 30, 6], [70, 67, 5, 25]])
  const a = prepareInput(tight), b = prepareInput(tall)
  assert.equal(a.windows.length, b.windows.length)
  for (let i = 0; i < a.windows.length; i++) {
    assert.deepEqual(a.windows[i].pixels, b.windows[i].pixels)
    assert.equal(a.windows[i].rect.width, b.windows[i].rect.width)
    assert.equal(a.windows[i].rect.height, b.windows[i].rect.height)
    assert.equal(a.windows[i].rect.x + 7, b.windows[i].rect.x)
    assert.equal(a.windows[i].rect.y + 59, b.windows[i].rect.y)
  }
})
