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

test('smallest two-tone image prepares as its pixels with a margin, through A → blank → A', () => {
  const image = fixture(2, 1, [[0, 0, 1, 1]])
  const result = prepareInput(image)
  assert.equal(result.status, 'ok')
  assert.equal(result.windows.length, 1)
  assert.deepEqual(result.windows[0].pixels, prepareInput(fixture(6, 5, [[2, 2, 1, 1]])).windows[0].pixels)
  assert.deepEqual(result.windows[0].rect, { x: 0, y: 0, width: 2, height: 1 })
  edges(result.windows[0])
  assert.deepEqual(prepareInput(image), result)
  assert.deepEqual(prepareInput(fixture(1, 1, [])), { status: 'blank', windows: [] })
  assert.deepEqual(prepareInput(image), result)
})

test('a crop that touches the text prepares as one with a margin, on any side', () => {
  const boxes = [[0, 0, 5, 25], [12, 4, 30, 6], [55, 0, 5, 25]], margin = fixture(80, 45, boxes.map(([x, y, w, h]) => [x + 10, y + 10, w, h]))
  for (const [crop, dx, dy] of [[fixture(60, 25, boxes), 0, 0], [fixture(70, 25, boxes), 0, 0], [fixture(60, 35, boxes.map(([x, y, w, h]) => [x, y + 10, w, h])), 0, 10]]) {
    const a = prepareInput(crop), b = prepareInput(margin)
    assert.deepEqual(a.windows.map(w => w.pixels), b.windows.map(w => w.pixels))
    for (const [i, { rect }] of a.windows.entries()) {
      const r = b.windows[i].rect, x = Math.max(0, r.x - 10 + dx), y = Math.max(0, r.y - 10 + dy)
      assert.deepEqual(rect, { x, y, width: Math.min(crop.width, r.x - 10 + dx + r.width) - x, height: Math.min(crop.height, r.y - 10 + dy + r.height) - y })
    }
  }
  // White text on a transparent PNG over a black matte: the frame repeats a transparent pixel, not an opaque one, which
  // here would read as ink.
  const clear = image => { for (let i = 0; i < image.data.length; i += 4) if (image.data[i] === 255) image.data[i + 3] = 0; else image.data.fill(255, i, i + 3); return image }
  const onMatte = image => prepareInput(clear(image), { background: 0 }).windows.map(w => w.pixels)
  assert.deepEqual(onMatte(fixture(60, 25, boxes)), onMatte(fixture(80, 45, boxes.map(([x, y, w, h]) => [x + 10, y + 10, w, h]))))
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
