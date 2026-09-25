import test from 'node:test'
import assert from 'node:assert/strict'
import { prepare } from '../src/prepare.mjs'

function fixture(rows, invert = false, alpha = 255) {
  const width = rows[0].length, height = rows.length
  const data = new Uint8ClampedArray(width * height * 4)
  rows.forEach((row, y) => [...row].forEach((c, x) => {
    const value = c === '#' ? 0 : c === '+' ? 128 : 255
    const p = (y * width + x) * 4
    data.fill(invert ? 255 - value : value, p, p + 3); data[p + 3] = alpha
  }))
  return { width, height, data }
}
const a = fixture(['.......', '..#....', '.......', '..#....', '..#....', '..##...', '.......'])
const b = fixture(['.......', '.#####.', '.#...#.', '.#####.', '.......'])

test('single-pixel blank and transparent crops take the explicit non-match path', () => {
  for (const img of [fixture(['.']), fixture(['#']), fixture(['#'], false, 0)]) {
    const result = prepare(img)
    assert.equal(result.status, 'blank'); assert.equal(result.rect, null)
    assert.ok(result.pixels.every(v => v === 1))
  }
})
test('rejects empty/null, mismatched buffers, malformed dimensions and out-of-bounds crops', () => {
  for (const img of [null, undefined, {}, { ...a, width: 0 }, { ...a, width: 1.5 }, { ...a, height: NaN }, { ...a, data: [] }, { ...a, data: a.data.slice(1) }, { ...a, width: 20_000, height: 20_000 }]) assert.throws(() => prepare(img))
  for (const rect of [{ x: -1, y: 0, width: 1, height: 1 }, { x: 0, y: 0, width: 0, height: 1 }, { x: 7, y: 0, width: 1, height: 1 }, { x: 0, y: 0, width: 7, height: 8 }]) assert.throws(() => prepare(a, { rect }))
  for (const options of [null, { width: 0 }, { padding: 16 }, { background: NaN }, { polarity: 'auto' }]) assert.throws(() => prepare(a, options))
})
test('preserves a detached dot, reports source coordinates, and uses uniform scale', () => {
  const result = prepare(a, { width: 28, height: 28 })
  assert.equal(result.status, 'ok')
  assert.deepEqual(result.rect, { x: 1, y: 0, width: 4, height: 7 })
  assert.equal(result.scale, 24 / 7)
  // Dot, gap, stem survive in separate rows.
  const inkRows = Array.from({ length: 28 }, (_, y) => result.pixels.slice(y * 28, (y + 1) * 28).some(v => v < .5))
  assert.ok(inkRows[7]); assert.ok(!inkRows[10]); assert.ok(inkRows[15])
  assert.ok(result.pixels.every(v => Number.isFinite(v) && v >= 0 && v <= 1))
})
test('opposite polarity produces the same normalized geometry and grayscale edges', () => {
  const dark = fixture(['.....', '.+#+.', '.###.', '.....'])
  const light = fixture(['.....', '.+#+.', '.###.', '.....'], true)
  const p = prepare(dark), q = prepare(light)
  assert.equal(p.polarity, 'dark'); assert.equal(q.polarity, 'light')
  assert.ok(p.pixels.some(v => v > 0 && v < 1))
  assert.ok(p.pixels.every((v, i) => Math.abs(v - q.pixels[i]) < 1e-6))
})
test('dark letters on a wall that reads dark are found on the other side, as if their polarity were given', () => {
  const wall = fixture(['.......', '.#.#.#.', '.#####.', '.......'])
  for (let p = 0; p < wall.data.length; p += 4) if (wall.data[p] === 255) wall.data.fill(100, p, p + 3); else wall.data.fill(10, p, p + 3)
  const found = prepare(wall), given = prepare(wall, { polarity: 'dark' })
  assert.equal(found.status, 'ok'); assert.equal(found.polarity, 'dark')
  assert.deepEqual(found, given)
  assert.notEqual(prepare(wall, { polarity: 'light' }).status, 'ok')  // a declared polarity is never second-guessed
})
test('declared alpha background matches opaque equivalent; low contrast is explicit', () => {
  const transparent = fixture(['.....', '.###.', '.....'])
  for (let p = 0; p < transparent.data.length; p += 4) if (transparent.data[p] === 255) transparent.data[p + 3] = 0
  assert.deepEqual(prepare(transparent).pixels, prepare(fixture(['.....', '.###.', '.....'])).pixels)
  const white = fixture(['.....', '.###.', '.....'], true)
  for (let p = 0; p < white.data.length; p += 4) if (white.data[p] === 0) white.data[p + 3] = 0
  assert.deepEqual(prepare(white, { background: 0 }).pixels, prepare(transparent).pixels)
  const faint = fixture(['.....', '.###.', '.....'], false, 12)
  assert.equal(prepare(faint).status, 'low-contrast')
})
test('explicit subrect and clamped subarray preserve offsets without mutating input', () => {
  const raw = new Uint8ClampedArray(a.data.length + 12)
  raw.set(a.data, 8)
  const image = { ...a, data: raw.subarray(8, 8 + a.data.length) }
  const before = raw.slice()
  assert.deepEqual(prepare(image).pixels, prepare(a).pixels)
  assert.deepEqual(prepare(image, { rect: { x: 1, y: 0, width: 4, height: 7 } }).rect, { x: 1, y: 0, width: 4, height: 7 })
  assert.deepEqual(raw, before)
})
test('area shrinking keeps thin strokes at fractional crop boundaries', () => {
  // The entire fitted crop is less than one output pixel high and straddles
  // two rows. Neither row's center falls inside it; footprint overlap matters.
  const image = fixture(['.'.repeat(101), '.' + '#'.repeat(99) + '.', '.'.repeat(101)])
  const result = prepare(image, { width: 16, height: 8, padding: 1 })
  const actualInk = result.pixels.reduce((s, v) => s + 1 - v, 0)
  const expectedInk = 99 * result.scale ** 2
  assert.ok(Math.abs(actualInk - expectedInk) < 1e-5, `${actualInk} != ${expectedInk}`)
})
test('edge-touching glyphs stay in bounds; A → A → B → blank → A owns independent outputs', () => {
  const edge = prepare(fixture(['#....', '#....', '##...', '.....']))
  assert.equal(edge.status, 'ok'); assert.equal(edge.rect.x, 0); assert.equal(edge.rect.y, 0)
  const first = prepare(a), copy = first.pixels.slice()
  assert.deepEqual(prepare(a).pixels, copy)
  assert.notDeepEqual(prepare(b).pixels, copy)
  assert.equal(prepare(fixture(['.'])).status, 'blank')
  const last = prepare(a)
  last.pixels[0] = 0
  assert.deepEqual(first.pixels, copy)
  assert.deepEqual(prepare(a).pixels, copy)
})
test('a border along any crop edge, full or faint, on it or one row in, prepares exactly as the crop without it', () => {
  const text = ['..........', '...#......', '...#......', '...##.....', '...#.#....', '..........']
  const clean = prepare(fixture(text), { width: 32, height: 16 })
  const variants = [
    [...text, '##########'],                  // bottom
    [...text, '++++++++++'],                  // bottom, faint
    [...text, '..........', '##########'],    // one blank row in from the edge
    ['##########', '##########', ...text],     // top, two rows thick
    text.map(row => '#' + row),               // left
    text.map(row => row + '.+'),              // right, faint, one column in
  ]
  // A box one pixel inside the crop, its sides crossing the rows after its top and bottom. The text sits in a crop large
  // enough for the box's sides, short of the crop by the margin, to still ink 90% of it.
  const blank = '....................', tall = [...Array(7).fill(blank), ...text.map(row => row + '..........'), ...Array(7).fill(blank)]
  const cleanWide = prepare(fixture(tall), { width: 32, height: 16 })
  const edge = '.' + '#'.repeat(22) + '.', boxed = ['.'.repeat(24), edge, ...tall.map(row => '.#' + row + '#.'), edge, '.'.repeat(24)]
  for (const [rows, expected] of [...variants.map(rows => [rows, clean]), [boxed, cleanWide]]) {
    const lined = prepare(fixture(rows), { width: 32, height: 16 })
    assert.deepEqual([lined.status, lined.contrast, Array.from(lined.pixels)], [expected.status, expected.contrast, Array.from(expected.pixels)], rows.join('|'))
  }
})

test('letters at a tight crop edge are never taken for a border', () => {
  // A Devanagari-like headline inks the whole top row, but its stems hang from it: no background follows.
  assert.deepEqual(prepare(fixture(['#########.', '.#..#..#..', '.#..#..#..'])).rect, { x: 0, y: 0, width: 10, height: 3 })
  // The l of "hill" at a tight crop's side is as long as the crop and has background inside it, but runs no longer than the text.
  assert.deepEqual(prepare(fixture(['#....#', '#....#', '###..#', '#.#..#'])).rect, { x: 0, y: 0, width: 6, height: 4 })
  // The h of "uPcBhE" at a window's cut rises past the capitals, but at one end only.
  assert.deepEqual(prepare(fixture(['#.....', '#..##.', '#.#..#', '#.#..#', '#..##.'])).rect, { x: 0, y: 0, width: 6, height: 5 })
  // Dense text is not a line: a row 80% inked still sets the box.
  assert.equal(prepare(fixture(['..........', '.########.', '.#......#.', '..........'])).rect.y, 0)
  // A crop that holds only a line (a dash, a rule) keeps it: there is no text for it to frame.
  assert.equal(prepare(fixture(['..........', '..........', '##########'])).status, 'ok')
})
