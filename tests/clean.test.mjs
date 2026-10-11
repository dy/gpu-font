import { test } from 'node:test'
import assert from 'node:assert/strict'
import { clean, inkOnWhite, TEXTURE } from '../src/clean.mjs'

const picture = (w, h, paint) => { const data = new Uint8ClampedArray(w * h * 4); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set([...paint(x, y), 255], 4 * (y * w + x)); return { width: w, height: h, data } }

test('ink on white draws coloured letters on a coloured wall black on white, opaque and grey', () => {
  const crop = picture(30, 20, (x, y) => x >= 10 && x < 20 && y >= 4 && y < 16 ? [220, 60, 30] : [60, 70, 200])  // orange on blue
  const out = inkOnWhite(crop), at = (x, y) => out.data[4 * (y * 30 + x)]
  assert.deepEqual([out.width, out.height, out.data.length], [30, 20, 30 * 20 * 4])
  assert.equal(at(15, 10), 0); assert.equal(at(2, 2), 255)
  for (let i = 0; i < out.data.length; i += 4) { assert.equal(out.data[i], out.data[i + 1]); assert.equal(out.data[i], out.data[i + 2]); assert.equal(out.data[i + 3], 255) }
})

test('light letters on a dark wall come out dark on light, and an anti-aliased edge keeps greys between', () => {
  const ramp = x => x < 6 ? 20 : x >= 14 ? 240 : 20 + (x - 6) * 220 / 8  // eight columns from the wall to the letter
  const crop = picture(30, 20, (x, y) => { const v = y >= 4 && y < 16 && x < 24 ? ramp(x) : 20; return [v, v, v] })
  const out = inkOnWhite(crop), at = (x, y) => out.data[4 * (y * 30 + x)]
  assert.equal(at(18, 10), 0); assert.equal(at(2, 2), 255)
  const edge = [6, 7, 8, 9, 10, 11, 12, 13].map(x => at(x, 10))
  assert.ok(edge.some(v => v > 0 && v < 255), `the edge reads ${edge}`)
  assert.deepEqual(edge, [...edge].sort((a, b) => b - a), 'from white to black across the edge')
})

test('a crop of one colour has no ink to find and comes out all white', () => {
  const out = inkOnWhite(picture(8, 8, () => [90, 90, 90]))
  assert.ok(out.data.every((v, i) => i % 4 === 3 || v === 255))
})

test('transparent pixels stand on the matte: white letters on a transparent crop are found on black, lost on white', () => {
  const data = new Uint8ClampedArray(20 * 10 * 4)
  for (let y = 2; y < 8; y++) for (let x = 5; x < 15; x++) data.set([255, 255, 255, 255], 4 * (y * 20 + x))  // opaque white letters, the rest transparent
  const crop = { width: 20, height: 10, data }, at = (out, x, y) => out.data[4 * (y * 20 + x)]
  const onBlack = inkOnWhite(crop, 0); assert.equal(at(onBlack, 10, 5), 0); assert.equal(at(onBlack, 1, 1), 255)
  assert.ok(inkOnWhite(crop, 255).data.every((v, i) => i % 4 === 3 || v === 255))
})

test('a large crop is cleaned like a small one: the means sampled, every pixel mapped', () => {
  const big = picture(1200, 300, (x, y) => (x % 100) < 40 && y > 100 && y < 200 ? [200, 30, 30] : [30, 30, 200])
  const out = inkOnWhite(big), at = (x, y) => out.data[4 * (y * 1200 + x)]
  assert.equal(at(1110, 150), 0); assert.equal(at(1150, 150), 255); assert.equal(at(10, 20), 255)
})

test('a two-tone crop, a render with only its anti-aliased edges between, is returned as it is; a textured one is cleaned', () => {
  const render = picture(40, 20, (x, y) => { const v = x >= 10 && x < 30 && y >= 5 && y < 15 ? 0 : (x === 9 || x === 30) && y >= 5 && y < 15 ? 128 : 255; return [v, v, v] })
  assert.equal(clean(render), render)
  let seed = 7; const noise = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31
  const wall = picture(40, 20, (x, y) => { const v = x >= 10 && x < 30 && y >= 5 && y < 15 ? 20 : 120 + 120 * noise(); return [v, v, v] })  // grain over the whole wall
  assert.notEqual(clean(wall), wall); assert.ok(TEXTURE > 0 && TEXTURE < .05)
})

test('lines: a crop of three lines gives each, tallest first, padded and inside the crop; one line, or one with the clipped edge of another, gives none', async () => {
  const { lines, cut } = await import('../src/clean.mjs')
  const w = 200, rows = (bars) => picture(w, 120, (x, y) => bars.some(([top, bottom]) => y >= top && y < bottom && x >= 10 && x < 190 && x % 12 < 7) ? [0, 0, 0] : [255, 255, 255])
  const three = lines(rows([[10, 30], [45, 60], [80, 110]]))
  assert.equal(three.length, 3)
  assert.deepEqual(three.map(r => [r.y, r.height]), [[72, 46], [5, 30], [41, 23]])  // the 30 px line first, then the 20 and 15 px ones, a quarter of each line's height around it
  for (const r of three) assert.ok(r.x >= 0 && r.y >= 0 && r.x + r.width <= w && r.y + r.height <= 120)
  assert.deepEqual(lines(rows([[40, 80]])), [])
  assert.deepEqual(lines(rows([[0, 6], [40, 80]])), [], 'a sliver under two fifths of the line is the edge of another line')
  const piece = cut(rows([[40, 80]]), { x: 12, y: 40, width: 7, height: 2 })  // a bar starts at x 12
  assert.deepEqual([piece.width, piece.height, piece.data.length, piece.data[0]], [7, 2, 56, 0])
})

test('readings: a level two-tone line is one reading, the crop as it is; three lines are three; a steep word is read as it stands and turned level; a textured line as it stands and cleaned', async () => {
  const { readings } = await import('../src/clean.mjs')
  const bars = (w, h, rows) => picture(w, h, (x, y) => rows.some(([t, b]) => y >= t && y < b) && x >= 10 && x < w - 10 && x % 12 < 7 ? [0, 0, 0] : [255, 255, 255])
  const one = bars(200, 60, [[15, 45]]), single = readings(one)
  assert.equal(single.length, 1); assert.equal(single[0].crop, one); assert.equal(single[0].image, one); assert.equal(single[0].angle, undefined)
  assert.equal(readings(bars(200, 120, [[10, 30], [45, 60], [80, 110]])).length, 3)
  const turned = picture(260, 260, (x, y) => { const a = 25 * Math.PI / 180, u = Math.cos(a) * (x - 130) + Math.sin(a) * (y - 130), v = -Math.sin(a) * (x - 130) + Math.cos(a) * (y - 130); return Math.abs(v) < 11 && Math.abs(u) < 90 && ((u + 90) % 25) < 12 ? [0, 0, 0] : [255, 255, 255] })
  const steep = readings(turned)
  assert.deepEqual(steep.map(r => r.kind), ['as it is', 'turned']); assert.ok(Math.abs(Math.abs(steep[1].angle) - 25) <= 1, `turned by ${steep[1].angle}`)
  let seed = 5; const noise = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31
  const wall = picture(200, 60, (x, y) => y >= 15 && y < 45 && x >= 10 && x < 190 && x % 12 < 7 ? [20, 20, 20] : [100 + 150 * noise(), 120, 140])
  const both = readings(wall)
  assert.deepEqual(both.map(r => r.kind), ['as it is', 'cleaned']); assert.equal(both[0].image, wall); assert.equal(both[1].crop, wall); assert.notEqual(both[1].image, wall)  // a textured line read as it stands and cleaned
})

test('steepAngle finds a word turned 20° or 35° either way, and turned by it the word lies level for the deskew; a level line and a square block have none', async () => {
  const { steepAngle, turn } = await import('../src/clean.mjs'), { prepareLine } = await import('../src/line.mjs')
  const word = angle => picture(260, 260, (x, y) => { const a = angle * Math.PI / 180, u = Math.cos(a) * (x - 130) + Math.sin(a) * (y - 130), v = -Math.sin(a) * (x - 130) + Math.cos(a) * (y - 130); return Math.abs(v) < 11 && Math.abs(u) < 90 && ((u + 90) % 25) < 12 ? [0, 0, 0] : [255, 255, 255] })
  for (const angle of [-35, -20, 20, 35]) {
    const image = word(angle), found = steepAngle(image)
    assert.ok(Math.abs(found - angle) <= 1, `Estimated ${found} for ${angle}`)
    const level = turn(image, found)
    assert.ok(Math.abs(steepAngle(level)) === 0, 'turned, the word is level')
    assert.ok(level.width > level.height * 1.8 || prepareLine(level, { deskew: true, sampler: 'windows' }).status === 'ok')
  }
  assert.equal(steepAngle(word(0)), 0)
  assert.equal(steepAngle(picture(120, 120, (x, y) => x >= 20 && x < 100 && y >= 20 && y < 100 ? [0, 0, 0] : [255, 255, 255])), 0)
  assert.equal(steepAngle(picture(1, 1, () => [255, 255, 255])), 0)
})

test('turning keeps the picture: a quarter turn of a wide bar is a tall bar, the corners the border\'s colour', async () => {
  const { turn } = await import('../src/clean.mjs')
  const bar = picture(40, 10, (x, y) => y >= 3 && y < 7 ? [0, 0, 0] : [255, 255, 255]), quarter = turn(bar, 90)
  assert.deepEqual([quarter.width, quarter.height], [10, 40])
  assert.equal(quarter.data[4 * (20 * 10 + 5)], 0); assert.equal(quarter.data[4 * (20 * 10 + 0)], 255)
  const same = turn(bar, 0); assert.deepEqual([same.width, same.height], [40, 10]); assert.deepEqual(same.data, bar.data)
})

test('small screen text, all anti-aliased edge, is two-tone however many of its pixels stand between ink and background', () => {
  // Stems one to two pixels wide with grey on both sides: a third of the pixels are neither ink nor background, all of them edges.
  const small = picture(60, 14, (x, y) => { if (y < 3 || y > 10) return [255, 255, 255]; const k = x % 4; const v = k === 0 ? 30 : k === 1 ? 140 : k === 3 ? 160 : 255; return [v, v, v] })
  assert.equal(clean(small), small)
})

test('the principal reading is the first line cleaned when it was, else the first line as it stands', async () => {
  const { principal } = await import('../src/clean.mjs'), a = {}, b = {}
  assert.equal(principal([{ rect: a, kind: 'as it is' }, { rect: a, kind: 'cleaned' }, { rect: b, kind: 'as it is' }]), 1)
  assert.equal(principal([{ rect: a, kind: 'as it is' }, { rect: b, kind: 'as it is' }, { rect: b, kind: 'cleaned' }]), 0)  // another line's cleaning does not stand for the crop
  assert.equal(principal([{ rect: a, kind: 'as it is' }, { rect: a, kind: 'turned' }]), 0)
  assert.equal(principal([]), 0)
})
