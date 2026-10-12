import { prepare } from './prepare.mjs'

// How a crop reaches the preparation: cut into its lines when it holds several, each line read as it stands and, unless it
// is already two-tone, with its ink found by colour, not by brightness alone, and drawn black on white with soft edges,
// and a line set steeply on a slant read turned level as well. A photographed or designed crop then reaches the model as the clean renders it learned
// from: coloured letters on a coloured wall, a grey wall under uneven light, a texture behind the letters (bench/photos.md,
// Cleaning).

// Ink axis: two means in RGB, seeded with the border's median colour (the background) and the pixel farthest from it,
// so coloured letters on a coloured wall separate by hue, not only by brightness; a pixel's ink is its position between
// the two along their axis. Transparent pixels stand on `matte`, as the page shows them. The means are found on at most a
// quarter of a million pixels spread over the crop, so a large crop costs no more passes than a small one.
export function inkMap(image, matte = 255) {
  const { width: w, height: h, data } = image, n = w * h, rgb = new Float32Array(n * 3), ring = []
  for (let i = 0; i < n; i++) {
    const a = data[4 * i + 3] / 255
    for (let c = 0; c < 3; c++) rgb[3 * i + c] = a * data[4 * i + c] + (1 - a) * matte
    const x = i % w, y = (i - x) / w
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) ring.push(i)
  }
  const median = c => { const v = ring.map(i => rgb[3 * i + c]).sort((a, b) => a - b); return v[v.length >> 1] }
  const step = Math.max(1, Math.floor(n / 250000))
  let back = [median(0), median(1), median(2)], ink = back, far = -1
  for (let i = 0; i < n; i += step) { const d = dist(rgb, i, back); if (d > far) { far = d; ink = [rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]] } }
  for (let pass = 0; pass < 6; pass++) {
    const sums = [[0, 0, 0, 0], [0, 0, 0, 0]]
    for (let i = 0; i < n; i += step) { const s = sums[dist(rgb, i, ink) < dist(rgb, i, back) ? 1 : 0]; s[0] += rgb[3 * i]; s[1] += rgb[3 * i + 1]; s[2] += rgb[3 * i + 2]; s[3]++ }
    if (!sums[0][3] || !sums[1][3]) break
    back = sums[0].slice(0, 3).map(v => v / sums[0][3]); ink = sums[1].slice(0, 3).map(v => v / sums[1][3])
  }
  const axis = ink.map((v, c) => v - back[c]), length = axis.reduce((s, v) => s + v * v, 0) || 1, map = new Float32Array(n)
  for (let i = 0; i < n; i++) map[i] = Math.max(0, Math.min(1, ((rgb[3 * i] - back[0]) * axis[0] + (rgb[3 * i + 1] - back[1]) * axis[1] + (rgb[3 * i + 2] - back[2]) * axis[2]) / length))
  return { map, width: w, height: h }
}
const dist = (rgb, i, p) => (rgb[3 * i] - p[0]) ** 2 + (rgb[3 * i + 1] - p[1]) ** 2 + (rgb[3 * i + 2] - p[2]) ** 2

// Otsu's threshold on the ink map, in 256 levels.
export function otsu(map) {
  const counts = new Uint32Array(256), n = map.length
  for (let i = 0; i < n; i++) counts[Math.min(255, Math.round(map[i] * 255))]++
  let best = -1, first = 128, last = 128, below = 0, sumBelow = 0, sum = 0
  for (let v = 0; v < 256; v++) sum += v * counts[v]
  for (let v = 0; v < 256; v++) {
    below += counts[v]; sumBelow += v * counts[v]
    if (!below || below === n) continue
    const between = below * (n - below) * (sumBelow / below - (sum - sumBelow) / (n - below)) ** 2
    if (between > best * (1 + 1e-9)) { best = between; first = last = v } else if (between >= best * (1 - 1e-9)) last = v
  }
  // Every level of an empty gap parts the two equally well: the middle of the gap, as far from both as it can be.
  return (first + last) / 2 / 255
}

// Ink above the threshold as black on white, a 3×3 majority vote removing lone pixels when `despeckle`. With `soft`, the
// ink within that much of the threshold keeps its grey, so edges stay anti-aliased as a render's are.
export function binary({ map, width: w, height: h }, threshold, despeckle = false, soft = 0) {
  const ink = new Uint8Array(map.length), level = new Float32Array(map.length)
  for (let i = 0; i < ink.length; i++) { level[i] = soft ? Math.max(0, Math.min(1, (map[i] - threshold + soft) / (2 * soft))) : map[i] > threshold ? 1 : 0; ink[i] = level[i] > .5 ? 1 : 0 }
  if (despeckle) {  // outside the picture counts as background
    const voted = new Uint8Array(ink.length)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let n = 0
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const yy = y + dy, xx = x + dx; if (yy >= 0 && yy < h && xx >= 0 && xx < w) n += ink[yy * w + xx] }
      voted[y * w + x] = n >= 5 ? 1 : 0
    }
    ink.set(voted)
  }
  const out = { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }  // ImageData's shape, in a page, a worker or Node
  for (let i = 0; i < ink.length; i++) { const v = Math.round(255 * (1 - (soft ? (ink[i] || !despeckle ? level[i] : 0) : ink[i]))); out.data[4 * i] = out.data[4 * i + 1] = out.data[4 * i + 2] = v; out.data[4 * i + 3] = 255 }
  return out
}

// A crop is already two-tone, a render or a screenshot, when the pixels the ink map leaves between ink and background are
// edges: each with an ink and a background pixel among its eight neighbours, the anti-aliasing a renderer leaves. At most
// this share of the crop may stand between ink and background away from any edge (clean renders leave under 0.1%,
// photographs mostly 2 to 35%). A share of all pixels between would grow as text shrinks: small screen text is all edge.
export const TEXTURE = .01

// The shipped cleaner: a crop already two-tone as it is, any other through the colour axis and Otsu's threshold, the ink
// within 0.15 of it keeping its grey; black on white.
export function clean(image, matte = 255) {
  const map = inkMap(image, matte), { width: w, height: h } = map; let interior = 0
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = map.map[y * w + x]; if (!(v > .15 && v < .85)) continue
    let low = false, high = false
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx; if (yy < 0 || xx < 0 || yy >= h || xx >= w) continue
      const u = map.map[yy * w + xx]; if (u <= .15) low = true; else if (u >= .85) high = true
    }
    if (!(low && high)) interior++
  }
  return interior <= w * h * TEXTURE ? image : inkOnWhite(image, matte, map)
}

// Any crop through the colour axis and Otsu's threshold, black on white, ungated.
export function inkOnWhite(image, matte = 255, map = inkMap(image, matte)) {
  return binary(map, otsu(map.map), false, .15)
}

// Running maximum or minimum over a window, one pass per axis (a monotonic deque: linear in the pixels).
function filter(map, w, h, radius, max) {
  const better = max ? (a, b) => a >= b : (a, b) => a <= b, out = new Float32Array(map), line = new Float32Array(Math.max(w, h)), deque = new Int32Array(Math.max(w, h))
  const run = (length, get, set) => {
    for (let i = 0; i < length; i++) line[i] = get(i)
    let head = 0, tail = 0
    for (let i = 0; i < length + radius; i++) {
      if (i < length) { while (tail > head && better(line[i], line[deque[tail - 1]])) tail--; deque[tail++] = i }
      const at = i - radius
      if (at < 0) continue
      while (deque[head] < at - radius) head++
      set(at, line[deque[head]])
    }
  }
  for (let y = 0; y < h; y++) run(w, x => out[y * w + x], (x, v) => { out[y * w + x] = v })
  for (let x = 0; x < w; x++) run(h, y => out[y * w + x], (y, v) => { out[y * w + x] = v })
  return out
}

// The ink with its background flattened: what an opening (the letters erased, the background kept) leaves under each pixel is taken away.
export function flatten({ map, width: w, height: h }, radius = Math.max(6, Math.round(Math.min(w, h) * .15))) {
  const background = filter(filter(map, w, h, radius, false), w, h, radius, true), out = new Float32Array(map.length)
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, map[i] - background[i])
  return { map: out, width: w, height: h }
}

// Text lines of an ink map: runs of rows whose ink reaches a tenth of the busiest row's; a run under two fifths of its
// neighbour's height (a hanging descender, the dots and accents above a line) joins it when the gap between them is under
// a fifth of the taller's height, two runs each tall enough to be a line never join, however tight the leading; each is
// then split where a gap of columns wider than its height parts it (two columns of words side by side). Each as
// { top, bottom, left, right } in pixels, rows trimmed to the part's own ink, tallest first. The crop's own line finder,
// before any model.
export function bands({ map, width: w, height: h }, threshold = .1) {
  const rows = new Float32Array(h)
  for (let y = 0; y < h; y++) { let s = 0; for (let x = 0; x < w; x++) s += map[y * w + x]; rows[y] = s }
  let peak = 0; for (const v of rows) peak = Math.max(peak, v)
  if (!peak) return []
  const unbroken = []
  for (let y = 0; y < h; y++) {
    if (rows[y] < peak * threshold) continue
    const last = unbroken.at(-1)
    if (last && y === last.bottom + 1) last.bottom = y; else unbroken.push({ top: y, bottom: y })
  }
  // A run's measure is its tallest unbroken stretch, so a descender that joins a line never makes the line taller and lets it
  // swallow the next one.
  const runs = []
  for (const { top, bottom } of unbroken) {
    const last = runs.at(-1), tall = bottom - top + 1, taller = last ? Math.max(last.tall, tall) : 0
    if (last && top - last.bottom - 1 <= taller * .2 && Math.min(last.tall, tall) < taller * .4) { last.bottom = bottom; last.tall = taller } else runs.push({ top, bottom, tall })
  }
  const parts = []
  for (const run of runs) {
    const tall = run.bottom - run.top + 1, inked = new Uint8Array(w)
    for (let y = run.top; y <= run.bottom; y++) for (let x = 0; x < w; x++) if (map[y * w + x] > .5) inked[x] = 1
    let left = -1, gap = 0
    const close = right => {
      let top = run.bottom, bottom = run.top
      for (let y = run.top; y <= run.bottom; y++) for (let x = left; x <= right; x++) if (map[y * w + x] > .5) { top = Math.min(top, y); bottom = Math.max(bottom, y); break }
      if (bottom >= top) parts.push({ top, bottom, left, right })
    }
    for (let x = 0; x <= w; x++) {
      if (x < w && inked[x]) { if (left < 0) left = x; gap = 0; continue }
      if (left < 0) continue
      gap++
      if (gap > tall || x === w) { close(x - gap); left = -1; gap = 0 }
    }
  }
  return parts.sort((a, b) => (b.bottom - b.top) - (a.bottom - a.top))
}

// The text lines of a crop that holds several, as rectangles to cut (a quarter of a line's height around each, at most
// six, the tallest first, none under two fifths of its height); [] for a crop of one line, which is read whole. Found on the cleaned ink with its background
// flattened, from a copy of at most a million and a half pixels.
export function lines(image, matte = 255) {
  const { width: w, height: h } = image, step = Math.max(1, Math.ceil(Math.sqrt(w * h / 1.5e6))), sw = Math.ceil(w / step), sh = Math.ceil(h / step)
  const small = step === 1 ? image : { width: sw, height: sh, data: new Uint8ClampedArray(sw * sh * 4) }
  if (step > 1) for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) small.data.set(image.data.subarray(4 * (y * step * w + x * step), 4 * (y * step * w + x * step) + 4), 4 * (y * sw + x))
  const map = flatten(inkMap(small, matte)), t = otsu(map.map), ink = new Float32Array(map.map.length)
  for (let i = 0; i < ink.length; i++) ink[i] = map.map[i] > t ? 1 : 0
  // The edges of lines the crop cuts through, shorter than two fifths of the tallest, are no lines of their own.
  const all = bands({ map: ink, width: sw, height: sh }), tallest = all.length ? all[0].bottom - all[0].top + 1 : 0
  const found = all.filter(b => b.bottom - b.top + 1 >= tallest * .4 && (b.bottom - b.top + 1) * step >= 8 && (b.right - b.left + 1) * step >= 16).slice(0, 6)
  if (found.length < 2) return []
  return found.map(b => {
    const pad = Math.round((b.bottom - b.top + 1) * .25), x = Math.max(0, (b.left - pad) * step), y = Math.max(0, (b.top - pad) * step)
    return { x, y, width: Math.min(w, (b.right + 1 + pad) * step) - x, height: Math.min(h, (b.bottom + 1 + pad) * step) - y }
  })
}

// A rectangle of an image, as its own image.
export function cut(image, { x, y, width, height }) {
  const data = new Uint8ClampedArray(width * height * 4)
  for (let row = 0; row < height; row++) data.set(image.data.subarray(4 * ((y + row) * image.width + x), 4 * ((y + row) * image.width + x + width)), 4 * row * width)
  return { width, height, data }
}

// How a crop is read: each of its lines (the crop itself when it holds one) as it stands; a line that is not two-tone,
// cleaned as well; and a line set steeply on a slant, turned level as well, cleaned when it was. Each reading is `crop`
// (the line as it is), `image` (what the preparation gets) and `kind` ('as it is', 'cleaned' or 'turned'). A caller
// prepares each image, matches each and keeps each family's best (bestOf): a cleaned photograph gains, and a screenshot
// with a picture behind its letters, which cleaning can hurt, keeps its own reading.
export function readings(image, matte = 255) {
  const parts = lines(image, matte), crops = parts.length ? parts.map(rect => ({ rect, crop: cut(image, rect) })) : [{ rect: { x: 0, y: 0, width: image.width, height: image.height }, crop: image }]
  return crops.flatMap(({ rect, crop }) => {
    const cleaned = clean(crop, matte), angle = steepAngle(cleaned, { background: matte })
    return [{ rect, crop, image: crop, kind: 'as it is' }, ...(cleaned !== crop ? [{ rect, crop, image: cleaned, kind: 'cleaned' }] : []),
      ...(angle ? [{ rect, crop, image: turn(cleaned, angle, matte), angle, kind: 'turned' }] : [])]
  })
}

// Which of a crop's readings stands for it, for the model's heads and the same-text stage, which compares it with clean
// renders: its first line cleaned, when that line was, else as it stands. An index into `list`.
export function principal(list) {
  const cleaned = list.findIndex(r => r.kind === 'cleaned' && r.rect === list[0]?.rect)
  return cleaned < 0 ? 0 : cleaned
}

// The image turned by `degrees` the way the line deskew turns a line level, on a canvas that holds it all, the corners
// filled with the border's median colour (white for a cleaned crop; `matte` when the border is transparent).
export function turn(image, degrees, matte = 255) {
  const { width: w, height: h, data } = image, a = degrees * Math.PI / 180, c = Math.cos(a), s = Math.sin(a)
  const W = Math.ceil(w * Math.abs(c) + h * Math.abs(s) - 1e-9), H = Math.ceil(h * Math.abs(c) + w * Math.abs(s) - 1e-9)  // a quarter turn is exact
  const ring = [[], [], [], []]
  for (let x = 0; x < w; x++) for (const y of [0, h - 1]) for (let k = 0; k < 4; k++) ring[k].push(data[4 * (y * w + x) + k])
  for (let y = 0; y < h; y++) for (const x of [0, w - 1]) for (let k = 0; k < 4; k++) ring[k].push(data[4 * (y * w + x) + k])
  const back = ring.map(v => v.sort((p, q) => p - q)[v.length >> 1]); if (back[3] === 0) back.splice(0, 4, matte, matte, matte, 255)
  const out = { width: W, height: H, data: new Uint8ClampedArray(W * H * 4) }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const sx = c * (x + .5 - W / 2) - s * (y + .5 - H / 2) + w / 2 - .5, sy = s * (x + .5 - W / 2) + c * (y + .5 - H / 2) + h / 2 - .5
    const ix = Math.floor(sx), iy = Math.floor(sy), fx = sx - ix, fy = sy - iy, o = 4 * (y * W + x)
    for (let k = 0; k < 4; k++) {
      let v = 0
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const xx = ix + dx, yy = iy + dy
        v += (xx < 0 || yy < 0 || xx >= w || yy >= h ? back[k] : data[4 * (yy * w + xx) + k]) * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy)
      }
      out.data[o + k] = Math.round(v)
    }
  }
  return out
}

const TURN = 45  // the steepest line read turned level

// The angle a line turned past the preparation's deskew (8 to 45 degrees) stands at, or 0: the principal axis of its ink,
// when the ink spreads along it at least twice as far as across, refined by the line deskew's own profile search within
// three degrees of it, taken when it gathers the ink into rows at least an eighth better than level. Not a step of the
// preparation, which the model pins: a calligraphic line can lie diagonally while level, so the crop is read both ways.
export function steepAngle(image, { background = 255 } = {}) {
  const found = prepare(image, { width: 4, height: 4, padding: 0, background })
  if (found.status !== 'ok' || found.rect.height < 8) return 0
  const border = [], value = i => { const p = i * 4, a = image.data[p + 3] / 255; return a * (image.data[p] * .2126 + image.data[p + 1] * .7152 + image.data[p + 2] * .0722) + (1 - a) * background }
  for (let x = 0; x < image.width; x++) border.push(value(x), value((image.height - 1) * image.width + x))
  for (let y = 0; y < image.height; y++) border.push(value(y * image.width), value(y * image.width + image.width - 1))
  border.sort((a, b) => a - b)
  const bg = border[border.length >> 1], threshold = Math.max(.02, found.contrast * .08) * 255
  const ink = v => found.polarity === 'light' ? v - bg > threshold : bg - v > threshold
  const points = [], step = Math.max(1, Math.ceil(found.rect.width * found.rect.height / 16000))
  let n = 0, mx = 0, my = 0
  for (let y = found.rect.y; y < found.rect.y + found.rect.height; y++) for (let x = found.rect.x; x < found.rect.x + found.rect.width; x++) {
    if (ink(value(y * image.width + x)) && n++ % step === 0) { points.push([x - image.width / 2, y - image.height / 2]); mx += x - image.width / 2; my += y - image.height / 2 }
  }
  if (points.length < 32) return 0
  mx /= points.length; my /= points.length
  let xx = 0, yy = 0, xy = 0
  for (const [x, y] of points) { xx += (x - mx) ** 2; yy += (y - my) ** 2; xy += (x - mx) * (y - my) }
  const axis = Math.atan2(2 * xy, xx - yy) / 2 * 180 / Math.PI, spread = Math.hypot(xx - yy, 2 * xy), long = (xx + yy + spread) / 2, short = (xx + yy - spread) / 2
  if (Math.abs(axis) <= 8 || Math.abs(axis) > TURN || long <= short * 4) return 0
  const bins = new Uint32Array(Math.ceil(Math.hypot(image.width, image.height)) + 4)
  const score = degrees => {
    bins.fill(0)
    const a = degrees * Math.PI / 180, c = Math.cos(a), s = Math.sin(a)
    for (const [x, y] of points) bins[Math.round(y * c - x * s + bins.length / 2)]++
    return bins.reduce((total, count) => total + count * count, 0)
  }
  const initial = score(0); let best = initial, angle = 0
  for (let a = Math.round(axis * 2) / 2 - 3; a <= Math.round(axis * 2) / 2 + 3; a += .5) {
    if (Math.abs(a) > TURN || Math.abs(a) <= 8) continue
    const v = score(a)
    if (v > best) { best = v; angle = a }
  }
  return best < initial * 1.12 ? 0 : angle
}
