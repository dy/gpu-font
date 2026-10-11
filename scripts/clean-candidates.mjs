// Candidate cleaners for photographed and designed text, run in the page before the pinned preparation to measure what
// each would buy with the model unchanged (scripts/photo-controls.mjs): the picture's own ink colour instead of
// luminance, the background flattened under the letters, and Otsu's threshold. Each returns the crop black on white, in
// ImageData's shape. The shipped one lives in src/clean.mjs.
import { inkMap, otsu, binary, clean, flatten, bands } from '../src/clean.mjs'
export { inkMap, otsu, binary, flatten, bands }

// Luminance as the pinned preparation reads it, as ink: distance from the border's median on the darker or lighter side, whichever holds more.
export function luminanceMap(image) {
  const { width: w, height: h, data } = image, n = w * h, map = new Float32Array(n), ring = []
  for (let i = 0; i < n; i++) {
    const a = data[4 * i + 3] / 255; map[i] = (a * (data[4 * i] * .2126 + data[4 * i + 1] * .7152 + data[4 * i + 2] * .0722) + (1 - a) * 255) / 255
    const x = i % w, y = (i - x) / w
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) ring.push(map[i])
  }
  ring.sort((a, b) => a - b); const back = ring[ring.length >> 1]
  let dark = 0, light = 0
  for (let i = 0; i < n; i++) { dark += Math.max(0, back - map[i]); light += Math.max(0, map[i] - back) }
  for (let i = 0; i < n; i++) map[i] = Math.max(0, dark >= light ? back - map[i] : map[i] - back)
  return { map, width: w, height: h }
}

// The crop's main line alone: its bands found on the ink (luminance, or the colour axis), the tallest kept with a tenth of
// its height around it, every other pixel painted the border's median colour, so the edges of the lines above and below
// and a second line no longer reach the preparation. A crop of one band is returned as it is.
export function isolate(image, map = luminanceMap(image)) {
  const t = otsu(map.map), ink = new Float32Array(map.map.length)
  for (let i = 0; i < ink.length; i++) ink[i] = map.map[i] > t ? 1 : 0
  const found = bands({ map: ink, width: image.width, height: image.height })
  if (found.length < 2) return image
  const { width: w, height: h, data } = image, b = found[0], pad = Math.ceil((b.bottom - b.top + 1) * .1)
  const x0 = b.left - pad, x1 = b.right + pad, y0 = b.top - pad, y1 = b.bottom + pad, ring = [[], [], [], []]
  for (let x = 0; x < w; x++) for (const y of [0, h - 1]) for (let c = 0; c < 4; c++) ring[c].push(data[4 * (y * w + x) + c])
  for (let y = 0; y < h; y++) for (const x of [0, w - 1]) for (let c = 0; c < 4; c++) ring[c].push(data[4 * (y * w + x) + c])
  const back = ring.map(v => v.sort((a, b) => a - b)[v.length >> 1]), out = { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = 4 * (y * w + x), inside = x >= x0 && x <= x1 && y >= y0 && y <= y1
    for (let c = 0; c < 4; c++) out.data[i + c] = inside ? data[i + c] : back[c]
  }
  return out
}

// The named cleaners, each an ImageData → ImageData.
export const CLEANERS = {
  binarized: image => { const m = luminanceMap(image); return binary(m, otsu(m.map)) },
  flattened: image => { const m = flatten(luminanceMap(image)); return binary(m, otsu(m.map)) },
  colour: image => { const m = inkMap(image); return binary(m, otsu(m.map)) },
  cleaned: image => { const m = flatten(inkMap(image)); return binary(m, otsu(m.map), true) },
  'colour flattened': image => { const m = flatten(inkMap(image)); return binary(m, otsu(m.map)) },
  'colour soft': image => { const m = inkMap(image); return binary(m, otsu(m.map), false, .15) },
  'colour flattened soft': image => { const m = flatten(inkMap(image)); return binary(m, otsu(m.map), false, .15) },
  isolated: image => isolate(image),
  gated: image => clean(image),  // the shipped cleaner: a two-tone crop left as it is
  'isolated colour soft': image => { const kept = isolate(image, inkMap(image)), m = inkMap(kept); return binary(m, otsu(m.map), false, .15) }
}
