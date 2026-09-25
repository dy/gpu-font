const MAX_PIXELS = 16_777_216

function integer(value, name, min, max) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new RangeError(`Invalid ${name}`)
  return value
}

// Ink bounds [left, top, right, bottom] off the skipped rows and columns; right < 0 when there is none.
function box(inked, w, h, skipRow, skipCol) {
  let left = w, top = h, right = -1, bottom = -1
  for (let row = 0; row < h; row++) for (let col = 0; col < w; col++) {
    if (!inked(row, col) || skipRow[row] || skipCol[col]) continue
    left = Math.min(left, col); right = Math.max(right, col)
    top = Math.min(top, row); bottom = Math.max(bottom, row)
  }
  return [left, top, right, bottom]
}

// Rows and columns that frame the text rather than belong to it, like a UI border caught in a screenshot crop. A border
// lies along an edge, past at most two blank lines, one to three lines thick, inking 90% of its length, which a line of
// text never does since letters have gaps. Background follows it, apart from other borders crossing it: a Devanagari
// headline inks a whole row too, but its stems hang from it. And it runs past the text at both ends, across the margin:
// a letter's bar or stem at a tight crop's edge reaches past the other letters at one end at most, and a line that is
// all the crop holds frames nothing.
function borders(inked, w, h) {
  const axes = [{ length: h, span: w, at: inked }, { length: w, span: h, at: (col, row) => inked(row, col) }]
  const candidates = axes.map(({ length, span, at }) => {
    const count = i => { let n = 0; for (let j = 0; j < span; j++) n += at(i, j); return n }
    const inside = i => i >= 0 && i < length
    return [[0, 1], [length - 1, -1]].map(([i, step]) => {
      for (let blank = 0; blank < 2 && inside(i) && !count(i); blank++) i += step
      const lines = []
      while (lines.length < 3 && inside(i) && count(i) >= span * .9) { lines.push(i); i += step }
      return { lines, next: i, inside }
    })
  })
  const skip = candidates.map((ends, axis) => {
    const flags = new Uint8Array(axes[axis].length)
    for (const { lines } of ends) for (const i of lines) flags[i] = 1
    return flags
  })
  const framed = axes.map(({ span, at }, axis) => candidates[axis].filter(({ lines, next, inside }) => {
    if (!lines.length || !inside(next)) return false
    let n = 0
    for (let j = 0; j < span; j++) n += !skip[1 - axis][j] && at(next, j)
    return n <= span * .02
  }))
  skip.forEach((flags, axis) => { flags.fill(0); for (const { lines } of framed[axis]) for (const i of lines) flags[i] = 1 })
  if (!framed.flat().length) return skip
  const [left, top, right, bottom] = box(inked, w, h, ...skip), text = [[left, right], [top, bottom]]
  skip.forEach((flags, axis) => flags.forEach((flag, i) => {
    if (!flag) return
    let first = -1, last = -1
    for (let j = 0; j < axes[axis].span; j++) if (axes[axis].at(i, j)) { if (first < 0) first = j; last = j }
    if (right < 0 || !(first < text[axis][0] && last > text[axis][1])) flags[i] = 0
  }))
  return skip
}

// The image with two pixels of its own background added on each side its ink box (prepare's rect) reaches, so a crop that
// touches the text prepares as one with a margin. Two, not one: the framed image's box then never reaches its edge. The added pixels repeat the ring's median pixel, the one prepare()
// takes for the background, so that estimate stays exactly the same. back() maps a rect in the framed image to the
// original. Null when no side reaches, or the framed image would exceed the pixel limit.
export function frame(image, rect, matte = 255) {
  const { width: w, height: h, data } = image
  const left = rect.x === 0 ? 2 : 0, top = rect.y === 0 ? 2 : 0, right = rect.x + rect.width === w ? 2 : 0, bottom = rect.y + rect.height === h ? 2 : 0
  const width = w + left + right, height = h + top + bottom
  if (width === w && height === h || width * height > MAX_PIXELS) return null
  const value = p => { const a = data[p * 4 + 3] / 255; return a * (data[p * 4] * .2126 + data[p * 4 + 1] * .7152 + data[p * 4 + 2] * .0722) + (1 - a) * matte }
  const ring = []
  for (let x = 0; x < w; x++) { ring.push(x); if (h > 1) ring.push((h - 1) * w + x) }
  for (let y = 1; y < h - 1; y++) { ring.push(y * w); if (w > 1) ring.push(y * w + w - 1) }
  ring.sort((a, b) => value(a) - value(b))
  const median = ring[Math.floor(ring.length / 2)] * 4, out = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < out.length; i += 4) out.set(data.subarray(median, median + 4), i)
  for (let y = 0; y < h; y++) out.set(data.subarray(y * w * 4, (y + 1) * w * 4), ((y + top) * width + left) * 4)
  const back = r => {
    const x = Math.max(0, r.x - left), y = Math.max(0, r.y - top)
    return { x, y, width: Math.min(w, r.x - left + r.width) - x, height: Math.min(h, r.y - top + r.height) - y }
  }
  return { image: { width, height, data: out }, x: left, y: top, back }
}

// Selected text line on a substantially uniform background. No text detector.
export function prepare(image, options = {}) {
  if (!image || !options || typeof options !== 'object') throw new TypeError('Expected image and options')
  const { width, height, data } = image
  integer(width, 'image width', 1, MAX_PIXELS)
  integer(height, 'image height', 1, Math.floor(MAX_PIXELS / width))
  if (!(data instanceof Uint8Array) && !(data instanceof Uint8ClampedArray)) throw new TypeError('Expected RGBA bytes')
  if (data.length !== width * height * 4) throw new RangeError('RGBA length does not match image dimensions')
  const targetWidth = integer(options.width ?? 128, 'output width', 4, 1024)
  const targetHeight = integer(options.height ?? 32, 'output height', 4, 1024)
  const padding = integer(options.padding ?? 2, 'padding', 0, Math.floor((Math.min(targetWidth, targetHeight) - 1) / 2))
  const rect = options.rect ?? { x: 0, y: 0, width, height }
  if (!rect || typeof rect !== 'object') throw new TypeError('Expected crop rectangle')
  const x = integer(rect.x, 'crop x', 0, width - 1)
  const y = integer(rect.y, 'crop y', 0, height - 1)
  const w = integer(rect.width, 'crop width', 1, width - x)
  const h = integer(rect.height, 'crop height', 1, height - y)
  const matte = options.background ?? 255
  if (!Number.isFinite(matte) || matte < 0 || matte > 255) throw new RangeError('Background must be in [0, 255]')

  const gray = new Float32Array(w * h)
  const border = []
  for (let row = 0; row < h; row++) for (let col = 0; col < w; col++) {
    const p = ((y + row) * width + x + col) * 4
    const alpha = data[p + 3] / 255
    const value = (alpha * (data[p] * .2126 + data[p + 1] * .7152 + data[p + 2] * .0722) + (1 - alpha) * matte) / 255
    gray[row * w + col] = value
    if (row === 0 || col === 0 || row === h - 1 || col === w - 1) border.push(value)
  }
  border.sort((a, b) => a - b)
  const background = border[Math.floor(border.length / 2)]
  let light = options.polarity === 'light' || (options.polarity == null && background < .5)
  if (options.polarity != null && !['light', 'dark'].includes(options.polarity)) throw new RangeError('Polarity must be light or dark')
  for (let i = 0; i < gray.length; i++) gray[i] = Math.max(0, light ? gray[i] - background : background - gray[i])
  const pixels = new Float32Array(targetWidth * targetHeight).fill(1)
  const result = { width: targetWidth, height: targetHeight, pixels, rect: null, polarity: light ? 'light' : 'dark', contrast: 0, scale: 0, status: 'blank' }
  // Contrast from actual ink; use a quantile so one hot/dead pixel does not set it.
  const contrastOf = (skipRow, skipCol) => {
    const histogram = new Uint32Array(256)
    let count = 0, cumulative = 0, level = 0
    for (let row = 0; row < h; row++) for (let col = 0; col < w; col++) {
      const ink = gray[row * w + col]
      if (ink > .03 && !skipRow?.[row] && !skipCol?.[col]) { histogram[Math.min(255, Math.round(ink * 255))]++; count++ }
    }
    for (; level < 255; level++) {
      cumulative += histogram[level]
      if (cumulative >= count * .95) break
    }
    return count ? level / 255 : null
  }
  let contrast = contrastOf()
  // The paper guess can pick the wrong side: dark letters on a mid-grey wall whose outer ring reads dark are taken for
  // light text, and no ink clears the cutoff. When the other side's ink does, it is the text. A crop that prepares on the
  // first guess never gets here, so its output is unchanged. The luminance repeats the first pass's arithmetic exactly.
  if (options.polarity == null && !(contrast >= .08)) {
    for (let row = 0; row < h; row++) for (let col = 0; col < w; col++) {
      const p = ((y + row) * width + x + col) * 4, alpha = data[p + 3] / 255
      const value = Math.fround((alpha * (data[p] * .2126 + data[p + 1] * .7152 + data[p + 2] * .0722) + (1 - alpha) * matte) / 255)
      gray[row * w + col] = Math.max(0, light ? background - value : value - background)
    }
    const other = contrastOf()
    if (other >= .08) { light = !light; contrast = other; result.polarity = light ? 'light' : 'dark' }
  }
  if (contrast === null) return result
  if (contrast < .08) { result.contrast = contrast; result.status = 'low-contrast'; return result }
  let threshold = Math.max(.02, contrast * .08)
  const inked = (row, col) => gray[row * w + col] > threshold
  // A border sets neither the text's box, which would shrink the letters, nor its contrast, which would make them read
  // lighter or heavier.
  const [skipRow, skipCol] = borders(inked, w, h)
  if (skipRow.includes(1) || skipCol.includes(1)) {
    contrast = contrastOf(skipRow, skipCol)
    if (contrast === null) return result
    if (contrast < .08) { result.contrast = contrast; result.status = 'low-contrast'; return result }
    threshold = Math.max(.02, contrast * .08)
  }
  result.contrast = contrast
  let [left, top, right, bottom] = box(inked, w, h, skipRow, skipCol)
  if (right < left) return result
  left = Math.max(0, left - 1); top = Math.max(0, top - 1)
  right = Math.min(w - 1, right + 1); bottom = Math.min(h - 1, bottom + 1)
  const cropWidth = right - left + 1, cropHeight = bottom - top + 1
  const scale = Math.min((targetWidth - padding * 2) / cropWidth, (targetHeight - padding * 2) / cropHeight)
  const outWidth = cropWidth * scale, outHeight = cropHeight * scale
  const offsetX = (targetWidth - outWidth) / 2, offsetY = (targetHeight - outHeight) / 2
  // Area sampling for shrink; bilinear for enlargement. One scale for both axes.
  for (let row = 0; row < targetHeight; row++) for (let col = 0; col < targetWidth; col++) {
    const u = (col + .5 - offsetX) / scale, v = (row + .5 - offsetY) / scale
    let value = 0
    if (scale >= 1) {
      if (u < 0 || u >= cropWidth || v < 0 || v >= cropHeight) continue
      const sx = Math.max(0, Math.min(cropWidth - 1, u - .5))
      const sy = Math.max(0, Math.min(cropHeight - 1, v - .5))
      const ix = Math.floor(sx), iy = Math.floor(sy), fx = sx - ix, fy = sy - iy
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        value += gray[(top + Math.min(cropHeight - 1, iy + dy)) * w + left + Math.min(cropWidth - 1, ix + dx)] * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy)
      }
    } else {
      const x0 = Math.max(0, (col - offsetX) / scale), x1 = Math.min(cropWidth, (col + 1 - offsetX) / scale)
      const y0 = Math.max(0, (row - offsetY) / scale), y1 = Math.min(cropHeight, (row + 1 - offsetY) / scale)
      if (x1 <= x0 || y1 <= y0) continue
      for (let iy = Math.floor(y0); iy < Math.ceil(y1); iy++) for (let ix = Math.floor(x0); ix < Math.ceil(x1); ix++) {
        value += gray[(top + iy) * w + left + ix] * (Math.min(x1, ix + 1) - Math.max(x0, ix)) * (Math.min(y1, iy + 1) - Math.max(y0, iy))
      }
      // The part of the destination pixel outside the crop is background.
      value *= scale * scale
    }
    pixels[row * targetWidth + col] = 1 - Math.min(1, value / contrast)
  }
  return { ...result, status: 'ok', rect: { x: x + left, y: y + top, width: cropWidth, height: cropHeight }, scale }
}
