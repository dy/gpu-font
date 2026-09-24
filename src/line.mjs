import { prepare, frame } from './prepare.mjs'
import { prepareInput } from './input.mjs'

// Experimental line preparation. Keep the deployed sampler unchanged until evaluated.
export function prepareLine(image, options = {}) {
  const { width = 128, height = 48, windows = 3, background = 255, deskew = false, sampler = 'groups' } = options
  if (!Number.isInteger(width) || width < 4 || width > 128 || !Number.isInteger(height) || height < 4 || height > 48 || !Number.isInteger(windows) || windows < 1 || windows > 3) throw new RangeError('Invalid line geometry')
  if (typeof deskew !== 'boolean' || (options.angle != null && (!Number.isFinite(options.angle) || Math.abs(options.angle) > 8))) throw new RangeError('Invalid line angle')
  if (!['groups', 'windows'].includes(sampler)) throw new RangeError('Invalid line sampler')
  if (options.regions != null && (!Array.isArray(options.regions) || !options.regions.length || options.regions.length > windows || options.regions.some(points => !Array.isArray(points) || points.length !== 4 || points.some(p => !Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite))))) throw new RangeError('Invalid word regions')
  const found = prepare(image, { width: 4, height: 4, padding: 0, background })
  if (found.status !== 'ok') return { status: found.status, windows: [], angle: 0 }
  const framed = frame(image, found.rect, background)
  if (!framed) return prepareFound(image, found, options)
  // Word regions move into the framed image, and every region back out of it.
  const { x, y } = framed, regions = options.regions?.map(points => points.map(([px, py]) => [px + x, py + y]))
  const result = prepareFound(framed.image, prepare(framed.image, { width: 4, height: 4, padding: 0, background }), { ...options, ...(regions && { regions }) })
  return { ...result, windows: result.windows.map(window => ({ ...window, rect: framed.back(window.rect), ...(window.polygon && { polygon: window.polygon.map(([px, py]) => [px - x, py - y]) }) })) }
}

function prepareFound(image, found, options) {
  const { width = 128, height = 48, windows = 3, background = 255, deskew = false, sampler = 'groups' } = options
  const gray = new Float32Array(image.width * image.height), border = []
  for (let y = 0; y < image.height; y++) for (let x = 0; x < image.width; x++) {
    const i = y * image.width + x, p = i * 4, a = image.data[p + 3] / 255
    gray[i] = a * (image.data[p] * .2126 + image.data[p + 1] * .7152 + image.data[p + 2] * .0722) + (1 - a) * background
    if (!x || !y || x === image.width - 1 || y === image.height - 1) border.push(gray[i])
  }
  border.sort((a, b) => a - b)
  const bg = border[border.length >> 1], threshold = Math.max(.02, found.contrast * .08) * 255
  const ink = value => found.polarity === 'light' ? value - bg > threshold : bg - value > threshold
  let angle = 0
  if (deskew && found.rect.width > found.rect.height * 2.5 && found.rect.height >= 8) {
    const points = [], step = Math.max(1, Math.ceil(found.rect.width * found.rect.height / 16000))
    let n = 0
    for (let y = found.rect.y; y < found.rect.y + found.rect.height; y++) for (let x = found.rect.x; x < found.rect.x + found.rect.width; x++) {
      if (ink(gray[y * image.width + x]) && n++ % step === 0) points.push([x - image.width / 2, y - image.height / 2])
    }
    const bins = new Uint32Array(Math.ceil(Math.hypot(image.width, image.height)) + 4)
    const score = degrees => {
      bins.fill(0)
      const a = degrees * Math.PI / 180, c = Math.cos(a), s = Math.sin(a)
      for (const [x, y] of points) bins[Math.round(y * c - x * s + bins.length / 2)]++
      return bins.reduce((total, count) => total + count * count, 0)
    }
    const initial = score(0); let best = initial
    for (let a = -8; a <= 8; a += .5) {
      const value = score(a)
      if (value > best) { best = value; angle = a }
    }
    if (best < initial * 1.12) angle = 0
  }
  if (options.angle != null) {
    angle = options.angle
  }
  if (!angle && sampler === 'windows') return { ...prepareInput(image, { width, height, windows, background }), angle }
  const a = angle * Math.PI / 180, c = Math.cos(a), s = Math.sin(a)
  const w = angle ? Math.ceil(image.width * c + image.height * Math.abs(s)) : image.width
  const h = angle ? Math.ceil(image.height * c + image.width * Math.abs(s)) : image.height
  if (w * h > 16_777_216) throw new RangeError('Rotated line exceeds pixel limit')
  const toSource = (x, y) => [c * (x - w / 2) - s * (y - h / 2) + image.width / 2, s * (x - w / 2) + c * (y - h / 2) + image.height / 2]
  const toLine = (x, y) => [c * (x - image.width / 2) + s * (y - image.height / 2) + w / 2, -s * (x - image.width / 2) + c * (y - image.height / 2) + h / 2]
  const originalRegion = r => {
    const polygon = [[r.x, r.y], [r.x + r.width, r.y], [r.x + r.width, r.y + r.height], [r.x, r.y + r.height]].map(([x, y]) => toSource(x, y))
    const x = Math.max(0, Math.floor(Math.min(...polygon.map(p => p[0])))), y = Math.max(0, Math.floor(Math.min(...polygon.map(p => p[1]))))
    return { polygon, rect: { x, y, width: Math.min(image.width, Math.ceil(Math.max(...polygon.map(p => p[0])))) - x, height: Math.min(image.height, Math.ceil(Math.max(...polygon.map(p => p[1])))) - y } }
  }
  const data = new Uint8Array(w * h * 4), columns = new Uint8Array(w)
  let top = h, bottom = -1
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [sx, sy] = angle ? toSource(x + .5, y + .5).map(v => v - .5) : [x, y]
    let value = 0
    const ix = Math.floor(sx), iy = Math.floor(sy), fx = sx - ix, fy = sy - iy
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const xx = ix + dx, yy = iy + dy
      value += (xx < 0 || yy < 0 || xx >= image.width || yy >= image.height ? bg : gray[yy * image.width + xx]) * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy)
    }
    const i = (y * w + x) * 4
    data.fill(Math.round(value), i, i + 3); data[i + 3] = 255
    if (ink(value)) { columns[x] = 1; top = Math.min(top, y); bottom = Math.max(bottom, y) }
  }
  if (bottom < top) return { status: 'blank', windows: [], angle }
  if (sampler === 'windows') {
    const input = prepareInput({ width: w, height: h, data }, { width, height, windows, background })
    return { ...input, angle, windows: input.windows.map(window => ({ ...window, ...originalRegion(window.rect) })) }
  }
  const runs = []
  for (let x = 0; x < w; x++) {
    if (!columns[x]) continue
    const left = x
    while (x + 1 < w && columns[x + 1]) x++
    const previous = runs.at(-1)
    if (previous && left - previous.right - 1 < Math.max(2, (bottom - top + 1) * .22)) previous.right = x
    else runs.push({ left, right: x })
  }
  // Merge neighboring words rather than slicing through glyphs to meet the window budget.
  while (runs.length > windows) {
    let index = 0
    for (let i = 1; i < runs.length - 1; i++) if (runs[i + 1].right - runs[i].left < runs[index + 1].right - runs[index].left) index = i
    runs.splice(index, 2, { left: runs[index].left, right: runs[index + 1].right })
  }
  const bounding = points => {
    const left = Math.max(0, Math.floor(Math.min(...points.map(p => p[0]))) - 1), right = Math.min(w, Math.ceil(Math.max(...points.map(p => p[0]))) + 1)
    const top = Math.max(0, Math.floor(Math.min(...points.map(p => p[1]))) - 1), bottom = Math.min(h, Math.ceil(Math.max(...points.map(p => p[1]))) + 1)
    return { x: left, y: top, width: right - left, height: bottom - top }
  }
  const rects = options.regions ? options.regions.map(points => bounding(points.map(([x, y]) => toLine(x, y)))) : runs.map(({ left, right }) => bounding([[left, top], [right + 1, bottom + 1]]))
  const result = []
  for (const rect of rects) {
    const input = prepare({ width: w, height: h, data }, { width, height, padding: 0, rect, background })
    if (input.status !== 'ok') continue
    const bytes = Uint8Array.from(input.pixels, p => Math.round(p * 255))
    let left = width, right = -1, top = height, bottom = -1
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (bytes[y * width + x] < 255) {
      left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y)
    }
    if (right < left) continue
    const ow = right - left + 1, oh = bottom - top + 1
    const pixels = Float32Array.from({ length: ow * oh }, (_, i) => bytes[(top + Math.floor(i / ow)) * width + left + i % ow] / 255)
    result.push({ width: ow, height: oh, pixels, ...originalRegion(input.rect) })
  }
  return { status: result.length ? 'ok' : 'blank', windows: result, angle }
}
