import { prepare, frame } from './prepare.mjs'

// Only these returned arrays are sent to inference. No display-only crop or padding.
export function prepareInput(image, options = {}) {
  const { height = 48, width = 128, windows = 3, background = 255 } = options
  if (!Number.isInteger(height) || height < 4 || height > 128 || !Number.isInteger(width) || width < 4 || width > 512 || !Number.isInteger(windows) || windows < 1 || windows > 3) throw new RangeError('Invalid input geometry')
  const found = prepare(image, { width: 4, height: 4, padding: 0, background })
  if (found.status !== 'ok') return { status: found.status, windows: [] }
  // A crop that touches the text prepares as one with a margin; rects map back to the crop.
  const framed = frame(image, found.rect, background)
  if (framed) {
    const result = prepareInput(framed.image, options)
    return { ...result, windows: result.windows.map(window => ({ ...window, rect: framed.back(window.rect) })) }
  }
  const bounds = found.rect
  const span = Math.min(bounds.width, Math.max(1, Math.round(bounds.height * width / height)))
  const count = Math.min(windows, Math.max(1, Math.ceil(bounds.width / span)))
  const result = []
  for (let i = 0; i < count; i++) {
    const x = bounds.x + Math.round((bounds.width - span) * (count === 1 ? .5 : i / (count - 1)))
    const rect = { ...bounds, x, width: span }
    const targetWidth = Math.min(width, Math.max(4, Math.round(span * height / bounds.height)))
    const prepared = prepare(image, { width: targetWidth, height, padding: 0, background, polarity: found.polarity, rect })
    if (prepared.status !== 'ok') continue
    // Quantize once, identically in training, display and inference.
    const bytes = Uint8Array.from(prepared.pixels, p => Math.round(p * 255))
    let left = targetWidth, top = height, right = -1, bottom = -1
    for (let y = 0; y < height; y++) for (let x = 0; x < targetWidth; x++) if (bytes[y * targetWidth + x] < 255) {
      left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y)
    }
    if (right < left) continue
    const w = right - left + 1, h = bottom - top + 1
    const pixels = new Float32Array(w * h)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pixels[y * w + x] = bytes[(y + top) * targetWidth + left + x] / 255
    result.push({ width: w, height: h, pixels, rect: prepared.rect })
  }
  return { status: result.length ? 'ok' : 'blank', windows: result }
}
