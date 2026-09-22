const MAX_PIXELS = 16_777_216

function integer(value, name, min, max) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new RangeError(`Invalid ${name}`)
  return value
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
  const light = options.polarity === 'light' || (options.polarity == null && background < .5)
  if (options.polarity != null && !['light', 'dark'].includes(options.polarity)) throw new RangeError('Polarity must be light or dark')
  // Contrast from actual ink; use a quantile so one hot/dead pixel does not set it.
  const histogram = new Uint32Array(256)
  let inkCount = 0
  for (let i = 0; i < gray.length; i++) {
    const ink = Math.max(0, light ? gray[i] - background : background - gray[i])
    gray[i] = ink
    if (ink > .03) { histogram[Math.min(255, Math.round(ink * 255))]++; inkCount++ }
  }
  const pixels = new Float32Array(targetWidth * targetHeight).fill(1)
  const result = { width: targetWidth, height: targetHeight, pixels, rect: null, polarity: light ? 'light' : 'dark', contrast: 0, scale: 0, status: 'blank' }
  if (!inkCount) return result
  let cumulative = 0, level = 0
  for (; level < 255; level++) {
    cumulative += histogram[level]
    if (cumulative >= inkCount * .95) break
  }
  const contrast = level / 255
  result.contrast = contrast
  if (contrast < .08) { result.status = 'low-contrast'; return result }
  const threshold = Math.max(.02, contrast * .08)
  let left = w, top = h, right = -1, bottom = -1
  for (let row = 0; row < h; row++) for (let col = 0; col < w; col++) {
    if (gray[row * w + col] <= threshold) continue
    left = Math.min(left, col); right = Math.max(right, col)
    top = Math.min(top, row); bottom = Math.max(bottom, row)
  }
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
