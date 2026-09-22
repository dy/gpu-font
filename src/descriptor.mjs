import { unit } from './rank.mjs'

// Deliberately cheap, untrained baseline: pooled edge directions and row ink.
export function descriptor(pixels, width, height) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 3 || height < 3 || pixels?.length !== width * height) throw new RangeError('Invalid descriptor image')
  const features = new Float32Array(40)
  for (const value of pixels) if (!Number.isFinite(value) || value < 0 || value > 1) throw new RangeError('Pixels must be finite and in [0, 1]')
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    const p = y * width + x
    const gx = pixels[p + 1] - pixels[p - 1], gy = pixels[p + width] - pixels[p - width]
    const angle = (Math.atan2(gy, gx) + Math.PI) % Math.PI
    const bin = Math.min(7, Math.floor(angle * 8 / Math.PI))
    features[Math.floor(y * 4 / height) * 8 + bin] += Math.hypot(gx, gy)
    features[32 + Math.floor(y * 8 / height)] += (1 - pixels[p]) * .25
  }
  return unit(features)
}

