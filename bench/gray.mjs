// Numeric probe kernel: caller validates the byte-array type before marshaling.
// Weighted grayscale with white alpha compositing; not a full crop normalizer.
export function gray(rgba) {
  if (rgba.length % 4 !== 0) throw new RangeError('Expected complete RGBA pixels')
  const out = new Float32Array(rgba.length / 4)
  for (let i = 0; i < out.length; i++) {
    const p = i * 4
    const alpha = rgba[p + 3] / 255
    const lum = (rgba[p] * 0.2126 + rgba[p + 1] * 0.7152 + rgba[p + 2] * 0.0722) / 255
    out[i] = 1 - alpha + alpha * lum
  }
  return out
}
