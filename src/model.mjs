import { unit } from './rank.mjs'

export const architecture = 'conv12-24-32-48-gap-32-v1'
export const channels = [1, 12, 24, 32, 48]

export function readModel(artifact) {
  if (artifact?.architecture !== architecture || artifact.version !== 1) throw new Error('Unsupported font model')
  const p = artifact.preparation
  if (p?.width !== 128 || p.height !== 32 || p.padding !== 2) throw new Error('Unsupported model preparation')
  const shapes = {}
  for (let i = 0; i < 4; i++) {
    shapes[`features.${i * 2}.weight`] = [channels[i + 1], channels[i], 3, 3]
    shapes[`features.${i * 2}.bias`] = [channels[i + 1]]
  }
  shapes['project.weight'] = [32, 48]; shapes['project.bias'] = [32]
  if (!artifact.tensors || Object.keys(artifact.tensors).length !== Object.keys(shapes).length) throw new Error('Unexpected model tensors')
  const tensors = {}
  for (const [name, shape] of Object.entries(shapes)) {
    const tensor = artifact.tensors[name]
    if (JSON.stringify(tensor?.shape) !== JSON.stringify(shape) || !Array.isArray(tensor.values) || tensor.values.length !== shape.reduce((a, b) => a * b, 1) || !tensor.values.every(Number.isFinite)) throw new Error(`Invalid tensor: ${name}`)
    tensors[name] = Float32Array.from(tensor.values)
    if (!tensors[name].every(Number.isFinite)) throw new Error(`Tensor overflow: ${name}`)
  }
  const ids = new Set()
  if (!Array.isArray(artifact.catalog) || !artifact.catalog.length) throw new Error('Empty font catalog')
  const catalog = artifact.catalog.map(({ family, vector }) => {
    if (typeof family !== 'string' || !family || ids.has(family) || !Array.isArray(vector) || vector.length !== 32 || !vector.every(Number.isFinite)) throw new Error('Invalid catalog entry')
    ids.add(family)
    return { family, vector: unit(vector) }
  })
  return { tensors, catalog, preparation: { width: 128, height: 32, padding: 2 } }
}

export function checkPixels(pixels) {
  if (!(pixels instanceof Float32Array) || pixels.length !== 4096 || !pixels.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) throw new Error('Expected 128 × 32 grayscale pixels in [0, 1]')
}

export function encodeCPU(model, pixels) {
  checkPixels(pixels)
  let input = Float32Array.from(pixels, v => 1 - v), width = 128, height = 32
  for (let layer = 0; layer < 4; layer++) {
    const ci = channels[layer], co = channels[layer + 1]
    const ow = width / 2, oh = height / 2, output = new Float32Array(co * ow * oh)
    const weight = model.tensors[`features.${layer * 2}.weight`], bias = model.tensors[`features.${layer * 2}.bias`]
    for (let c = 0; c < co; c++) for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
      let sum = bias[c]
      for (let k = 0; k < ci; k++) for (let dy = 0; dy < 3; dy++) for (let dx = 0; dx < 3; dx++) {
        const ix = x * 2 + dx - 1, iy = y * 2 + dy - 1
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) sum += input[(k * height + iy) * width + ix] * weight[((c * ci + k) * 3 + dy) * 3 + dx]
      }
      output[(c * oh + y) * ow + x] = Math.max(0, sum)
    }
    input = output; width = ow; height = oh
  }
  const pooled = new Float32Array(48)
  for (let c = 0; c < 48; c++) for (let p = 0; p < 16; p++) pooled[c] += input[c * 16 + p] / 16
  const vector = new Float32Array(32)
  for (let d = 0; d < 32; d++) {
    let sum = model.tensors['project.bias'][d]
    for (let c = 0; c < 48; c++) sum += pooled[c] * model.tensors['project.weight'][d * 48 + c]
    vector[d] = sum
  }
  return unit(vector)
}
