import { base64, unpack, validDimensions } from './catalog.mjs'

export const architecture = 'font-conv16-32-48-64-v1'
export const contextArchitecture = 'font-conv16-32-48-64-64-v2'
export const corpusArchitecture = 'font-conv32-64-96-128-128-v3'
export const largeArchitecture = 'font-conv64-128-192-256-256-v4'
export const widerArchitecture = 'font-conv96-192-288-384-384-v5'
export const widestArchitecture = 'font-conv128-256-384-512-512-v6'
export const studentArchitecture = 'font-sep96-192-288-384-384-v7'  // depthwise-separable blocks past the first
export const wideStudentArchitecture = 'font-sep128-256-384-512-512-v7'
export const channels = [1, 16, 32, 48, 64]
export const strides = [1, 2, 2, 2]
// Channels per layer for each supported architecture; every one but the first adds a stride-1 context layer.
const shapes = new Map([[architecture, channels], [contextArchitecture, [...channels, 64]], [corpusArchitecture, [1, 32, 64, 96, 128, 128]],
  [largeArchitecture, [1, 64, 128, 192, 256, 256]], [widerArchitecture, [1, 96, 192, 288, 384, 384]], [widestArchitecture, [1, 128, 256, 384, 512, 512]], [studentArchitecture, [1, 96, 192, 288, 384, 384]], [wideStudentArchitecture, [1, 128, 256, 384, 512, 512]]])

export function readNetwork(artifact) {
  if (artifact?.version !== 1 || !shapes.has(artifact.architecture)) throw new Error('Unsupported font classifier')
  const fonts = artifact.fonts, p = artifact.preparation
  const encoder = artifact.kind === 'font-encoder'
  if (encoder && (!validDimensions(artifact.dimensions) || artifact.normalization !== 'l2' || fonts !== undefined)) throw new Error('Invalid encoder output')
  const context = artifact.architecture !== architecture
  const shapeChannels = shapes.get(artifact.architecture), shapeStrides = context ? [...strides, 1] : strides
  const depth = shapeStrides.length
  const dilations = artifact.dilations ?? Array(depth).fill(1)
  if (!Array.isArray(dilations) || dilations.length !== depth || dilations.some(d => d !== 1 && d !== 2)) throw new Error('Invalid filter dilations')
  if (!encoder && (!Array.isArray(fonts) || !fonts.length || fonts.length > 10000 || fonts.some(f => typeof f !== 'string' || !f) || new Set(fonts).size !== fonts.length)) throw new Error('Invalid font labels')
  const outputs = encoder ? artifact.dimensions : fonts.length
  if (p?.width !== 128 || p.height !== 48 || p.windows !== 3) throw new Error('Unsupported input preparation')
  const method = p.method ?? 'windows'
  if (!['windows', 'deskew-windows'].includes(method)) throw new Error('Unsupported input preparation method')
  // The layers in artifact order: one 3×3 conv per block, or for a separable block past the first a depthwise 3×3 (the
  // block's stride and dilation) then a pointwise 1×1; the head last.
  const separable = artifact.architecture.startsWith('font-sep'), plan = []
  for (let i = 0; i < depth; i++) {
    const [ci, co, stride, dilation] = [shapeChannels[i], shapeChannels[i + 1], shapeStrides[i], dilations[i]]
    if (separable && i) plan.push({ kind: 'depthwise', shape: [ci, 1, 3, 3], stride, dilation }, { kind: 'pointwise', shape: [co, ci, 1, 1], stride: 1, dilation: 1 })
    else plan.push({ kind: 'conv', shape: [co, ci, 3, 3], stride, dilation })
  }
  plan.push({ kind: 'head', shape: [outputs, shapeChannels.at(-1)] })
  if (!Array.isArray(artifact.layers) || artifact.layers.length !== plan.length) throw new Error('Invalid network layers')
  const layers = artifact.layers.map((layer, i) => {
    const { shape } = plan[i]
    if (JSON.stringify(layer.shape) !== JSON.stringify(shape)) throw new Error('Invalid layer shape')
    const [rows] = shape, count = shape.reduce((a, b) => a * b, 1)
    if (!Array.isArray(layer.scale) || layer.scale.length !== rows || layer.scale.some(s => !Number.isFinite(s) || s <= 0) || !Array.isArray(layer.bias) || layer.bias.length !== rows || !layer.bias.every(Number.isFinite)) throw new Error('Invalid layer values')
    // Weights are packed `bits` apiece, 8 unless the layer says fewer; with a codebook, each is an index into 2^bits shared values.
    const bits = layer.bits ?? 8, codebook = layer.codebook ?? null
    if (!Number.isInteger(bits) || bits < 2 || bits > 8 || typeof layer.weights !== 'string' || !base64(layer.weights)) throw new Error('Invalid packed weights')
    if (codebook !== null && (!Array.isArray(codebook) || codebook.length !== 2 ** bits || !codebook.every(Number.isFinite))) throw new Error('Invalid codebook')
    const binary = atob(layer.weights), bytes = new Uint8Array(binary.length)
    if (binary.length !== Math.ceil(count * bits / 8)) throw new Error('Invalid weight count')
    // Plain loops: per-element callbacks cost ~100 ms on a million-weight encoder.
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
    const values = unpack(bytes, bits, count), weights = new Float32Array(count), per = count / rows, half = 2 ** (bits - 1)
    for (let n = 0; n < count; n++) weights[n] = (codebook ? codebook[values[n] + half] : values[n]) * layer.scale[Math.floor(n / per)]
    const bias = Float32Array.from(layer.bias)
    if (!weights.every(Number.isFinite) || !bias.every(Number.isFinite)) throw new Error('Weight overflow')
    return { ...plan[i], weights, bias }
  })
  // An encoder may ship a rejection threshold: a best score below it means the font is probably in no catalog.
  const threshold = artifact.rejection?.threshold ?? null
  if (threshold !== null && !(Number.isFinite(threshold) && threshold > -1 && threshold < 1)) throw new Error('Invalid rejection threshold')
  return { fonts: encoder ? null : [...fonts], outputs, kind: encoder ? 'font-encoder' : 'font-classifier', layers, channels: [...shapeChannels], strides: [...shapeStrides], dilations: [...dilations], preparation: { width: p.width, height: p.height, windows: p.windows, method }, threshold }
}

export function checkInput(input) {
  if (!input || !Number.isInteger(input.width) || input.width < 1 || input.width > 128 || !Number.isInteger(input.height) || input.height < 1 || input.height > 48 || !(input.pixels instanceof Float32Array) || input.pixels.length !== input.width * input.height || !input.pixels.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) throw new Error('Expected a bounded grayscale text window')
}

export function inferCPU(model, window) {
  checkInput(window)
  const { channels } = model
  let { width, height } = window, input = Float32Array.from(window.pixels, p => 1 - p)
  for (const layer of model.layers.slice(0, -1)) {
    const { kind, stride, dilation, weights, bias } = layer, [co, ci] = layer.shape, taps = kind === 'pointwise' ? 1 : 3
    const ow = Math.ceil(width / stride), oh = Math.ceil(height / stride), output = new Float32Array(co * ow * oh)
    for (let c = 0; c < co; c++) for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
      let sum = bias[c]
      for (let k = 0; k < ci; k++) for (let dy = 0; dy < taps; dy++) for (let dx = 0; dx < taps; dx++) {
        const ix = x * stride + (dx - 1) * dilation * (taps > 1), iy = y * stride + (dy - 1) * dilation * (taps > 1), source = kind === 'depthwise' ? c : k
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) sum += input[(source * height + iy) * width + ix] * weights[((c * ci + k) * taps + dy) * taps + dx]
      }
      output[(c * oh + y) * ow + x] = Math.max(0, sum)
    }
    input = output; width = ow; height = oh
  }
  const features = channels.at(-1), pooled = new Float32Array(features), count = width * height
  for (let c = 0; c < features; c++) {
    let sum = 0
    for (let p = 0; p < count; p++) sum += input[c * count + p]
    pooled[c] = sum / count
  }
  const { weights, bias } = model.layers.at(-1)
  return Float32Array.from({ length: model.outputs }, (_, d) => {
    let sum = bias[d]
    for (let c = 0; c < features; c++) sum += weights[d * features + c] * pooled[c]
    return sum
  })
}

export function rankWindows(logits, fonts, temperature = 1) {
  if (!Number.isFinite(temperature) || temperature <= 0) throw new Error('Invalid temperature')
  if (!Array.isArray(logits) || !logits.length) throw new Error('No window scores')
  const scores = new Float64Array(fonts.length)
  for (const values of logits) {
    if (values.length !== fonts.length || !values.every(Number.isFinite)) throw new Error('Invalid window scores')
    const max = Math.max(...values), p = Array.from(values, v => Math.exp((v - max) / temperature)), total = p.reduce((a, b) => a + b)
    for (let i = 0; i < scores.length; i++) scores[i] += p[i] / total / logits.length
  }
  return fonts.map((family, i) => ({ family, score: scores[i] })).sort((a, b) => b.score - a.score || a.family.localeCompare(b.family))
}
