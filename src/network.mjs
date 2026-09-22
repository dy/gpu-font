export const architecture = 'font-conv16-32-48-64-v1'
export const contextArchitecture = 'font-conv16-32-48-64-64-v2'
export const corpusArchitecture = 'font-conv32-64-96-128-128-v3'
export const channels = [1, 16, 32, 48, 64]
export const strides = [1, 2, 2, 2]

export function readNetwork(artifact) {
  if (artifact?.version !== 1 || ![architecture, contextArchitecture, corpusArchitecture].includes(artifact.architecture)) throw new Error('Unsupported font classifier')
  const fonts = artifact.fonts, p = artifact.preparation
  const context = artifact.architecture !== architecture
  const shapeChannels = artifact.architecture === corpusArchitecture ? [1, 32, 64, 96, 128, 128] : context ? [...channels, 64] : channels, shapeStrides = context ? [...strides, 1] : strides
  const depth = shapeStrides.length
  const dilations = artifact.dilations ?? Array(depth).fill(1)
  if (!Array.isArray(dilations) || dilations.length !== depth || dilations.some(d => d !== 1 && d !== 2)) throw new Error('Invalid filter dilations')
  if (!Array.isArray(fonts) || !fonts.length || fonts.length > 10000 || fonts.some(f => typeof f !== 'string' || !f) || new Set(fonts).size !== fonts.length) throw new Error('Invalid font labels')
  if (p?.width !== 128 || p.height !== 48 || p.windows !== 3) throw new Error('Unsupported input preparation')
  if (!Array.isArray(artifact.layers) || artifact.layers.length !== depth + 1) throw new Error('Invalid network layers')
  const layers = artifact.layers.map((layer, i) => {
    const shape = i === depth ? [fonts.length, shapeChannels.at(-1)] : [shapeChannels[i + 1], shapeChannels[i], 3, 3]
    if (JSON.stringify(layer.shape) !== JSON.stringify(shape)) throw new Error('Invalid layer shape')
    const [rows] = shape, count = shape.reduce((a, b) => a * b, 1)
    if (!Array.isArray(layer.scale) || layer.scale.length !== rows || layer.scale.some(s => !Number.isFinite(s) || s <= 0) || !Array.isArray(layer.bias) || layer.bias.length !== rows || !layer.bias.every(Number.isFinite)) throw new Error('Invalid layer values')
    if (typeof layer.weights !== 'string' || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(layer.weights)) throw new Error('Invalid packed weights')
    const bytes = Uint8Array.from(atob(layer.weights), c => c.charCodeAt(0))
    if (bytes.length !== count) throw new Error('Invalid weight count')
    const integers = new Int8Array(bytes.buffer), weights = Float32Array.from(integers, (v, n) => v * layer.scale[Math.floor(n / (count / rows))])
    const bias = Float32Array.from(layer.bias)
    if (!weights.every(Number.isFinite) || !bias.every(Number.isFinite)) throw new Error('Weight overflow')
    return { shape, weights, bias }
  })
  return { fonts: [...fonts], layers, channels: [...shapeChannels], strides: [...shapeStrides], dilations: [...dilations], preparation: { width: p.width, height: p.height, windows: p.windows } }
}

export function checkInput(input) {
  if (!input || !Number.isInteger(input.width) || input.width < 1 || input.width > 128 || !Number.isInteger(input.height) || input.height < 1 || input.height > 48 || !(input.pixels instanceof Float32Array) || input.pixels.length !== input.width * input.height || !input.pixels.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) throw new Error('Expected a bounded grayscale text window')
}

export function inferCPU(model, window) {
  checkInput(window)
  const { channels, strides } = model
  let { width, height } = window, input = Float32Array.from(window.pixels, p => 1 - p)
  for (let layer = 0; layer < strides.length; layer++) {
    const ci = channels[layer], co = channels[layer + 1], stride = strides[layer]
    const ow = Math.ceil(width / stride), oh = Math.ceil(height / stride)
    const output = new Float32Array(co * ow * oh), { weights, bias } = model.layers[layer]
    for (let c = 0; c < co; c++) for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
      let sum = bias[c]
      for (let k = 0; k < ci; k++) for (let dy = 0; dy < 3; dy++) for (let dx = 0; dx < 3; dx++) {
        const ix = x * stride + (dx - 1) * model.dilations[layer], iy = y * stride + (dy - 1) * model.dilations[layer]
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) sum += input[(k * height + iy) * width + ix] * weights[((c * ci + k) * 3 + dy) * 3 + dx]
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
  return Float32Array.from(model.fonts, (_, d) => {
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
