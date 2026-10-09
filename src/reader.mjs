// A line reader in the browser: the characters of one line of Latin text, read from the strip prepareLine's `line` sampler
// gives (48 pixels tall, the line's own width) by the tiny convolutional CTC model train/read.py exports. Every column of
// the output stands for eight pixels of the strip; greedy decoding collapses repeats and drops blanks. On the CPU: a
// strip of 600 pixels is about twenty million multiply-adds.
import { base64, unpack } from './catalog.mjs'

const KINDS = ['layer', 'fold', 'head']
export const HEIGHT = 48

/** The reader from its artifact: validated, weights unpacked, the geometry of every layer checked to end in one row of
 * one logit per class. `fold` names the layer before which the remaining rows become channels. */
export function readReader(artifact) {
  if (artifact?.version !== 1 || artifact.kind !== 'line-reader' || typeof artifact.architecture !== 'string') throw new Error('Unsupported line reader')
  const { classes, preparation: p, layers } = artifact
  if (typeof classes !== 'string' || !classes.length || classes.length > 1024 || new Set(classes).size !== classes.length) throw new Error('Invalid reader classes')
  if (p?.height !== HEIGHT || p.method !== 'line' || !Number.isInteger(p.stride) || p.stride < 1 || !Number.isInteger(p.maxWidth) || p.maxWidth < 4 || p.maxWidth > 1024) throw new Error('Unsupported reader preparation')
  if (!Array.isArray(layers) || !layers.length || layers.length > 64 || layers.at(-1).kind !== 'head' || layers.filter(l => l?.kind === 'head').length !== 1 || layers.filter(l => l?.kind === 'fold').length > 1) throw new Error('Invalid reader layers')
  let channels = 1, height = HEIGHT, stride = 1
  const read = layers.map(layer => {
    const { kind, shape, stride: strides, dilation, groups, scale, bias, bits = 8, weights } = layer
    if (!KINDS.includes(kind) || !Array.isArray(shape) || shape.length !== 4 || !shape.every(n => Number.isInteger(n) && n > 0)) throw new Error('Invalid layer shape')
    const [co, cig, kh, kw] = shape
    if (!Array.isArray(strides) || strides.length !== 2 || !strides.every(s => s === 1 || s === 2) || !Array.isArray(dilation) || dilation.length !== 2 || !dilation.every(d => Number.isInteger(d) && d >= 1 && d <= 8)) throw new Error('Invalid layer geometry')
    if (!Number.isInteger(groups) || groups < 1 || kh % 2 !== 1 || kw % 2 !== 1 || kh > 3 || kw > 3) throw new Error('Invalid layer geometry')
    if (kind === 'fold') { channels *= height; height = 1 }
    if (cig * groups !== channels || co % groups !== 0) throw new Error('Invalid layer channels')
    const count = co * cig * kh * kw
    if (!Array.isArray(scale) || scale.length !== co || scale.some(s => !Number.isFinite(s) || s <= 0) || !Array.isArray(bias) || bias.length !== co || !bias.every(Number.isFinite)) throw new Error('Invalid layer scales')
    if (!Number.isInteger(bits) || bits < 2 || bits > 8 || typeof weights !== 'string' || !base64(weights)) throw new Error('Invalid packed weights')
    const binary = atob(weights), bytes = new Uint8Array(binary.length)
    if (binary.length !== Math.ceil(count * bits / 8)) throw new Error('Invalid weight count')
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
    const values = unpack(bytes, bits, count), decoded = new Float32Array(count), per = count / co
    for (let n = 0; n < count; n++) decoded[n] = values[n] * scale[Math.floor(n / per)]
    channels = co; height = Math.ceil(height / strides[0]); stride *= strides[1]
    return { kind, co, cig, kh, kw, sy: strides[0], sx: strides[1], dy: dilation[0], dx: dilation[1], groups, weights: decoded, bias: Float32Array.from(bias) }
  })
  if (height !== 1) throw new Error('Reader does not end in one row')
  if (channels !== classes.length + 1) throw new Error('Reader head does not match its classes')
  if (stride !== p.stride) throw new Error('Reader stride does not match its preparation')
  return { kind: 'line-reader', classes, stride, maxWidth: p.maxWidth, layers: read }
}

/** Greedy CTC: the class read at each column, repeats collapsed, blanks (0) dropped. */
export function decode(columns, classes) {
  let text = '', last = 0
  for (const c of columns) { if (c && c !== last) text += classes[c - 1]; last = c }
  return text
}

/** The characters of one strip ({ width, height: 48, pixels } as prepareLine's `line` sampler gives it, paper 1 and ink 0):
 * `text`, the class index read at each column (0 blank) and the logits, classes+1 per column, column-major. */
export function readLine(model, strip) {
  if (!strip || !Number.isInteger(strip.width) || strip.width < 1 || strip.width > model.maxWidth || strip.height !== HEIGHT || !(strip.pixels instanceof Float32Array) || strip.pixels.length !== strip.width * HEIGHT) throw new Error('Invalid strip')
  let height = HEIGHT, width = strip.width, input = Float32Array.from(strip.pixels, p => 1 - p)
  for (const layer of model.layers) {
    const { co, cig, kh, kw, sy, sx, dy, dx, groups } = layer, last = layer === model.layers.at(-1)
    if (layer.kind === 'fold') height = 1  // (c, y, x) read as (c·rows + y, 0, x): the same numbers, rows become channels
    const ow = Math.ceil(width / sx), oh = Math.ceil(height / sy), py = (kh - 1) / 2 * dy, px = (kw - 1) / 2 * dx, per = co / groups
    const output = new Float32Array(co * oh * ow)
    for (let c = 0; c < co; c++) {
      const g = Math.floor(c / per)
      for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
        let sum = layer.bias[c]
        for (let k = 0; k < cig; k++) {
          const plane = (g * cig + k) * height, row = (c * cig + k) * kh
          for (let ty = 0; ty < kh; ty++) {
            const iy = y * sy - py + ty * dy
            if (iy < 0 || iy >= height) continue
            for (let tx = 0; tx < kw; tx++) {
              const ix = x * sx - px + tx * dx
              if (ix >= 0 && ix < width) sum += input[(plane + iy) * width + ix] * layer.weights[(row + ty) * kw + tx]
            }
          }
        }
        output[(c * oh + y) * ow + x] = last ? sum : Math.max(0, sum)
      }
    }
    input = output; height = oh; width = ow
  }
  const classes = model.classes.length + 1, columns = new Array(width)
  for (let t = 0; t < width; t++) {
    let best = 0, value = -Infinity
    for (let c = 0; c < classes; c++) { const v = input[c * width + t]; if (v > value) { value = v; best = c } }
    columns[t] = best
  }
  return { text: decode(columns, model.classes), columns, logits: input }
}
