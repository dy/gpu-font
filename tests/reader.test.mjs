import test from 'node:test'
import assert from 'node:assert/strict'
import { readReader, readLine, decode, HEIGHT } from '../src/reader.mjs'

// Weights as train/read.py packs them: 8 bits, one scale per output row.
const packed = values => btoa(String.fromCharCode(...Int8Array.from(values).map(v => v & 255)))
const layer = (kind, shape, values, { stride = [1, 1], dilation = [1, 1], groups = 1, bias } = {}) =>
  ({ kind, shape, stride, dilation, groups, scale: Array(shape[0]).fill(1), bias: bias ?? Array(shape[0]).fill(0), bits: 8, weights: packed(values) })
// Three stride-2 single-channel passes bring 48 rows to 6, the fold turns them into 6 channels, the head reads two classes.
const tiny = (head = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], biases) => ({
  version: 1, kind: 'line-reader', architecture: 'test', classes: 'a', preparation: { height: HEIGHT, maxWidth: 1024, method: 'line', stride: 8 },
  layers: [layer('layer', [1, 1, 1, 1], [1], { stride: [2, 2] }), layer('layer', [1, 1, 1, 1], [1], { stride: [2, 2] }), layer('layer', [1, 1, 1, 1], [1], { stride: [2, 2] }),
    layer('fold', [2, 6, 1, 1], head, { bias: biases })]
    .map((l, i, all) => i === all.length - 1 ? { ...l, kind: 'fold' } : l).concat([layer('head', [2, 2, 1, 1], [1, 0, 0, 1])])
})
const strip = (width, ink = () => 1) => ({ width, height: HEIGHT, pixels: Float32Array.from({ length: width * HEIGHT }, (_, i) => ink(i % width, Math.floor(i / width))) })

test('greedy decoding collapses repeats and drops blanks', () => {
  assert.equal(decode([0, 1, 1, 0, 2, 0, 0], 'hi'), 'hi'); assert.equal(decode([1, 1, 0, 1], 'h'), 'hh'); assert.equal(decode([], 'h'), ''); assert.equal(decode([0, 0], 'h'), '')
})

test('a reader is checked layer by layer: channels chain, the rows end as one, the head matches the classes', () => {
  const model = readReader(tiny())
  assert.equal(model.layers.length, 5); assert.equal(model.stride, 8); assert.equal(model.classes, 'a')
  const broken = (change, message) => { const a = tiny(); change(a); assert.throws(() => readReader(a), new RegExp(message), message) }
  broken(a => { a.kind = 'font-encoder' }, 'Unsupported line reader')
  broken(a => { a.classes = 'aa' }, 'Invalid reader classes')
  broken(a => { a.preparation.height = 32 }, 'Unsupported reader preparation')
  broken(a => { a.layers.pop() }, 'Invalid reader layers')            // no head last
  broken(a => { a.layers[1] = layer('layer', [2, 1, 1, 1], [1, 1], { stride: [2, 2] }) }, 'Invalid layer channels')  // the next layer expects one channel, not two
  broken(a => { a.layers[3] = layer('layer', [2, 1, 1, 1], [1, 1]) }, 'does not end in one row')  // no fold: six rows reach the head
  broken(a => { a.layers[4].shape = [3, 2, 1, 1]; a.layers[4].scale = [1, 1, 1]; a.layers[4].bias = [0, 0, 0]; a.layers[4].weights = packed([1, 0, 0, 1, 0, 0]) }, 'does not match its classes')
  broken(a => { a.preparation.stride = 4 }, 'stride does not match')
  broken(a => { a.layers[0].weights = 'AAAA' }, 'Invalid weight count')  // three bytes where one weight needs one
  broken(a => { a.layers[0].scale = [0] }, 'Invalid layer scales')
})

test('a strip reads through the layers as a grouped convolution with same padding, ink as one, the last layer unrectified', () => {
  // The pass-through layers keep every eighth pixel of every eighth row (stride 2, 1×1 kernels sample at 0, 2, 4...); the fold
  // weights pick row 0 for class blank and row 1 for class "a"; the head is the identity. Paper everywhere: all logits are zero
  // and the text is empty. Ink on the strip's top row only: blank wins every column.
  const model = readReader(tiny()), paper = readLine(model, strip(16))
  assert.deepEqual([...paper.logits], [0, 0, 0, 0]); assert.equal(paper.text, ''); assert.deepEqual(paper.columns, [0, 0])
  const top = readLine(model, strip(16, (x, y) => y === 0 ? 0 : 1))
  assert.deepEqual([...top.logits], [1, 1, 0, 0]); assert.equal(top.text, '')
  // Ink on row 8 reaches channel 1 of the fold (row 8 → 4 → 2 → 1): class "a" at both columns, read once.
  const second = readLine(model, strip(16, (x, y) => y === 8 ? 0 : 1))
  assert.deepEqual([...second.logits], [0, 0, 1, 1]); assert.equal(second.text, 'a'); assert.deepEqual(second.columns, [1, 1])
  // Only the left column inked on row 8: "a" then blank.
  const left = readLine(model, strip(16, (x, y) => y === 8 && x < 8 ? 0 : 1))
  assert.deepEqual(left.columns, [1, 0]); assert.equal(left.text, 'a')
  // A negative bias on the head's last layer is kept as is: logits may go below zero where hidden layers would rectify.
  const biased = readReader({ ...tiny(), layers: tiny().layers.map((l, i, all) => i === all.length - 1 ? { ...l, bias: [0, -2] } : l) })
  assert.deepEqual([...readLine(biased, strip(8)).logits], [0, -2])
  assert.throws(() => readLine(model, strip(2000)), /Invalid strip/); assert.throws(() => readLine(model, { width: 8, height: 32, pixels: new Float32Array(256) }), /Invalid strip/)
})
