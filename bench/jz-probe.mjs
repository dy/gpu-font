import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { gray } from './gray.mjs'

if (!process.argv[2]) {
  throw new Error('Usage: node bench/jz-probe.mjs /absolute/path/to/jz')
}

const entry = pathToFileURL(resolve(process.argv[2], 'index.js'))
const { compile, instantiate } = await import(entry.href)
const source = await readFile(new URL('./gray.mjs', import.meta.url), 'utf8')
const patterned = new Uint8Array(128 * 32 * 4)
for (let i = 0; i < patterned.length; i++) patterned[i] = (i * 37 + Math.floor(i / 4)) % 256

const opaqueAndTransparent = new Uint8Array([
  0, 0, 0, 255,
  255, 255, 255, 255,
  0, 0, 0, 0,
])
assert.deepEqual(Array.from(gray(opaqueAndTransparent)), [0, 1, 1])

const wasm = compile(source)
const runtime = instantiate(wasm)

function run(input) {
  // Validate before JZ marshaling, which does not preserve every typed-array kind.
  if (!(input instanceof Uint8Array) && !(input instanceof Uint8ClampedArray)) {
    throw new TypeError('Expected RGBA bytes')
  }
  const bytes = input instanceof Uint8ClampedArray
    ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
    : input
  try {
    return new Float32Array(runtime.exports.gray(bytes))
  } finally {
    runtime.memory.reset()
  }
}

const single = new Uint8Array([0, 0, 0, 255])
const clamped = new Uint8ClampedArray([
  255, 0, 0, 255, 0, 0, 0, 255, 0, 255, 0, 255,
]).subarray(4, 8)
const alphaRamp = new Uint8Array(256 * 4)
for (let alpha = 0; alpha < 256; alpha++) alphaRamp[alpha * 4 + 3] = alpha
let pixels = 0
let maxError = 0
const retained = []
for (const input of [single, single, opaqueAndTransparent, new Uint8Array(0), single, clamped, alphaRamp, patterned]) {
  const expected = gray(input)
  const before = input.slice()
  const actual = run(input)
  assert.equal(actual.length, expected.length)
  for (let i = 0; i < expected.length; i++) {
    assert(Number.isFinite(actual[i]))
    assert(actual[i] >= 0 && actual[i] <= 1)
    const error = Math.abs(actual[i] - expected[i])
    maxError = Math.max(maxError, error)
    assert(error <= 1e-6, `Pixel ${i}: JS/WASM difference ${error}`)
  }
  assert.deepEqual(input, before)
  retained.push([actual, actual.slice()])
  pixels += expected.length
}

for (const input of [null, undefined, [], 'rgba', {}, new Float32Array(4), new Uint16Array(4)]) {
  assert.throws(() => run(input), { name: 'TypeError', message: 'Expected RGBA bytes' })
}

for (const length of [1, 2, 3, 5, 6, 7]) {
  assert.throws(() => run(new Uint8Array(length)), {
    name: 'RangeError', message: 'Expected complete RGBA pixels',
  })
  assert.deepEqual(Array.from(run(single)), [0])
}
for (const [actual, snapshot] of retained) assert.deepEqual(actual, snapshot)

console.log(JSON.stringify({
  checks: 'JS/WASM parity, alpha, subarrays, repeated calls, empty input, errors, recovery, immutability',
  pixels,
  wasmBytes: wasm.byteLength,
  maxError,
  note: 'Compilation smoke check only; no speed or recognition measurement.',
}, null, 2))
