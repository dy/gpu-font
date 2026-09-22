import assert from 'node:assert/strict'
import { test } from 'node:test'
import { gray } from '../bench/gray.mjs'

test('gray accepts an empty byte array and a single complete pixel', () => {
  assert.deepEqual(gray(new Uint8Array(0)), new Float32Array(0))
  assert.deepEqual(gray(new Uint8Array([0, 0, 0, 255])), new Float32Array([0]))
})

test('gray rejects null and missing input', () => {
  for (const input of [null, undefined]) {
    assert.throws(() => gray(input), TypeError)
  }
})

test('gray rejects incomplete pixels before and after a complete pixel', () => {
  for (const length of [1, 2, 3, 5, 6, 7]) {
    assert.throws(() => gray(new Uint8Array(length)), {
      name: 'RangeError', message: 'Expected complete RGBA pixels',
    })
  }
})

test('gray preserves fractional edge intensities and RGB channel contributions', () => {
  const input = new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 0, 255, 255,
    128, 128, 128, 255,
  ])
  const actual = gray(input)
  for (const [i, expected] of [0.2126, 0.7152, 0.0722, 128 / 255].entries()) {
    assert(Math.abs(actual[i] - expected) < 1e-6)
  }
})

test('gray composites all alpha values to white without losing coverage', () => {
  const input = new Uint8Array(256 * 4)
  for (let alpha = 0; alpha < 256; alpha++) input[alpha * 4 + 3] = alpha
  const actual = gray(input)
  for (let alpha = 0; alpha < 256; alpha++) {
    assert(Math.abs(actual[alpha] - (255 - alpha) / 255) < 1e-6)
    assert(actual[alpha] >= 0 && actual[alpha] <= 1)
  }
  assert.deepEqual(gray(new Uint8Array([170, 30, 200, 0])), new Float32Array([1]))
})

test('gray respects a clamped subarray offset and does not mutate its backing array', () => {
  const backing = new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 0, 255, 0, 255, 0, 255])
  const before = backing.slice()
  assert.deepEqual(gray(backing.subarray(4, 8)), new Float32Array([0]))
  assert.deepEqual(backing, before)
})

test('gray keeps independent outputs through A → A → B → empty → A calls', () => {
  const a = new Uint8Array([0, 0, 0, 255])
  const b = new Uint8Array([255, 255, 255, 255])
  const first = gray(a)
  const second = gray(a)
  assert.notEqual(first, second)
  assert.deepEqual(second, first)
  assert.deepEqual(gray(b), new Float32Array([1]))
  assert.deepEqual(gray(new Uint8Array(0)), new Float32Array(0))
  assert.deepEqual(gray(a), first)
  first[0] = 1
  assert.deepEqual(second, new Float32Array([0]))
  assert.deepEqual(a, new Uint8Array([0, 0, 0, 255]))
})
