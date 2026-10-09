import test from 'node:test'
import assert from 'node:assert/strict'
import { readable, distance, sample, summary } from '../scripts/sametext.mjs'

test('a label reads as the reader could read it: accents to their letters, other characters dropped, ends trimmed', () => {
  assert.deepEqual(['HUGGING', 'café crème', 'Rock & Roll!', ' naïve ', '東京 2026', ''].map(readable), ['HUGGING', 'cafe creme', 'Rock  Roll', 'naive', '2026', ''])
})

test('edit distance', () => {
  assert.deepEqual([['', ''], ['abc', 'abc'], ['abc', 'abd'], ['abc', 'ab'], ['', 'abc'], ['kitten', 'sitting']].map(([a, b]) => distance(a, b)), [0, 0, 1, 1, 3, 3])
})

test('a seeded sample is the same every time, holds each item once, and is all of a short list', () => {
  const items = Array.from({ length: 50 }, (_, i) => i)
  const a = sample(items, 10, 3), b = sample(items, 10, 3)
  assert.deepEqual(a, b); assert.equal(new Set(a).size, 10); assert.ok(a.every(i => items.includes(i)))
  assert.notDeepEqual(sample(items, 10, 4), a)
  assert.deepEqual(sample([1, 2, 3], 10).sort(), [1, 2, 3])
})

test('a summary counts photographs and families', () => {
  const s = summary([{ truth: { family: 'Asap' }, rank: 1 }, { truth: { family: 'Asap' }, rank: null }, { truth: { family: 'Lora' }, rank: 4 }])
  assert.deepEqual([s.photographs, s.families, s.top1, s.top5], [3, 2, 0.3333, 0.6667])
})
