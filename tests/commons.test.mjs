import test from 'node:test'
import assert from 'node:assert/strict'
import { line, indexed, summary } from '../scripts/commons.mjs'
import { normalize, accuracy } from '../scripts/photos.mjs'

const names = new Map(['Futura', 'Helvetica'].map(family => [normalize(family), { family, catalog: 'adobe-fonts' }]))
const found = (text, height, confidence = 1) => ({ box: [10, 20, 200, 20 + height], text, confidence })

test('a photo is read by its tallest confident line of three characters or more', () => {
  assert.equal(line({ lines: [found('small print', 12), found('EXIT', 80), found('Way out', 40)] }).text, 'EXIT')
  assert.equal(line({ lines: [found('EXIT', 80, .3), found('Way out', 40, .5)] }).text, 'Way out')  // a doubtful line is passed over
  assert.equal(line({ lines: [found('A 1', 90), found('Way', 40)] }).text, 'Way')  // spaces are not characters
  assert.equal(line({ lines: [found('Way', 40), found('Out', 40)] }).text, 'Way')  // a tie keeps the first found
  assert.equal(line({ lines: [found('ЗАГС', 60)] }).text, 'ЗАГС')
})

test('a photo without such a line is not read', () => {
  for (const lines of [[], [found('OK', 90)], [found('EXIT', 80, .3)]]) assert.equal(line({ lines }), null)
})

test('photos are counted by why they were left', () => {
  const photos = [{ typeface: 'Futura', lines: [found('Shell', 50)] }, { typeface: 'Futura', lines: [] }, { typeface: 'Clearview', lines: [found('Exit 12', 50)] }, { typeface: 'HELVETICA', lines: [found('Subway', 50)] }]
  const { kept, left } = indexed(photos, names)
  assert.deepEqual(kept.map(k => [k.truth.family, k.line.text]), [['Futura', 'Shell'], ['Helvetica', 'Subway']])
  assert.deepEqual(left, { 'not indexed': 1, 'no line': 1 })
  assert.deepEqual(indexed([], names), { kept: [], left: { 'not indexed': 0, 'no line': 0 } })
})

test('accuracy counts photos, and typefaces once each; nothing scored gives no share', () => {
  const photo = (family, rank) => ({ truth: { family }, rank })
  const s = summary([photo('Futura', 1), photo('Futura', 5), photo('Futura', 6), photo('Helvetica', null)])
  assert.deepEqual(s, { photos: 4, typefaces: 2, top1: 0.25, top5: 0.5, top20: 0.75, byFamily: { top1: 0.1667, top5: 0.3333 } })
  assert.deepEqual(accuracy([]), { families: 0, top1: null, top5: null, top20: null, byFamily: { top1: null, top5: null } })
})
