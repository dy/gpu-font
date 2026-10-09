import test from 'node:test'
import assert from 'node:assert/strict'
import { threadRole, lineOf, padded, previewText, indexed, summary } from '../scripts/forum.mjs'
import { normalize } from '../scripts/photos.mjs'

const names = new Map(['Lobster', 'Montserrat', 'Verve'].map(family => [normalize(family), { family, catalog: 'dafont', familyId: `dafont/${normalize(family)}` }]))
const line = (text, box, confidence = 1) => ({ text, box, confidence })
const request = (id, name = 'Verve', url = 'https://www.dafont.com/verve.font?text=Lovesick+Girls') => ({ id, identified: [{ name, url, by: 'x' }], files: [{ file: `${id}-0.png`, width: 400, height: 200 }] })

test('a thread is held out by the hash of its id, one in ten, the same rule as train.style_forum', () => {
  assert.deepEqual([603084, 603080, 464694, 384803].map(threadRole), ['development', 'development', 'held-out', 'development'])
  const held = Array.from({ length: 2000 }, (_, i) => 400000 + i).filter(id => threadRole(id) === 'held-out').length
  assert.ok(held > 150 && held < 250, `${held} of 2000 held out`)
})

test('the line is the one the answerer previewed, else the tallest confident horizontal lettered one', () => {
  const lines = [line('ADMITS ONE', [43, 42, 58, 126]), line('GIRLS', [96, 74, 255, 139]), line('LOVESICK', [88, 29, 251, 72]), line('Honey', [0, 0, 300, 200], .4), line('', [0, 0, 300, 100])]
  assert.equal(lineOf(lines, 'Lovesick Girls').text, 'GIRLS')  // both share a word; the first of the two, which Vision lists largest first
  assert.equal(lineOf(lines, 'lovesick').text, 'LOVESICK')
  assert.equal(lineOf(lines, '').text, 'GIRLS')
  assert.equal(lineOf([line('x', [0, 0, 10, 40]), line('y', [0, 0, 40, 10], .2), line('z', [5, 5, 5, 5])]), null)  // upright, unsure, no size: none is a line
  assert.deepEqual(padded([100, 50, 300, 90]), [90, 40, 310, 100])
  assert.deepEqual([previewText('https://www.dafont.com/verve.font?text=Lovesick+Girls'), previewText('https://fonts.google.com/specimen/Lobster'), previewText('')], ['Lovesick Girls', '', ''])
})

test('held-out threads with a boxed line and an indexed answer are read; the rest are counted by why they were left', () => {
  // 464694 and 6000xx are held out by the hash; 603084 is development
  const boxes = new Map([['464694-0.png', { lines: [line('Fresh bread', [10, 10, 200, 60])] }], ['600021-0.png', { lines: null }], ['600039-0.png', { lines: [line('Fresh bread', [10, 10, 200, 60])] }], ['600042-0.png', { lines: [line('x', [0, 0, 10, 40])] }]])
  const { kept, left } = indexed([request(464694), request(603084), request(600021), request(600039, 'Gotham'), request(600042), request(600045), { id: 1, identified: [], files: [] }], boxes, names)
  assert.deepEqual(kept.map(k => [k.request.id, k.truth.family, k.line.text]), [[464694, 'Verve', 'Fresh bread']])
  assert.deepEqual(left, { development: 1, unboxed: 2, 'no line': 1, 'not indexed': 1 })
})

test('accuracy is given over threads and over families, each family once', () => {
  const s = summary([{ truth: { family: 'Verve' }, rank: 1 }, { truth: { family: 'Verve' }, rank: 3 }, { truth: { family: 'Lobster' }, rank: null }])
  assert.deepEqual([s.threads, s.families, s.top1, s.top5, s.byFamily], [3, 2, 0.3333, 0.6667, { top1: 0.25, top5: 0.5 }])
})
