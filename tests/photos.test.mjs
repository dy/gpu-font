import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { rankOf, normalize, catalogNames } from '../scripts/photos.mjs'

const row = (name, siblings = []) => ({ family: normalize(name), name, siblings, score: 0.5 })
test('a photo counts at the first row that names the true family, or folds it in, without case or punctuation', () => {
  const rows = [row('Roboto Flex'), row('IBM Plex Sans', ['IBM Plex Sans KR']), row('Roboto')]
  assert.equal(rankOf(rows, 'Roboto'), 3)
  assert.equal(rankOf(rows, 'ibm plex sans kr'), 2)
  assert.equal(rankOf(rows, 'Roboto-Flex'), 1)
  assert.equal(rankOf(rows, 'Inter'), null)
  assert.equal(rankOf([], 'Inter'), null)
})
test('catalog names cover every shipped family, each in the catalog that ships it', async () => {
  const names = await catalogNames(), { catalogs } = await import('../src/match.mjs'), all = new Set()
  const seen = []
  for (const [id, url] of Object.entries(catalogs)) {
    for (const face of JSON.parse(await readFile(url)).faces) {
      const entry = names.get(normalize(face.family)); assert.ok(entry.catalogs.includes(id), face.family)
      assert.equal(entry.catalog, [...seen, id].find(c => entry.catalogs.includes(c)), face.family); all.add(normalize(face.family))
    }
    seen.push(id)
  }
  assert.equal(names.size, all.size); assert.ok([...names.values()].some(e => e.catalogs.length > 1), 'some family name is in more than one catalog')
})
test('the DaFont crops are integer boxes inside their pictures, and every skip gives its reason', async () => {
  const { rule, crops, skipped } = JSON.parse(await readFile(new URL('../bench/dafont-crops.json', import.meta.url)))
  assert.ok(rule.length > 100)
  for (const box of Object.values(crops)) assert.ok(box.length === 4 && box.every(Number.isInteger) && box[0] >= 0 && box[1] >= 0 && box[2] > box[0] && box[3] > box[1])
  for (const [id, reason] of Object.entries(skipped)) assert.ok(!crops[id] && reason.length > 20)
})
test('bench/finders.json has a place for exactly the finders the page compares, in its order, and gpu-font answers every crop', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8'), table = html.match(/<section id="compare"[\s\S]*?<\/section>/)[0]
  const page = [...table.matchAll(/<th scope="row"><a [^>]*>([^<]+)<\/a>/g)].map(m => m[1])
  const { finders, images } = JSON.parse(await readFile(new URL('../bench/finders.json', import.meta.url)))
  assert.ok(page.length >= 2); assert.deepEqual(Object.keys(finders), page)
  for (const finder of Object.values(finders)) assert.ok(finder.how.length > 10)
  assert.deepEqual(Object.keys(finders['gpu-font'].answers), images.map(i => i.id))
})
