import test from 'node:test'
import assert from 'node:assert/strict'
import { familyOf, indexed, summary, siteRole } from '../scripts/screens.mjs'
import { normalize } from '../scripts/photos.mjs'

const names = new Map(['Roboto', 'Montserrat', 'Google Sans', 'Roboto Condensed', 'Inter'].map(family => [normalize(family), { family, catalog: 'google-fonts' }]))
const record = (family, capturedAt = '2026-09-25T10:00:00Z') => ({ capturedAt, origin: 'https://a.test', font: { family }, css: { fontSize: 16 }, text: 'Quiet rivers' })

test('a drawn font belongs to its own family before the family its style words leave', () => {
  assert.equal(familyOf('Roboto', names).family, 'Roboto')
  assert.equal(familyOf('Roboto Condensed', names).family, 'Roboto Condensed')  // indexed by that name: nothing is cut
  for (const [drawn, family] of [['Roboto Medium', 'Roboto'], ['Montserrat Thin', 'Montserrat'], ['Montserrat ExtraLight Italic', 'Montserrat'], ['Google Sans 18pt', 'Google Sans'], ['Inter Semi Bold', 'Inter'], ['Roboto 500', 'Roboto']])
    assert.equal(familyOf(drawn, names).family, family, drawn)
  assert.equal(familyOf('Medium', names), null)       // style words alone name no family
  assert.equal(familyOf('Helvetica Neue', names), null)
})

test('empty and garbled names are left, never matched', () => {
  for (const name of ['', '.\x7f', '\x01\x02', '   ', undefined]) assert.equal(familyOf(name, names), null, JSON.stringify(name))
})

test('records are counted by why they were left, and a cut in time leaves the later ones', () => {
  const records = [record('Roboto'), record('Roboto Medium'), record(''), record('.\x7f'), record('Helvetica'), record('Inter', '2026-09-26T10:00:00Z')]
  const { kept, left } = indexed(records, names, Date.parse('2026-09-25T23:59:59Z'))
  assert.deepEqual(kept.map(k => k.truth.family), ['Roboto', 'Roboto'])
  assert.deepEqual(left, { later: 1, unnamed: 2, 'not indexed': 1 })
  assert.deepEqual(indexed([], names), { kept: [], left: { later: 0, unnamed: 0, 'not indexed': 0 } })
  assert.equal(indexed(records, names).kept.length, 3)  // no cut: every record is read
})

test('accuracy is given over crops and over families, each family once', () => {
  const crop = (family, rank) => ({ truth: { family }, record: { origin: 'https://a.test' }, rank })
  const s = summary([crop('Arial', 1), crop('Arial', 1), crop('Arial', 3), crop('Arial', 1), crop('Inter', null), crop('Inter', 7)])
  assert.deepEqual([s.crops, s.families, s.top1, s.top5, s.top20], [6, 2, 0.5, 0.6667, 0.8333])
  assert.deepEqual(s.byFamily, { top1: 0.375, top5: 0.5 })
  assert.deepEqual(summary([]), { crops: 0, families: 0, sites: 0, top1: null, top5: null, top20: null, byFamily: { top1: null, top5: null } })
})

test('a site is held out by the hash of its origin, one in ten, the same rule as train.style_screens', () => {
  assert.deepEqual(['https://www.chess.com', 'https://m.youtube.com', 'https://site3.test', 'https://site13.test'].map(siteRole), ['development', 'held-out', 'development', 'held-out'])
  const held = Array.from({ length: 2000 }, (_, i) => `https://site${i}.test`).filter(o => siteRole(o) === 'held-out').length
  assert.ok(held > 150 && held < 250, `${held} of 2000 held out`)
  assert.equal(indexed([record('Roboto')], names).kept[0].role, 'development')
})
