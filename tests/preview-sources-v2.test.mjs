import test from 'node:test'
import assert from 'node:assert/strict'
import { SELECTION, resolveFamilies } from '../scripts/preview-sources-v2.mjs'
import { providedTextAllowed, hasSearchMargin } from '../scripts/preview-provided.mjs'

const catalogue = () => SELECTION.map(([name], index) => ({
  id: `family-${index}`, name, slug: `font-${index}`, publisher: { name: 'Foundry' },
  styles: [{ id: `style-${index}`, weight: { name: 'Regular', number: 400 },
    is_variable: false, is_italic: false, file: `//example.invalid/font-${index}` }],
}))

test('second batch is source-qualified, disjoint, and covers all categories', () => {
  const families = resolveFamilies(catalogue(), new Set())
  assert.equal(families.length, 24)
  assert.equal(new Set(families.map(item => item.family)).size, 24)
  assert.deepEqual([...new Set(families.map(item => item.category))].sort(), ['display', 'mono', 'sans', 'serif', 'slab'])
  assert.equal(families[0].familyId, 'fontshare:family:family-0')
  assert.equal(families[0].faces[0].faceId, 'fontshare:style:style-0')
  assert.throws(() => resolveFamilies(catalogue(), new Set([SELECTION[0][0]])), /overlaps/)
})

test('no substitution for missing, variable, or italic Regular faces', () => {
  for (const mutate of [item => { item.styles = [] }, item => { item.styles[0].is_variable = true }, item => { item.styles[0].is_italic = true }]) {
    const items = catalogue()
    mutate(items[0])
    assert.throws(() => resolveFamilies(items, new Set()), /No verified static Regular/)
  }
})

test('provided text policy accepts phrases and rejects labels or short text', () => {
  const policy = { minChars: 8, excluded: ['Regular', 'Extra Bold'] }
  assert.equal(providedTextAllowed('Never hold grudges', policy), true)
  assert.equal(providedTextAllowed('Amsterdam', policy), true)
  assert.equal(providedTextAllowed('  EXTRA   BOLD ', policy), false)
  assert.equal(providedTextAllowed('Regular', policy), false)
  assert.equal(providedTextAllowed('Rome', policy), false)
  assert.equal(providedTextAllowed(null, policy), false)
  assert.equal(providedTextAllowed('a'.repeat(91), policy), false)
  assert.equal(providedTextAllowed('Regular'), true) // preserve the first pilot's recipe policy
})

test('a swash touching a search boundary cannot become an accepted text region', () => {
  const ink = { x: 180, y: 30, width: 320, height: 140 }
  assert.equal(hasSearchMargin(ink, { x: 180, y: 20, width: 350, height: 160 }), false)
  assert.equal(hasSearchMargin(ink, { x: 170, y: 30, width: 370, height: 170 }), false)
  assert.equal(hasSearchMargin(ink, { x: 170, y: 20, width: 330, height: 160 }), false)
  assert.equal(hasSearchMargin(ink, { x: 170, y: 20, width: 350, height: 150 }), false)
  assert.equal(hasSearchMargin(ink, { x: 150, y: 20, width: 400, height: 160 }), true)
})
