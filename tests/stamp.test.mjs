import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { pages, stamp } from '../scripts/stamp.mjs'

// A page whose stamps lag its files would mix deploys again: run `node scripts/stamp.mjs` after changing any of them.
test('every page pins its stylesheets and modules to their current content', async () => {
  for (const page of pages) {
    const html = await readFile(page, 'utf8')
    assert.equal(html, await stamp(html), `${page} is stale: run node scripts/stamp.mjs`)
    assert.doesNotMatch(html, /(href|src)="\.\/[^"?]+\.(css|mjs)"/, `${page} loads a local file without a stamp`)
  }
})

test('stamping replaces old stamps and the old import map instead of adding more', async () => {
  const html = await readFile('index.html', 'utf8'), stale = html.replaceAll(/\?v=\w+/g, '?v=0000000000')
  assert.equal(await stamp(stale), html)
})
