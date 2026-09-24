import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile, stat } from 'node:fs/promises'

test('site.json sizes the model and every catalog for download progress', async () => {
  const site = JSON.parse(await readFile('site.json', 'utf8'))
  assert.equal(site.modelBytes, (await stat(site.model)).size)
  for (const option of site.catalogs) assert.equal(option.bytes, (await stat(option.file)).size, option.id)
})

const meta = (html, attribute, key) => html.match(new RegExp(`<meta ${attribute}="${key}" content="([^"]*)">`))?.[1]
test('each page describes itself once: the link preview repeats the search description', async () => {
  for (const page of ['index.html', 'catalogs.html']) {
    const html = await readFile(page, 'utf8'), description = meta(html, 'name', 'description')
    assert.ok(description, page); assert.equal(meta(html, 'property', 'og:description'), description, page)
  }
})

// Other is a bucket of small sources, not a name to search by.
test('the search description names the five largest shipped catalogs', async () => {
  const site = JSON.parse(await readFile('site.json', 'utf8')), description = meta(await readFile('index.html', 'utf8'), 'name', 'description')
  for (const { name } of site.catalogs.filter(c => c.id !== 'other').toSorted((a, b) => b.families - a.families).slice(0, 5)) assert.ok(description.includes(name.split(/[ .]/)[0]), name)
})

// Link previews: every page shows og.png, a 1200 × 630 PNG served from the site, as a large card with alt text.
const link = (html, rel) => html.match(new RegExp(`<link rel="${rel}" href="([^"]*)">`))?.[1]
test('each page previews as a large card of the committed 1200 × 630 og.png', async () => {
  const png = await readFile('og.png'), size = [png.readUInt32BE(16), png.readUInt32BE(20)]
  assert.equal(png.toString('latin1', 1, 4), 'PNG'); assert.deepEqual(size, [1200, 630])
  for (const page of ['index.html', 'catalogs.html']) {
    const html = await readFile(page, 'utf8'), home = new URL('./', link(html, 'canonical'))
    assert.equal(meta(html, 'property', 'og:image'), new URL('og.png', home).href, page)
    assert.deepEqual(['og:image:width', 'og:image:height'].map(key => Number(meta(html, 'property', key))), size, page)
    assert.ok(meta(html, 'property', 'og:image:alt'), page); assert.equal(meta(html, 'name', 'twitter:card'), 'summary_large_image', page)
    assert.equal(meta(html, 'property', 'og:url'), link(html, 'canonical'), page)
  }
})

test('the structured data repeats the page’s own address, description and preview', async () => {
  const html = await readFile('index.html', 'utf8'), data = JSON.parse(html.match(/<script type="application\/ld\+json">(.*?)<\/script>/s)[1])
  assert.equal(data['@type'], 'WebApplication'); assert.equal(data.url, link(html, 'canonical'))
  assert.equal(data.description, meta(html, 'name', 'description')); assert.equal(data.image, meta(html, 'property', 'og:image'))
})

test('the sitemap lists every page by its canonical address', async () => {
  const listed = [...(await readFile('sitemap.xml', 'utf8')).matchAll(/<loc>([^<]*)<\/loc>/g)].map(m => m[1])
  assert.deepEqual(listed, await Promise.all(['index.html', 'catalogs.html'].map(async page => link(await readFile(page, 'utf8'), 'canonical'))))
})
