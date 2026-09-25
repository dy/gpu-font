import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile, stat } from 'node:fs/promises'

test('site.json sizes the model and every catalog for download progress', async () => {
  const site = JSON.parse(await readFile('site.json', 'utf8'))
  assert.equal(site.modelBytes, (await stat(site.model)).size)
  for (const option of site.catalogs) assert.equal(option.bytes, (await stat(option.file)).size, option.id)
})

// A link searches one source of a grouped catalog alone (?catalog=collletttivo): the group names each source, each face
// says which source it came from, and no source id shadows a catalog id the link could also mean.
test('a grouped catalog names each source it holds, and every face comes from one of them', async () => {
  const site = JSON.parse(await readFile('site.json', 'utf8')), catalogIds = new Set(['all', 'my-fonts', ...site.catalogs.map(c => c.id)])
  for (const option of site.catalogs.filter(c => c.sources)) {
    const faces = JSON.parse(await readFile(option.file, 'utf8')).faces, ids = option.sources.map(s => s.id)
    assert.ok(option.sources.every(s => typeof s.name === 'string' && s.name), option.id)
    assert.deepEqual([...new Set(faces.map(f => f.sourceId))].toSorted(), ids.toSorted(), option.id)
    assert.deepEqual(ids.filter(id => catalogIds.has(id)), [], option.id)
  }
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

// Catalogs is private for now: reachable for My fonts, but out of the sitemap and search engines.
test('the sitemap lists the public page by its canonical address; Catalogs stays unlisted', async () => {
  const listed = [...(await readFile('sitemap.xml', 'utf8')).matchAll(/<loc>([^<]*)<\/loc>/g)].map(m => m[1])
  assert.deepEqual(listed, [link(await readFile('index.html', 'utf8'), 'canonical')])
  assert.equal(meta(await readFile('catalogs.html', 'utf8'), 'name', 'robots'), 'noindex')
})

test('per-script accuracy on the page comes from the shipped breakdown', async () => {
  const { metrics } = JSON.parse(await readFile('site.json', 'utf8')), { groups } = JSON.parse(await readFile('bench/style-breakdown.json', 'utf8'))
  assert.deepEqual([metrics.latinTop5, metrics.cyrillicTop5, metrics.arabicTop5], ['script/Latn', 'script/Cyrl', 'script/Arab'].map(key => groups[key].twin5))
})

// Favicons are stored with the site, one PNG per host (scripts/favicons.mjs): each a page names and each a catalog shows.
test('every favicon a page or catalog names is stored with the site', async () => {
  const site = JSON.parse(await readFile('site.json', 'utf8')), pages = await Promise.all(['index.html', 'catalogs.html'].map(page => readFile(page, 'utf8')))
  const named = [...new Set([...pages.flatMap(html => [...html.matchAll(/"\.\/(favicons\/[^"]+)"/g)].map(m => m[1])), ...site.catalogs.map(c => c.icon).filter(Boolean)])]
  assert.ok(named.length > site.catalogs.length)
  for (const path of named) {
    const bytes = await readFile(path)
    assert.ok(path.endsWith('.svg') ? bytes.toString('utf8').startsWith('<svg') : bytes.toString('latin1', 1, 4) === 'PNG', path)
  }
})

// Accuracy on pictures people bring is quoted only from reports of the shipped encoder: WhatFontIs-Bench's final part and
// the hand-cropped DaFont forum requests, both searched as the page's All.
test('photo and forum accuracy on the page come from reports of the shipped encoder', async () => {
  const site = JSON.parse(await readFile('site.json', 'utf8')), { metrics } = site
  const photos = JSON.parse(await readFile('bench/whatfontis.json', 'utf8')), requests = JSON.parse(await readFile('bench/dafont.json', 'utf8'))
  assert.deepEqual([photos.encoderSha256, requests.encoderSha256], [site.encoderSha256, site.encoderSha256])
  assert.deepEqual([metrics.photosTop1, metrics.photosTop5, metrics.photosCount], [photos.results.final.top1, photos.results.final.top5, photos.results.final.images])
  assert.deepEqual([metrics.requestsTop1, metrics.requestsTop5, metrics.requestsCount], [requests.gpuFont.top1, requests.gpuFont.top5, requests.gpuFont.requests])
})
