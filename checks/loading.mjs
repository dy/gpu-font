// Startup: the model and first catalog download together behind a progress bar; every failure names its cause.
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { chromium } from 'playwright'

const data = JSON.parse(await readFile('site.json', 'utf8')), catalog = data.catalogs[0]
const base = `http://127.0.0.1:${process.env.PORT || 4179}`
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
// While loading, the matcher shows only its centred download; its content keeps its place, hidden.
const bar = page => page.evaluate(() => { const e = document.querySelector('#loading'), loader = document.querySelector('.loader')
  return { busy: document.querySelector('#workbench').getAttribute('aria-busy'), content: getComputedStyle(document.querySelector('.source-panel')).visibility,
    loader: getComputedStyle(loader).display === 'none' ? null : loader.textContent, value: e.value, max: e.max } })
async function open({ file, handle }) {
  const page = await browser.newPage(), errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route(`**/${file}`, handle)
  await page.goto(base)
  return { page, errors }
}
try {
  // Both downloads start before either finishes; the bar counts every byte site.json sizes, then hides.
  let release, requested = false
  const gate = new Promise(r => { release = r })
  const held = await open({ file: catalog.file, handle: async route => { requested = true; await gate; await route.continue() } })
  await held.page.waitForFunction(() => document.querySelector('#loading').value > 0)
  assert.ok(requested, 'The catalog is requested while the model downloads')
  await held.page.waitForTimeout(300)
  const waiting = await bar(held.page)
  assert.equal(waiting.busy, 'true'); assert.equal(waiting.content, 'hidden', 'The loader stands alone while the catalog is outstanding')
  assert.match(waiting.loader, /^Loading the model \d{1,2}%$/, 'The loader names the wait and counts it, short of 100%')
  release(); await held.page.waitForSelector('body[data-ready="true"]')
  assert.deepEqual(await bar(held.page), { busy: null, content: 'visible', loader: null, value: data.modelBytes + catalog.bytes, max: data.modelBytes + catalog.bytes })
  assert.equal(await held.page.locator('#loading-percent').textContent(), '100%')
  await held.page.waitForFunction(() => document.querySelectorAll('.result').length === 5)
  assert.deepEqual(held.errors, []); await held.page.close()
  // A failed or altered download stops loading with its own message and no unhandled rejection, even with the other still in flight.
  for (const [file, response, expected] of [[catalog.file, { status: 404, body: '' }, 'Could not load the initial catalog.'],
    [data.model, { status: 404, body: '' }, `Could not load ${data.model}.`],
    [catalog.file, { status: 200, contentType: 'application/json', body: '{}' }, 'Catalog checksum failed.']]) {
    const failed = await open({ file, handle: route => route.fulfill(response) })
    await failed.page.waitForFunction(() => document.querySelector('#backend').textContent === 'Model unavailable')
    await failed.page.waitForTimeout(300)
    assert.equal(await failed.page.locator('#message').textContent(), expected)
    assert.deepEqual([(await bar(failed.page)).loader, await failed.page.locator('#message').isVisible()], [null, true], 'The failure shows in place of the loader'); assert.equal(await failed.page.locator('body').getAttribute('data-ready'), null)
    assert.deepEqual(failed.errors, [], expected); await failed.page.close()
  }
  // A later visit reads the model and catalog kept under their checksums. Once site.json names a new checksum, that file
  // alone downloads again and its old copy is dropped; kept bytes that no longer match their checksum download again.
  const context = await browser.newContext(), visit = async (routes = {}) => {
    const page = await context.newPage(), files = new Set(), errors = []
    page.on('request', request => files.add(new URL(request.url()).pathname.slice(1))); page.on('pageerror', error => errors.push(error.message))
    for (const [file, body] of Object.entries(routes)) await page.route(`**/${file}`, route => route.fulfill({ contentType: 'application/json', body }))
    await page.goto(base); await page.waitForSelector('body[data-ready="true"]'); assert.deepEqual(errors, [])
    return { page, files }
  }
  const kept = async (page, ...wanted) => { for (let tries = 0; ; tries++) {
    const keys = await page.evaluate(async () => (await (await caches.open('gpu-font')).keys()).map(r => { const u = new URL(r.url); return `${u.pathname.slice(1)}?${u.search.slice(1)}` }))
    if (tries > 50 || wanted.every(key => keys.includes(key))) return keys
    await page.waitForTimeout(100)
  } }
  const first = await visit(); assert.ok(first.files.has(data.model) && first.files.has(catalog.file)); await kept(first.page, `${data.model}?${data.modelSha256}`, `${catalog.file}?${catalog.sha256}`)
  const later = await visit(); assert.deepEqual([later.files.has(data.model), later.files.has(catalog.file)], [false, false], 'A later visit downloads neither')
  const changed = Buffer.concat([await readFile(catalog.file), Buffer.from('\n')]), checksum = createHash('sha256').update(changed).digest('hex')
  const updated = await visit({ 'site.json': JSON.stringify({ ...data, catalogs: [{ ...catalog, sha256: checksum, bytes: changed.length }, ...data.catalogs.slice(1)] }), [catalog.file]: changed })
  assert.deepEqual([updated.files.has(data.model), updated.files.has(catalog.file)], [false, true], 'A new checksum downloads that file alone')
  const keys = await kept(updated.page, `${catalog.file}?${checksum}`); assert.deepEqual([keys.includes(`${catalog.file}?${checksum}`), keys.includes(`${catalog.file}?${catalog.sha256}`)], [true, false], 'The new copy replaces the old')
  await updated.page.evaluate(async key => (await caches.open('gpu-font')).put(key, new Response('{}')), `${data.model}?${data.modelSha256}`)
  assert.ok((await visit()).files.has(data.model), 'Kept bytes that fail their checksum download again')
  await context.close()
  // Where the browser has no store or refuses one, the page loads as before and the catalogs wait in its HTTP cache.
  for (const [label, refuse] of [['No store', () => Object.defineProperty(window, 'caches', { value: undefined })],
    ['A refused store', () => { CacheStorage.prototype.open = () => Promise.reject(new DOMException('', 'SecurityError')) }]]) {
    const bare = await browser.newContext(), page = await bare.newPage(), files = new Set(), errors = []
    page.on('request', request => files.add(new URL(request.url()).pathname.slice(1))); page.on('pageerror', error => errors.push(error.message))
    await page.addInitScript(refuse); await page.goto(base); await page.waitForFunction(() => document.querySelectorAll('.result').length === 5)
    await page.evaluate(() => new Promise(resolve => requestIdleCallback(() => setTimeout(resolve, 500))))
    assert.deepEqual([errors, data.catalogs.filter(c => !files.has(c.file)).map(c => c.id)], [[], []], label); await bare.close()
  }
  console.log('Loading: parallel downloads, byte-exact progress, catalog and model failures, altered catalog, kept files by checksum, no store')
} finally { await browser.close() }
