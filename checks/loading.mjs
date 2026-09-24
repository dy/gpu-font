// Startup: the model and first catalog download together behind a progress bar; every failure names its cause.
import assert from 'node:assert/strict'
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
  console.log('Loading: parallel downloads, byte-exact progress, catalog and model failures, altered catalog')
} finally { await browser.close() }
