import assert from 'node:assert/strict'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { chromium } from 'playwright'
import { readCatalog, rankCatalog, embedWindows } from '../src/catalog.mjs'
import { readNetwork, inferCPU } from '../src/network.mjs'

const read = async p => JSON.parse(await readFile(p, 'utf8'))
const data = await read('dist/assets/catalog.json'), artifact = await read('dist/assets/model.json'), model = readNetwork(artifact)
const base = `http://127.0.0.1:${process.env.PORT || 4179}`
execFileSync(process.execPath, ['scripts/python.mjs', '-m', 'scripts.encoder_reference', 'models/encoder/encoder.json', '.data/encoder/runtime-reference.json'], { stdio: 'inherit' })
const reference = await read('.data/encoder/runtime-reference.json')
assert.equal(reference.encoderSha256, data.encoderSha256)
const error = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))
let cpuError = 0, gpuError = 0
for (const c of reference.cases) cpuError = Math.max(cpuError, error(inferCPU(model, { ...c, pixels: Float32Array.from(c.pixels) }), c.projection))
assert.ok(cpuError < 1e-4)
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
const issues = [], dir = '.data/catalog-demo-checks'
await mkdir(dir, { recursive: true })
async function result(page) {
  await page.waitForFunction(() => !document.querySelector('#save').disabled)
  return page.evaluate(() => {
    const original = HTMLAnchorElement.prototype.click; let value
    try { HTMLAnchorElement.prototype.click = function () { value = fetch(this.href).then(r => r.json()) }; document.querySelector('#save').click() }
    finally { HTMLAnchorElement.prototype.click = original }
    return value
  })
}
async function choose(page, id) {
  await page.locator('#catalog-button').click()
  await page.locator(`[data-catalog="${id}"]`).click()
  await page.waitForFunction(() => !document.querySelector('#catalog-button').hasAttribute('aria-busy'))
}
async function verify(page, value, option) {
  const catalog = readCatalog(await read(`dist/${option.file}`), data), expected = rankCatalog(value.embedding, catalog)
  assert.deepEqual(value.matches, expected)
  assert.equal(value.catalog.sha256, option.sha256); assert.equal(value.accepted, null); assert.equal(value.scoreType, 'cosine')
  assert.equal(value.matches.length, catalog.families)
  assert.equal(await page.locator('.result').count(), Math.min(5, catalog.families))
  assert.deepEqual(await page.locator('.result-score').allTextContents(), value.matches.slice(0, 5).map(m => m.score.toFixed(3)))
  const canvases = await page.locator('#normalized canvas').evaluateAll(cs => cs.map(c => ({ width: c.width, height: c.height, pixels: Array.from(c.getContext('2d').getImageData(0, 0, c.width, c.height).data).filter((_, i) => i % 4 === 0) })))
  assert.deepEqual(canvases, value.inputs.map(i => ({ width: i.width, height: i.height, pixels: i.pixels.map(p => Math.round(p * 255)) })))
  assert.equal(await page.locator('#detection-time').textContent(), `${value.milliseconds.toFixed(1)} ms`)
}
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.on('pageerror', e => issues.push(e.message))
  await page.goto(base); await page.waitForFunction(() => document.body.dataset.ready === 'true')
  const original = await result(page)
  assert.equal(original.backend, 'WebGPU'); await verify(page, original, data.catalogs[0])
  const outputs = await page.evaluate(async ({ artifact, cases }) => {
    const { readNetwork } = await import('./src/network.mjs'), { createNetworkGPU } = await import('./src/network-gpu.mjs')
    const gpu = await createNetworkGPU(readNetwork(artifact)), outputs = []
    try { for (const i of [0, 0, 1, 2, 3, 0]) outputs.push({ i, values: Array.from(await gpu.infer({ ...cases[i], pixels: Float32Array.from(cases[i].pixels) })) }) }
    finally { gpu.destroy() }
    return outputs
  }, { artifact, cases: reference.cases })
  for (const { i, values } of outputs) gpuError = Math.max(gpuError, error(values, reference.cases[i].projection))
  assert.ok(gpuError < 1e-4)
  const projections = original.inputs.map(i => inferCPU(model, { ...i, pixels: Float32Array.from(i.pixels) }))
  assert.ok(error(embedWindows(projections), original.embedding) < 1e-4)
  // Each shipped catalog is a real candidate set; A → A → B → A reuses pixels and embedding.
  for (const option of [...data.catalogs, data.catalogs[0]]) {
    await choose(page, option.id); const value = await result(page)
    await verify(page, value, option); assert.deepEqual(value.embedding, original.embedding); assert.deepEqual(value.inputs, original.inputs)
    assert.equal(value.cachedEmbedding, true)
    if (option.id.startsWith('previews')) {
      assert.equal(await page.locator('.result-preview').count(), 0)
      assert.equal(await page.locator('.reference-preview').count(), 5)
      await page.waitForFunction(() => [...document.querySelectorAll('.reference-preview')].every(i => i.complete && i.naturalWidth > 0))
    }
  }
  assert.deepEqual((await result(page)).matches, original.matches)
  // Failed imports preserve the selected catalog and current answer.
  for (const payload of ['{', '{}', JSON.stringify({ ...await read(`dist/${data.catalogs[0].file}`), encoderSha256: 'wrong' })]) {
    await page.locator('#catalog-file').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from(payload) })
    await page.waitForFunction(() => !document.querySelector('#catalog-error').hidden)
    assert.deepEqual((await result(page)).matches, original.matches)
  }
  const custom = await read(`dist/${data.catalogs[0].file}`)
  // Smallest compatible catalog, retaining one real vector and exact owner.
  custom.faces = custom.faces.slice(0, 1)
  custom.vectors = { ...custom.vectors, shape: [1, 128], data: Buffer.from(custom.vectors.data, 'base64').subarray(0, 128).toString('base64'), scales: custom.vectors.scales.slice(0, 1), owners: [0] }
  await page.locator('#catalog-file').setInputFiles({ name: 'myfonts.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(custom)) })
  await page.waitForFunction(() => document.querySelector('#catalog-label').textContent === 'myfonts.json')
  const imported = await result(page)
  assert.equal(imported.matches.length, 1); assert.equal(imported.matches[0].family, custom.faces[0].familyId)
  assert.equal(await page.locator('.result-preview, .reference-preview').count(), 0, 'Imported metadata must not claim a local specimen')
  await choose(page, data.catalogs[0].id)
  // A slow response cannot replace a later catalog choice.
  if (data.catalogs.length > 1) {
    let release, entered
    const gate = new Promise(r => { release = r }), started = new Promise(r => { entered = r })
    const slow = data.catalogs[1]
    await page.route(`**/${slow.file}`, async route => { entered(); await gate; await route.continue() })
    await page.locator('#catalog-button').click(); await page.locator(`[data-catalog="${slow.id}"]`).click(); await started
    await choose(page, data.catalogs[0].id); release()
    await page.waitForTimeout(100); await page.unroute(`**/${slow.file}`)
    assert.deepEqual((await result(page)).matches, original.matches)
  }
  // New input after a switch must infer again, and returning to A retains exact tensors.
  await page.locator('#resolution').selectOption('50'); const half = await result(page)
  assert.equal(half.resolution.scale, .5); assert.equal(half.cachedEmbedding, undefined)
  await page.locator('#resolution').selectOption('100'); assert.deepEqual((await result(page)).inputs, original.inputs)
  await page.locator('#image-frame').focus(); await page.keyboard.press('Shift+ArrowLeft')
  const cropped = await result(page); assert.equal(cropped.crop.width, original.crop.width - 1)
  await page.keyboard.press('Home'); assert.deepEqual((await result(page)).inputs, original.inputs)
  await page.locator('#clear').click(); await choose(page, data.catalogs.at(-1).id)
  assert.equal(await page.locator('.result, #normalized canvas').count(), 0); assert.ok(await page.locator('#save').isDisabled())
  await page.locator('#sample').click(); await page.locator('[data-font="lora"]').click()
  await verify(page, await result(page), data.catalogs.at(-1))
  // Corrupt images and blank pixels do not retain stale matches; replacement recovers.
  await page.locator('#file').setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('bad') })
  await page.waitForFunction(() => document.querySelector('#message').textContent.includes('Could not read'))
  assert.equal(await page.locator('.result').count(), 0)
  await page.locator('#sample').click(); await page.locator('[data-font="lora"]').click(); await result(page)
  await choose(page, data.catalogs[0].id)
  for (const width of [320, 375, 768, 1440]) {
    await page.setViewportSize({ width, height: 900 })
    await page.locator('#catalog-button').click(); await page.keyboard.press('End')
    assert.equal(await page.locator('#import-catalog').evaluate(e => e === document.activeElement), true)
    const rect = await page.locator('#catalog-menu').boundingBox()
    assert.ok(rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= width + 1 && rect.y + rect.height <= 901)
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    await page.screenshot({ path: `${dir}/selector-${width}.png`, fullPage: true })
    await page.keyboard.press('Escape'); assert.ok(await page.locator('#catalog-button').evaluate(e => e === document.activeElement))
  }
  await page.screenshot({ path: `${dir}/desktop.png`, fullPage: true })
  // An image may arrive after the encoder loads but before its first catalog.
  const sourcePng = Buffer.from((await page.locator('#source').evaluate(c => c.toDataURL())).split(',')[1], 'base64')
  const early = await browser.newPage(); let releaseInitial, enteredInitial
  const initialGate = new Promise(r => { releaseInitial = r }), initialStarted = new Promise(r => { enteredInitial = r })
  await early.route(`**/${data.catalogs[0].file}`, async route => { enteredInitial(); await initialGate; await route.continue() })
  await early.goto(base); await initialStarted
  await early.locator('#file').setInputFiles({ name: 'early.png', mimeType: 'image/png', buffer: sourcePng })
  await early.waitForFunction(() => !document.querySelector('#image-frame').hidden)
  await early.waitForTimeout(100)
  assert.equal(await early.locator('#message').textContent(), ''); assert.ok(await early.locator('#save').isDisabled())
  releaseInitial(); const earlyResult = await result(early)
  assert.equal(earlyResult.source.name, 'early.png'); assert.deepEqual(earlyResult.inputs, original.inputs)
  await early.close()
  await page.close()
  const cpu = await browser.newPage()
  await cpu.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }))
  await cpu.goto(base); const fallback = await result(cpu)
  assert.equal(fallback.backend, 'CPU'); assert.ok(error(fallback.embedding, original.embedding) < 1e-4)
  await cpu.close(); assert.deepEqual(issues, [])
  const report = { encoderSha256: data.encoderSha256, catalogs: data.catalogs.map(o => ({ id: o.id, sha256: o.sha256, families: o.families, faces: o.faces })), cpuError, gpuError,
    detectionMilliseconds: original.milliseconds, checks: ['native CPU/PyTorch/WebGPU projections', 'exact displayed tensors', 'catalog A → A → B → A', 'cached embedding reuse', 'one-face JSON import', 'invalid imports preserve result', 'stale catalog response', 'resolution and crop round trip', 'clear/switch/replacement', 'corrupt image recovery', 'keyboard and responsive selector', 'image before initial catalog', 'CPU fallback'],
    scope: 'Runtime correctness and UI lifecycle only; not recognition accuracy.' }
  await writeFile('bench/catalog-demo.json', JSON.stringify(report, null, 2) + '\n'); console.log(report)
} finally { await browser.close() }
