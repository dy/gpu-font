import assert from 'node:assert/strict'
import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { readNetwork, inferCPU, rankWindows } from '../src/network.mjs'

const bytes = await readFile('dist/assets/model.json'), artifact = JSON.parse(bytes)
const reference = JSON.parse(await readFile('.data/demo-reference.json'))
assert.equal(createHash('sha256').update(bytes).digest('hex'), reference.modelSha256)
const model = readNetwork(artifact)
const catalog = JSON.parse(await readFile('dist/assets/catalog.json'))
const temperature = catalog.calibration?.temperature ?? 1
const base = `http://127.0.0.1:${process.env.PORT || 4179}`
assert.equal((await fetch(base, { method: 'HEAD' })).status, 200)
assert.equal((await fetch(base, { method: 'POST' })).status, 405)
assert.equal((await fetch(`${base}/..%2fpackage.json`)).status, 403)
assert.equal((await fetch(`${base}/%ZZ`)).status, 404)
const maxError = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))
async function saveResult(page) {
  await page.waitForFunction(() => !document.querySelector('#save').disabled)
  const downloading = page.waitForEvent('download')
  await page.locator('#save').evaluate(button => button.click())
  const download = await downloading
  assert.equal(download.suggestedFilename(), 'gpu-font-result.json')
  const result = JSON.parse(await readFile(await download.path(), 'utf8'))
  // Display the full classifier distribution, never a re-normalized top five.
  const scores = await page.locator('.result-score').evaluateAll(elements => elements.map(el => ({ text: el.textContent, label: el.getAttribute('aria-label') })))
  assert.equal(scores.length, 5)
  for (const [i, score] of scores.entries()) {
    const actual = result.matches[i].score
    assert.equal(score.label, `Model score: ${score.text}`)
    if (score.text === '<0.1%') assert.ok(actual < .001)
    else if (score.text === '>99.9%') assert.ok(actual > .999)
    else {
      assert.match(score.text, /^\d+\.\d%$/)
      assert.ok(Math.abs(parseFloat(score.text) / 100 - actual) <= .000500001)
    }
  }
  assert.equal(await page.locator('#result-summary').textContent() === 'Below threshold', !result.accepted)
  return result
}
async function chooseSample(page, id) {
  await page.locator('#sample').click()
  await page.locator(`#sample-list [data-font="${id}"]`).click()
}
async function assertInput(page, result) {
  assert.ok(await page.locator('#model-input').isVisible())
  const canvases = await page.locator('#normalized canvas').evaluateAll(elements => elements.map(canvas => ({ width: canvas.width, height: canvas.height,
    cssWidth: canvas.getBoundingClientRect().width, data: Array.from(canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data) })))
  assert.equal(canvases.length, result.inputs.length)
  for (const [n, input] of result.inputs.entries()) {
    const canvas = canvases[n]
    assert.equal(canvas.width, input.width); assert.equal(canvas.height, input.height)
    assert.ok(canvas.cssWidth <= input.width * 2 + 1, 'Preview must not stretch into an empty panel')
    assert.equal(input.pixels.length, input.width * input.height)
    for (const [i, value] of input.pixels.entries()) {
      const byte = Math.round(value * 255)
      assert.deepEqual(canvas.data.slice(i * 4, i * 4 + 4), [byte, byte, byte, 255])
    }
    const { width: w, height: h, pixels } = input
    for (const edge of [pixels.slice(0, w), pixels.slice(-w), Array.from({ length: h }, (_, y) => pixels[y * w]), Array.from({ length: h }, (_, y) => pixels[y * w + w - 1])]) {
      assert.ok(edge.some(v => v < 1), 'The tensor itself must have no all-white outer border')
    }
  }
}
let cpuError = 0
for (const c of reference.cases) {
  const vector = inferCPU(model, { ...c, pixels: Float32Array.from(c.pixels) })
  cpuError = Math.max(cpuError, maxError(vector, c.logits))
  assert.deepEqual(rankWindows([vector], model.fonts).map(m => m.family), rankWindows([c.logits], model.fonts).map(m => m.family))
}
assert.ok(cpuError < 1e-4, `CPU/PyTorch error ${cpuError}`)
console.log(`CPU/PyTorch parity: ${cpuError}`)

const browser = await chromium.launch({ channel: 'chromium', headless: true, args: ['--enable-unsafe-webgpu'] })
const errors = []
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } })
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(base)
  assert.match(await page.title(), /gpu-font/, 'The test port must serve gpu-font')
  await page.waitForSelector('body[data-ready="true"]').catch(async error => {
    console.error({ errors, page: await page.locator('body').innerText() })
    throw error
  })
  await page.waitForFunction(() => document.querySelectorAll('.result').length === 5)
  assert.equal(await page.locator('#backend').getAttribute('data-backend'), 'webgpu', 'Actual WebGPU is required for this integration test')
  const gpuResult = await page.evaluate(async ({ artifact, cases }) => {
    const { readNetwork, rankWindows } = await import('/src/network.mjs')
    const { createNetworkGPU } = await import('/src/network-gpu.mjs')
    const model = readNetwork(artifact), gpu = await createNetworkGPU(model)
    const outputs = []
    try {
      for (const c of cases) outputs.push(Array.from(await gpu.infer({ ...c, pixels: Float32Array.from(c.pixels) })))
      const a = { ...cases[0], pixels: Float32Array.from(cases[0].pixels) }, b = { ...cases[2], pixels: Float32Array.from(cases[2].pixels) }
      const pending = [gpu.infer(a), gpu.infer(a), gpu.infer(b), gpu.infer(a)]
      a.pixels.fill(.25) // Pending calls must own their inputs.
      const repeated = await Promise.all(pending)
      let invalid = false
      try { await gpu.infer({ width: 1, height: 1, pixels: new Float32Array(0) }) } catch { invalid = true }
      const recovered = await gpu.infer({ ...cases[0], pixels: Float32Array.from(cases[0].pixels) })
      gpu.destroy()
      let destroyed = false
      try { await gpu.infer(b) } catch { destroyed = true }
      return { outputs, repeated: repeated.map(v => Array.from(v)), recovered: Array.from(recovered), invalid, destroyed, info: gpu.info,
        rankings: outputs.map(v => rankWindows([v], model.fonts).map(m => m.family)) }
    } finally { gpu.destroy() }
  }, { artifact, cases: reference.cases })
  let gpuError = 0
  reference.cases.forEach((c, i) => {
    gpuError = Math.max(gpuError, maxError(gpuResult.outputs[i], c.logits))
    assert.deepEqual(gpuResult.rankings[i], rankWindows([c.logits], model.fonts).map(m => m.family))
  })
  assert.ok(gpuError < 1e-4, `WebGPU/PyTorch error ${gpuError}`)
  for (const [i, c] of [0, 0, 2, 0].entries()) assert.ok(maxError(gpuResult.repeated[i], reference.cases[c].logits) < 1e-4)
  assert.ok(gpuResult.invalid && gpuResult.destroyed)
  assert.ok(maxError(gpuResult.recovered, reference.cases[0].logits) < 1e-4)
  console.log(`WebGPU/PyTorch parity: ${gpuError}; adapter: ${gpuResult.info}`)

  await mkdir('.data/demo-checks', { recursive: true })
  await page.screenshot({ path: '.data/demo-checks/desktop.png', fullPage: true })
  const png = await page.locator('#source').evaluate(canvas => canvas.toDataURL('image/png').split(',')[1])
  const initialResult = await saveResult(page)
  await assertInput(page, initialResult)
  assert.equal(initialResult.accepted, initialResult.matches[0].score >= (catalog.calibration?.threshold ?? 1.01))

  // Exact score boundaries cannot be selected reliably with trained-model logits.
  // Substitute only ranking on an isolated page; exercise the real UI/export lifecycle.
  const scorePage = await browser.newPage()
  scorePage.on('pageerror', error => errors.push(error.message))
  const networkSource = await readFile('dist/src/network.mjs', 'utf8')
  await scorePage.route('**/src/network.mjs', route => route.fulfill({ contentType: 'text/javascript', body:
    networkSource.replace('export function rankWindows(', 'function originalRankWindows(') + '\nexport function rankWindows(logits, fonts, temperature) { return globalThis.testScores ? fonts.map((family, i) => ({ family, score: globalThis.testScores[i] ?? 0 })) : originalRankWindows(logits, fonts, temperature) }\n' }))
  await scorePage.goto(base)
  await scorePage.waitForFunction(() => !document.querySelector('#save').disabled)
  const scoreCases = [
    [[.6, .3, .098, .001, .001], ['60.0%', '30.0%', '9.8%', '0.1%', '0.1%']],
    [[.6, .3, .098, .001, .001], ['60.0%', '30.0%', '9.8%', '0.1%', '0.1%']],
    [[.6, .3, .098, .0010001, .0009999], ['60.0%', '30.0%', '9.8%', '0.1%', '<0.1%']],
    [[.999, .001, 0, 0, 0], ['99.9%', '0.1%', '<0.1%', '<0.1%', '<0.1%']],
    [[.9990001, .0009999, 0, 0, 0], ['>99.9%', '<0.1%', '<0.1%', '<0.1%', '<0.1%']],
    [[1, 0, 0, 0, 0], ['>99.9%', '<0.1%', '<0.1%', '<0.1%', '<0.1%']],
  ]
  for (const [scores, labels] of scoreCases) {
    await scorePage.evaluate(scores => { window.testScores = scores }, scores)
    await chooseSample(scorePage, 'lora')
    const result = await saveResult(scorePage)
    assert.deepEqual(result.matches.slice(0, 5).map(m => m.score), scores)
    assert.deepEqual(await scorePage.locator('.result-score').allTextContents(), labels)
    assert.match(await scorePage.locator('#detection-time').textContent(), /^\d+\.\d ms$/)
    assert.equal(result.accepted, scores[0] >= (catalog.calibration?.threshold ?? 1.01))
  }
  const whitePixel = await scorePage.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 1
    const c = canvas.getContext('2d'); c.fillStyle = 'white'; c.fillRect(0, 0, 1, 1)
    return canvas.toDataURL().split(',')[1]
  })
  await scorePage.locator('#file').setInputFiles({ name: 'white-pixel.png', mimeType: 'image/png', buffer: Buffer.from(whitePixel, 'base64') })
  await scorePage.waitForFunction(() => document.querySelector('#message').textContent.includes('No visible text'))
  assert.equal(await scorePage.locator('.result-score').count(), 0)
  assert.equal(await scorePage.locator('#result-summary').textContent(), '')
  assert.ok(await scorePage.locator('#save').isDisabled())
  await scorePage.evaluate(scores => { window.testScores = scores }, scoreCases[0][0])
  await chooseSample(scorePage, 'lora')
  await saveResult(scorePage)
  assert.deepEqual(await scorePage.locator('.result-score').allTextContents(), scoreCases[0][1])
  await scorePage.close()

  if (model.fonts.length === 100) {
    await page.locator('#about summary').click()
    assert.equal(await page.locator('#accuracy tbody tr').count(), 4)
    assert.equal(await page.locator('#accuracy tbody tr').last().locator('td').first().textContent(), `${(catalog.metrics.groups['8+/clean'].accuracy * 100).toFixed(1)}%`)
    await page.screenshot({ path: '.data/demo-checks/metrics-desktop.png', fullPage: true })
    await page.setViewportSize({ width: 320, height: 1000 })
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    await page.screenshot({ path: '.data/demo-checks/metrics-mobile.png', fullPage: true })
    await page.locator('#about summary').click()
    await page.setViewportSize({ width: 1440, height: 1100 })
  }
  const timings = await page.evaluate(async ({ artifact, inputs }) => {
    const { readNetwork, inferCPU, rankWindows } = await import('/src/network.mjs')
    const { createNetworkGPU } = await import('/src/network-gpu.mjs')
    const model = readNetwork(artifact), gpu = await createNetworkGPU(model)
    const windows = inputs.map(input => ({ ...input, pixels: Float32Array.from(input.pixels) }))
    const result = {}
    try {
      for (const backend of ['gpu', 'cpu']) {
        const times = []
        for (let i = 0; i < 21; i++) {
          const start = performance.now()
          const logits = backend === 'gpu' ? await Promise.all(windows.map(w => gpu.infer(w))) : windows.map(w => inferCPU(model, w))
          rankWindows(logits, model.fonts)
          if (i) times.push(performance.now() - start)
        }
        times.sort((a, b) => a - b)
        result[backend] = { median: times[10], p95: times[18], count: times.length }
      }
      return { ...result, windows: windows.map(w => [w.width, w.height]), scope: 'Warm inference + rank, excludes image preparation and font loading; desktop Chromium only.' }
    } finally { gpu.destroy() }
  }, { artifact, inputs: initialResult.inputs })

  const sampleResults = []
  let screenshotExample
  for (const family of model.fonts) {
    await chooseSample(page, family)
    const result = await saveResult(page)
    assert.equal(result.source.known, family)
    await assertInput(page, result)
    sampleResults.push({ family, predicted: result.matches[0].family, correct: result.matches[0].family === family })
    if (family === 'geist-mono') {
      screenshotExample = { family, accepted: result.accepted, matches: result.matches.slice(0, 5), windows: result.inputs.map(({ pixels, ...geometry }) => geometry) }
      await page.screenshot({ path: '.data/demo-checks/geistmono.png', fullPage: true })
    }
  }
  const fragments = await page.evaluate(async ({ ids, temperature, threshold }) => {
    const { prepareInput } = await import('/src/input.mjs')
    const { readNetwork, rankWindows } = await import('/src/network.mjs')
    const { createNetworkGPU } = await import('/src/network-gpu.mjs')
    const model = readNetwork(await (await fetch('/assets/model.json')).json()), gpu = await createNetworkGPU(model), records = []
    try {
      for (const id of ids) for (const text of ['ri', 'ri+sliver', 'rivers']) {
        const canvas = document.createElement('canvas'); let c = canvas.getContext('2d'); c.font = `56px "specimen-${id}"`
        const value = text === 'ri+sliver' ? 'riv' : text, m = c.measureText(value)
        const advance = c.measureText('ri').width
        canvas.width = Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + 12
        canvas.height = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + 12
        c = canvas.getContext('2d'); c.fillStyle = 'white'; c.fillRect(0, 0, canvas.width, canvas.height)
        c.fillStyle = 'black'; c.font = `56px "specimen-${id}"`; c.fillText(value, 6 + m.actualBoundingBoxLeft, 6 + m.actualBoundingBoxAscent)
        const width = text === 'ri+sliver' ? Math.min(canvas.width, Math.ceil(6 + m.actualBoundingBoxLeft + advance + 2)) : canvas.width
        const input = prepareInput(c.getImageData(0, 0, width, canvas.height))
        const logits = await Promise.all(input.windows.map(w => gpu.infer(w))), matches = rankWindows(logits, model.fonts, temperature)
        records.push({ family: id, text, predicted: matches[0].family, score: matches[0].score, accepted: matches[0].score >= threshold, correct: matches[0].family === id })
      }
    } finally { gpu.destroy() }
    return records
  }, { ids: ['inter', 'roboto', 'open-sans', 'source-sans-3', 'montserrat', 'poppins', 'nunito', 'lora', 'merriweather', 'playfair-display'], temperature, threshold: catalog.calibration?.threshold ?? 1.01 })
  await chooseSample(page, 'lora')
  await saveResult(page)
  const beforeEditing = await saveResult(page)
  await page.locator('.result-preview').nth(2).fill('Editable in every face')
  assert.equal(await page.locator('.result-preview').nth(2).evaluate(input => input === document.activeElement), true)
  assert.deepEqual(await page.locator('.result-preview').evaluateAll(inputs => inputs.map(input => input.value)), Array(5).fill('Editable in every face'))
  assert.deepEqual((await saveResult(page)).inputs, beforeEditing.inputs, 'Preview edits must not alter inference pixels')
  await page.locator('#preview-text').fill('A different phrase')
  assert.equal(await page.locator('.result-preview').nth(1).inputValue(), 'A different phrase')
  const specimen = await page.locator('#source').getAttribute('aria-label')
  await page.locator('#preview-text').fill('你好 🪷')
  assert.ok(await page.locator('#preview-error').isVisible())
  assert.equal(await page.locator('.result-preview').nth(1).inputValue(), 'A different phrase')
  await chooseSample(page, 'inter')
  assert.equal(await page.locator('#source').getAttribute('aria-label'), specimen, 'Unsupported glyphs must not produce mislabeled samples')
  assert.equal(await page.evaluate(() => document.activeElement.id), 'preview-text')
  await page.locator('#preview-text').fill('   ')
  assert.ok(await page.locator('#preview-error').isHidden())
  assert.equal(await page.locator('.result-preview').nth(1).inputValue(), 'Quiet rivers flow')
  await page.locator('#preview-text').fill('A different phrase')
  await page.waitForFunction(() => !document.querySelector('#save').disabled)

  await page.locator('#preview-text').fill('你好')
  await page.locator('#sample').click()
  const menuChooser = page.waitForEvent('filechooser')
  await page.locator('#upload').click()
  assert.equal(await page.locator('#sample-menu').evaluate(el => el.matches(':popover-open')), false)
  await (await menuChooser).setFiles({ name: 'menu-open.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') })
  await saveResult(page)
  assert.ok(await page.locator('#preview-error').isHidden(), 'Replacing an image clears stale preview validation')
  await chooseSample(page, 'lora'); await saveResult(page)

  await page.locator('#file').setInputFiles({ name: 'specimen.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') })
  await page.waitForFunction(() => !document.querySelector('#save').disabled)
  const download = page.waitForEvent('download')
  await page.locator('#save').evaluate(button => button.click())
  assert.equal((await download).suggestedFilename(), 'gpu-font-result.json')
  await page.evaluate(base64 => {
    const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0))
    const dt = new DataTransfer(); dt.items.add(new File([bytes], 'pasted.png', { type: 'image/png' }))
    document.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }))
  }, png)
  await page.waitForFunction(() => document.querySelector('#source').getAttribute('aria-label') === 'pasted.png')
  await page.evaluate(base64 => {
    const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0))
    const dt = new DataTransfer(); dt.items.add(new File([bytes], 'dropped.png', { type: 'image/png' }))
    document.querySelector('#stage').dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }))
  }, png)
  await page.waitForFunction(() => document.querySelector('#source').getAttribute('aria-label') === 'dropped.png')
  // A delayed decode must not overwrite a newer upload.
  for (const action of ['replace']) {
    await page.evaluate(() => {
      const original = window.createImageBitmap
      window.testDecode = { original }
      window.createImageBitmap = (...args) => {
        window.createImageBitmap = original
        return new Promise(resolve => { window.testDecode.resume = async () => resolve(await original(...args)) })
      }
    })
    await page.evaluate(base64 => {
      const dt = new DataTransfer()
      dt.items.add(new File([Uint8Array.from(atob(base64), c => c.charCodeAt(0))], 'late-paste.png', { type: 'image/png' }))
      document.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }))
    }, png)
    if (action === 'replace') {
      await page.locator('#file').setInputFiles({ name: 'newer.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') })
      await page.waitForFunction(() => document.querySelector('#source').getAttribute('aria-label') === 'newer.png')
    }
    const before = await page.locator('#source').getAttribute('aria-label')
    await page.evaluate(async () => {
      await window.testDecode.resume()
      await new Promise(resolve => requestAnimationFrame(resolve))
      delete window.testDecode
    })
    assert.equal(await page.locator('#source').getAttribute('aria-label'), before)

  }
  await page.locator('#file').setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('broken image') })
  await page.waitForFunction(() => document.querySelector('#message').textContent.includes('Could not read'))
  const blank = await page.evaluate(() => { const c = document.createElement('canvas'); c.width = 64; c.height = 32; const x = c.getContext('2d'); x.fillStyle = 'white'; x.fillRect(0, 0, 64, 32); return c.toDataURL().split(',')[1] })
  await page.locator('#file').setInputFiles({ name: 'blank.png', mimeType: 'image/png', buffer: Buffer.from(blank, 'base64') })
  await page.waitForFunction(() => document.querySelector('#source').getAttribute('aria-label') === 'blank.png')
  await page.waitForFunction(() => document.querySelector('#message').textContent.includes('No visible text'))
  assert.equal(await page.locator('.result').count(), 0)
  assert.ok(await page.locator('#model-input').isHidden())
  assert.equal(await page.locator('#normalized canvas').count(), 0)
  await page.locator('#sample').click()
  assert.equal(await page.locator('#sample-list button').count(), model.fonts.length)
  assert.equal(await page.evaluate(() => document.activeElement.id), 'sample-search')
  await page.keyboard.press('ArrowUp')
  assert.equal(await page.evaluate(() => document.activeElement.dataset.font), model.fonts.at(-1))
  await page.keyboard.press('ArrowRight')
  assert.equal(await page.evaluate(() => document.activeElement.dataset.font), 'inter')
  await page.locator('#sample-search').fill('not a font')
  assert.ok(await page.locator('#sample-empty').isVisible())
  await page.keyboard.press('Enter')
  assert.equal(await page.locator('#source').getAttribute('aria-label'), 'blank.png')
  await page.keyboard.press('Escape')
  assert.equal(await page.evaluate(() => document.activeElement.id), 'sample')
  await page.locator('#sample').click()
  await page.locator('#sample-search').fill('Merri')
  assert.equal(await page.locator('#sample-list button').count(), 1)
  await page.keyboard.press('Enter')
  await page.waitForFunction(() => document.querySelector('#source').getAttribute('aria-label') === 'Merriweather sample' && !document.querySelector('#save').disabled)
  await page.locator('#sample').click()
  assert.equal(await page.locator('[data-font="merriweather"]').getAttribute('aria-pressed'), 'true')
  await page.screenshot({ path: '.data/demo-checks/popover.png', fullPage: true })
  await page.locator('.wordmark').focus(); await page.keyboard.press('Escape')
  assert.equal(await page.locator('#sample-menu').evaluate(el => el.matches(':popover-open')), false)

  const readCrop = () => page.locator('#selection').evaluate(el => {
    const canvas = document.querySelector('#source')
    return { x: Math.round(parseFloat(el.style.left) / 100 * canvas.width), y: Math.round(parseFloat(el.style.top) / 100 * canvas.height),
      width: Math.round(parseFloat(el.style.width) / 100 * canvas.width), height: Math.round(parseFloat(el.style.height) / 100 * canvas.height) }
  })
  const full = await readCrop(), frame = page.locator('#image-frame')
  const box = await page.locator('#source').boundingBox()
  async function resize(handle, x, y) {
    const h = await page.locator(`[data-handle="${handle}"]`).boundingBox()
    await page.mouse.move(h.x + h.width / 2, h.y + h.height / 2); await page.mouse.down()
    await page.mouse.move(box.x + box.width * x, box.y + box.height * y)
  }
  await resize('nw', .1, .1)
  assert.deepEqual((await saveResult(page)).crop, await readCrop(), 'Matches update before pointer release')
  await page.mouse.up()
  await resize('se', .9, .9); await page.mouse.up()
  const dragged = await readCrop()
  assert.ok(dragged.x > 0 && dragged.width < full.width && dragged.height < full.height)
  await assertInput(page, await saveResult(page))
  await page.mouse.move(box.x + box.width * .5, box.y + box.height * .5); await page.mouse.down()
  await page.mouse.move(box.x + box.width * .55, box.y + box.height * .55); await page.mouse.up()
  const moved = await readCrop()
  assert.equal(moved.width, dragged.width); assert.equal(moved.height, dragged.height)
  assert.ok(moved.x > dragged.x && moved.y > dragged.y, 'Interior drag translates rather than replaces the crop')
  await frame.press('Home'); assert.deepEqual(await readCrop(), full)
  // Click a handle and its destination is the non-drag pointer alternative.
  await page.locator('[data-handle="nw"]').click()
  await page.mouse.click(box.x + box.width * .1, box.y + box.height * .1)
  await page.locator('[data-handle="se"]').click()
  await page.mouse.click(box.x + box.width * .9, box.y + box.height * .9)
  assert.deepEqual(await readCrop(), dragged)
  for (const interruption of ['pointercancel', 'lostpointercapture']) {
    await frame.press('Home')
    await frame.evaluate(el => el.addEventListener('pointerdown', event => { el.dataset.testPointer = event.pointerId }, { once: true }))
    await resize('nw', .1, .1)
    const owned = await readCrop()
    await frame.evaluate((el, box) => {
      for (const type of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel']) el.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerId: 9999, isPrimary: false, clientX: box.x, clientY: box.y }))
    }, box)
    assert.deepEqual(await readCrop(), owned, 'Only the owner pointer may update the crop')
    await frame.evaluate((el, type) => {
      const pointerId = Number(el.dataset.testPointer)
      if (type === 'lostpointercapture') el.releasePointerCapture(pointerId)
      else el.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerId }))
    }, interruption)
    await page.mouse.move(box.x + box.width * .4, box.y + box.height * .4); await page.mouse.up()
    assert.deepEqual(await readCrop(), owned, `${interruption} ends the gesture`)
    await resize('se', .9, .9); await page.mouse.up()
    assert.deepEqual(await readCrop(), dragged, 'A new handle drag works after cancellation')
  }
  await frame.press('Home')
  await frame.press('Shift+ArrowLeft'); await frame.press('ArrowRight'); await frame.press('ArrowRight')
  assert.deepEqual(await readCrop(), { ...full, x: 1, width: full.width - 1 })
  await frame.press('Shift+ArrowRight')
  assert.deepEqual(await readCrop(), { ...full, x: 1, width: full.width - 1 }, 'Resize stays inside the image')
  await frame.press('Home')
  await page.locator('[data-handle="nw"]').press('ArrowRight')
  assert.deepEqual(await readCrop(), { ...full, x: 1, width: full.width - 1 })
  await frame.press('Home')
  await frame.evaluate((el, full) => {
    const handle = el.querySelector('[data-handle="nw"]')
    for (const [key, n] of [['ArrowRight', full.width], ['ArrowDown', full.height]]) for (let i = 0; i < n; i++) handle.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }))
  }, full)
  assert.deepEqual(await readCrop(), { x: full.width - 1, y: full.height - 1, width: 1, height: 1 })
  await frame.press('Home')
  const originalResolution = await saveResult(page)
  for (const scale of [50, 10, 100]) {
    await page.locator('#resolution').evaluate((el, scale) => { el.value = scale; el.dispatchEvent(new Event('input', { bubbles: true })) }, scale)
    const result = await saveResult(page)
    assert.equal(result.resolution.width, Math.max(1, Math.round(full.width * scale / 100)))
    assert.equal(result.resolution.height, Math.max(1, Math.round(full.height * scale / 100)))
    await assertInput(page, result)
    if (scale === 100) assert.deepEqual(result.inputs, originalResolution.inputs, '100 → 50 → 10 → 100 restores the exact original tensors')
    else assert.notDeepEqual(result.inputs, originalResolution.inputs, 'Resolution must change the actual inference input')
  }
  assert.equal(await page.locator('#method, #paste, #clear, #full-crop, .crop-details, .crop-hint, #analyze, #matte, #source-meta').count(), 0)
  assert.equal(await page.locator('#sample-menu #upload').count(), 1)
  assert.equal(await page.locator('.scope, .score-label, .result-foot, .result-preview-control').count(), 0)
  assert.ok(!await page.locator('footer').textContent().then(text => text.includes('Images stay')))
  assert.match(await page.locator('#detection-time').textContent(), /^\d+\.\d ms$/)

  // Hold one real GPU readback while 50 new crop updates arrive.
  await page.evaluate(() => {
    const map = GPUBuffer.prototype.mapAsync, read = CanvasRenderingContext2D.prototype.getImageData
    const state = window.testLive = { calls: 0, reads: [], map, read }
    GPUBuffer.prototype.mapAsync = async function (...args) {
      const hold = ++state.calls === 1 || state.holdNext
      state.holdNext = false
      await map.apply(this, args)
      if (hold) await new Promise(resolve => { state.release = resolve })
    }
    CanvasRenderingContext2D.prototype.getImageData = function (...args) {
      if (this.canvas.id === 'source') state.reads.push(args)
      return read.apply(this, args)
    }
  })
  await frame.press('Home'); await frame.press('ArrowLeft')
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  assert.equal(await page.evaluate(() => window.testLive.calls), 0, 'Unchanged crop bounds must do no GPU work')
  assert.deepEqual(await page.evaluate(() => window.testLive.reads), [], 'Unchanged bounds must not reread the canvas')
  await frame.press('Shift+ArrowLeft')
  await page.waitForFunction(() => !!window.testLive.release)
  await frame.evaluate(el => {
    for (let i = 0; i < 50; i++) el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', shiftKey: true, bubbles: true }))
  })
  assert.equal(await page.evaluate(() => window.testLive.calls), 1, 'New crops must not queue GPU readbacks')
  assert.ok(await page.locator('#save').isDisabled(), 'A stale result cannot be exported')
  await page.evaluate(() => window.testLive.release())
  const liveResult = await saveResult(page), latestCrop = await readCrop()
  assert.deepEqual(liveResult.crop, { ...full, width: full.width - 51 })
  assert.deepEqual(liveResult.crop, latestCrop)
  const live = await page.evaluate(() => ({ calls: window.testLive.calls, reads: window.testLive.reads }))
  assert.ok(live.calls >= 2 && live.calls <= 6, 'At most three windows per active/newest crop')
  assert.deepEqual(live.reads, [[0, 0, full.width - 1, full.height], [0, 0, full.width - 51, full.height]])
  assert.deepEqual(liveResult.matches.map(m => m.family), rankWindows(liveResult.inputs.map(input => inferCPU(model, { ...input, pixels: Float32Array.from(input.pixels) })), model.fonts, temperature).map(m => m.family))
  // Replacing the image cancels pending crops, including a held GPU readback.
  await page.evaluate(() => { window.testLive.holdNext = true; delete window.testLive.release })
  await frame.press('Shift+ArrowLeft')
  await page.waitForFunction(() => !!window.testLive.release)
  await frame.press('Shift+ArrowLeft')
  await page.locator('#file').setInputFiles({ name: 'replacement.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') })
  await page.waitForFunction(() => document.querySelector('#source').getAttribute('aria-label') === 'replacement.png')
  assert.equal(await page.evaluate(() => window.testLive.calls), live.calls + 1)
  await page.evaluate(() => window.testLive.release())
  const replaced = await saveResult(page)
  assert.equal(replaced.source.name, 'replacement.png')
  assert.deepEqual(replaced.crop, initialResult.crop)
  assert.deepEqual(replaced.inputs, initialResult.inputs)
  assert.deepEqual(replaced.matches, initialResult.matches)
  assert.equal((await page.evaluate(() => window.testLive.reads)).length, 4, 'Only the active crop and replacement image may prepare input after replacement')
  await page.evaluate(() => {
    GPUBuffer.prototype.mapAsync = window.testLive.map; CanvasRenderingContext2D.prototype.getImageData = window.testLive.read
    delete window.testLive
  })

  await assertInput(page, replaced)
  const responsive = []
  for (const width of [320, 375, 414, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    if (width < 640) {
      const sizes = await page.locator('#preview-text, #sample-search').evaluateAll(elements =>
        elements.map(e => ({ id: e.id, size: parseFloat(getComputedStyle(e).fontSize) })))
      assert.ok(sizes.every(e => e.size >= 16), `Mobile form text at ${width}px: ${JSON.stringify(sizes)}`)
    }
    await page.screenshot({ path: `.data/demo-checks/${width}.png`, fullPage: true })
    const layout = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth,
      clipped: [...document.querySelectorAll('button:not(:disabled)')].filter(e => e.offsetParent && e.scrollWidth > e.clientWidth + 2).map(e => e.textContent) }))
    assert.ok(layout.scroll <= width, `Horizontal overflow at ${width}`)
    assert.deepEqual(layout.clipped, [], `Clipped controls at ${width}`)
    await page.locator('#sample').click()
    const menu = await page.locator('#sample-menu').boundingBox()
    assert.ok(menu.x >= 0 && menu.x + menu.width <= width && menu.y >= 0 && menu.y + menu.height <= 1000, `Popover clipped at ${width}px: ${JSON.stringify(menu)}`)
    await page.screenshot({ path: `.data/demo-checks/popover-${width}.png`, fullPage: true })
    await page.keyboard.press('Escape')
    responsive.push(layout)
  }
  await page.setViewportSize({ width: 375, height: 850 })
  const contrast = await page.evaluate(() => {
    const c = document.createElement('canvas').getContext('2d', { willReadFrequently: true })
    const style = getComputedStyle(document.documentElement)
    function luminance(token) {
      c.fillStyle = style.getPropertyValue(token); c.fillRect(0, 0, 1, 1)
      const rgb = [...c.getImageData(0, 0, 1, 1).data].slice(0, 3).map(v => { v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4 })
      return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722
    }
    return [['--color-ink', '--color-paper'], ['--color-muted', '--color-paper'], ['--color-muted', '--color-well'], ['--color-accent', '--color-paper'], ['--color-accent-ink', '--color-accent']].map(([a, b]) => {
      const x = luminance(a), y = luminance(b)
      return { pair: [a, b], ratio: (Math.max(x, y) + .05) / (Math.min(x, y) + .05) }
    })
  })
  assert.ok(contrast.every(c => c.ratio >= 4.5), JSON.stringify(contrast))

  const alphaCases = await page.evaluate(() => {
    return ['opaque-dark', 'opaque-light', 'transparent-dark', 'transparent-light', 'half-dark', 'half-light', 'boundary-127', 'boundary-128', 'transparent-empty'].map(name => {
      const c = document.createElement('canvas'); c.width = 128; c.height = 64
      const x = c.getContext('2d'), light = name.endsWith('light')
      if (name.startsWith('opaque')) { x.fillStyle = light ? 'black' : 'white'; x.fillRect(0, 0, 128, 64) }
      if (!name.endsWith('empty')) {
        const gray = name.split('-')[1]
        x.fillStyle = name.startsWith('boundary') ? `rgb(${gray}, ${gray}, ${gray})` : light ? 'white' : 'black'
        for (const rect of [[20, 12, 7, 40], [20, 12, 30, 7], [50, 12, 7, 40], [20, 30, 37, 7]]) x.fillRect(...rect)
        if (name.startsWith('half')) {
          // Apply opacity once to the completed mask; overlapping strokes stay uniform.
          const mask = x.getImageData(0, 0, 128, 64)
          for (let p = 3; p < mask.data.length; p += 4) if (mask.data[p]) mask.data[p] = 128
          x.putImageData(mask, 0, 0)
        }
      }
      return { name, png: c.toDataURL().split(',')[1] }
    })
  })
  let expectedAlpha
  for (const { name, png } of alphaCases) {
    await page.locator('#file').setInputFiles({ name: `${name}.png`, mimeType: 'image/png', buffer: Buffer.from(png, 'base64') })
    await page.waitForFunction(name => document.querySelector('#source').getAttribute('aria-label') === `${name}.png`, name)
    if (name.endsWith('empty')) {
      await page.waitForFunction(() => document.querySelector('#message').textContent.includes('No visible text'))
      assert.equal(await page.locator('.result').count(), 0)
      assert.ok(await page.locator('#save').isDisabled())
    } else {
      const result = await saveResult(page)
      expectedAlpha ??= result
      await assertInput(page, result)
      assert.deepEqual(result.inputs.map(w => [w.width, w.height]), expectedAlpha.inputs.map(w => [w.width, w.height]))
      assert.equal(result.background, ['transparent-light', 'half-light', 'boundary-128'].includes(name) ? 0 : 255)
      const error = maxError(result.inputs.flatMap(w => w.pixels), expectedAlpha.inputs.flatMap(w => w.pixels))
      assert.ok(error < 1e-6, `${name} must retain the same ink; max difference ${error}`)
      assert.deepEqual(result.matches.map(m => m.family), expectedAlpha.matches.map(m => m.family))
    }
  }

  const failure = await browser.newPage()
  failure.on('pageerror', error => errors.push(error.message))
  await failure.goto(base)
  await failure.waitForFunction(() => !document.querySelector('#save').disabled)
  const loaded = await failure.evaluate(() => [...document.fonts].map(f => f.family))
  const missing = model.fonts.find(f => !loaded.includes(`specimen-${f}`))
  assert.ok(missing)
  const beforeFailure = await failure.locator('#source').getAttribute('aria-label')
  const beforeLabel = await failure.locator('#sample-label').textContent()
  await failure.route(`**/assets/fonts/${missing}.ttf`, route => route.abort())
  await chooseSample(failure, missing)
  await failure.waitForFunction(() => document.querySelector('#message').hasAttribute('data-error'))
  assert.equal(await failure.locator('#source').getAttribute('aria-label'), beforeFailure)
  assert.equal(await failure.locator('#sample-label').textContent(), beforeLabel)
  assert.ok(await failure.locator('#save').isDisabled())
  await failure.unroute(`**/assets/fonts/${missing}.ttf`)
  await chooseSample(failure, missing)
  const recoveredSample = await saveResult(failure)
  assert.equal(recoveredSample.source.known, missing)
  await failure.close()

  const fallback = await browser.newPage({ viewport: { width: 1100, height: 900 } })
  fallback.on('pageerror', error => errors.push(error.message))
  await fallback.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }))
  await fallback.goto(base)
  await fallback.waitForFunction(() => document.querySelectorAll('.result').length === 5)
  assert.equal(await fallback.locator('#backend').getAttribute('data-backend'), 'cpu')
  let resumeModel
  let delayedModel = new Promise(resolve => { resumeModel = resolve })
  await fallback.route('**/assets/model.json', async route => { await delayedModel; await route.continue() })
  await fallback.reload()
  assert.ok(await fallback.locator('#empty').isVisible())
  const chooser = fallback.waitForEvent('filechooser')
  await fallback.locator('#empty').click()
  await (await chooser).setFiles({ name: 'empty-open.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') })
  await fallback.waitForFunction(() => document.querySelector('#source').getAttribute('aria-label') === 'empty-open.png')
  await fallback.locator('#file').setInputFiles({ name: 'retained-before-model.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') })
  await fallback.waitForFunction(() => document.querySelector('#source').getAttribute('aria-label') === 'retained-before-model.png')
  resumeModel()
  const retained = await saveResult(fallback)
  assert.equal(retained.source.name, 'retained-before-model.png')
  assert.deepEqual(retained.inputs, initialResult.inputs)
  // Exercise browser/device failure independently of deliberate runtime.destroy().
  const loss = await browser.newPage()
  loss.on('pageerror', error => errors.push(error.message))
  await loss.addInitScript(() => {
    const request = GPUAdapter.prototype.requestDevice
    GPUAdapter.prototype.requestDevice = async function (...args) {
      const device = await request.apply(this, args)
      window.testGPUDevice = device
      return device
    }
  })
  await loss.goto(base)
  await loss.waitForFunction(() => document.querySelectorAll('.result').length === 5)
  await loss.evaluate(() => {
    window.testGPUDevice.destroy()
    for (const key of ['ArrowLeft', 'ArrowRight']) document.querySelector('#image-frame').dispatchEvent(new KeyboardEvent('keydown', { key, shiftKey: true, bubbles: true }))
  })
  await loss.waitForFunction(() => document.querySelector('#backend').dataset.backend === 'cpu' && document.querySelectorAll('.result').length === 5)
  assert.ok(!(await loss.locator('#message').getAttribute('data-error')))
  assert.deepEqual(errors, [])
  const report = { modelSha256: reference.modelSha256, timings, sampleResults, screenshotExample, fragments, cpuMaxError: cpuError, gpuMaxError: gpuError, adapter: gpuResult.info, browser: browser.version(), cases: reference.cases.map(c => c.name), responsive, contrast, liveCrop: { rapidUpdates: 50, inferences: live.calls, lastInferenceMs: liveResult.milliseconds }, automaticBackground: alphaCases.map(c => c.name), deviceLossRecovery: true, interactionErrors: errors }
  await writeFile('bench/demo.json', JSON.stringify(report, null, 2) + '\n')
  console.log('Passed: WebGPU, CPU fallback, upload, paste, drop, replacement/reuse, corrupt/blank image, crop handles and resolution, glyph coverage, samples, model input, export, responsive layouts')
} finally { await browser.close() }
