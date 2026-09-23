import assert from 'node:assert/strict'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { chromium } from 'playwright'
import { readCatalog, matchCatalog, foldTwins, embedWindows } from '../src/catalog.mjs'
import { readNetwork, inferCPU } from '../src/network.mjs'

const read = async p => JSON.parse(await readFile(p, 'utf8'))
// The site runs from the repository: site.json names the model and catalogs in place.
const data = await read('site.json'), artifact = await read(data.model), model = readNetwork(artifact)
// Shipped distinct families and catalogs, fixed at load; accuracy is not credited to a selected or imported catalog.
const expectedIntro = [`Finds the closest of ${data.families.toLocaleString('en-US')} font families, from ${data.catalogs.length} catalogs, to any line of text.`,
  `Experimental: on rendered text in fonts it never saw, it names the right family first ${(data.metrics.top1 * 100).toFixed(1)}% of the time.`]
const intro = page => page.locator('#intro-lead, #intro-scope').allTextContents()
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
  assert.deepEqual(await intro(page), expectedIntro, 'Shipped figures and measured accuracy stay fixed through catalog switches')
  assert.match(await page.locator('#faq').textContent(), /300 Google Fonts families it never saw/)
  assert.doesNotMatch(await page.locator('body').textContent(), /never leaves|nothing is uploaded|images stay/i)
  const catalog = readCatalog(await read(option.file), data), expected = matchCatalog(value.embedding, catalog, value.verdict)
  assert.deepEqual(value.matches, expected)
  assert.equal(value.catalog.sha256, option.sha256); assert.equal(value.accepted, null); assert.equal(value.scoreType, 'cosine')
  // Only families that can draw the detected script are ranked; a catalog with none of them ranks every family by style.
  const script = value.verdict?.script?.[0]?.label, drawable = new Set(catalog.faces.filter(f => !script || !Array.isArray(f.scripts) || f.scripts.includes(script)).map(f => f.familyId)).size
  assert.equal(value.matches.length, drawable || catalog.families)
  if (artifact.heads) assert.ok(value.verdict && script, 'Encoders with heads report a typed verdict')
  // The page shows one row per design (families with identical letters in the crop's script folded); the saved result keeps every family.
  const shown = foldTwins(value.matches, catalog, { script, limit: 5 })
  assert.equal(await page.locator('.result').count(), Math.min(5, shown.length))
  assert.deepEqual(await page.locator('.result-score').allTextContents(), shown.map(m => m.score.toFixed(3)))
  assert.deepEqual(await page.locator('.font-twins').evaluateAll(notes => notes.map(n => JSON.parse(n.dataset.twins))), shown.filter(m => m.siblings.length).map(m => m.siblings), 'Each note carries its full twin list')
  const canvases = await page.locator('#normalized canvas').evaluateAll(cs => cs.map(c => ({ width: c.width, height: c.height, pixels: Array.from(c.getContext('2d').getImageData(0, 0, c.width, c.height).data).filter((_, i) => i % 4 === 0) })))
  assert.deepEqual(canvases, value.inputs.map(i => ({ width: i.width, height: i.height, pixels: i.pixels.map(p => Math.round(p * 255)) })))
  assert.equal(await page.locator('#input-count').textContent(), `${value.inputs.length} window${value.inputs.length === 1 ? '' : 's'}`)
  const ms = value.milliseconds.toFixed(1)
  assert.equal(await page.locator('#detection-time').textContent(), value.cachedEmbedding ? `Ranked in ${ms}ms` : `Detected in ${ms}ms`)
}
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.on('pageerror', e => issues.push(e.message))
  await page.goto(base); await page.waitForFunction(() => document.body.dataset.ready === 'true')
  const original = await result(page)
  assert.equal(original.backend, 'WebGPU'); await verify(page, original, data.catalogs[0])
  // Parastoo draws Lora's Latin letters identically (below the renderer noise); either is the correct first match.
  const googleFaces = (await read(data.catalogs[0].file)).faces, lora = googleFaces.find(f => f.familyId === 'lora' && f.default)
  assert.ok(['lora', ...(lora.twins?.Latn ?? [])].includes(original.matches[0].family), 'The reported full-resolution Lora crop must match Lora or an identical design first')
  // Every Google family has preview hints, and each preview asks Google Fonts for the matched face's own weight and style,
  // so a bold or italic match never borrows its family's regular face.
  assert.equal(Object.keys(data.previews).length, new Set(googleFaces.map(f => f.familyId)).size, 'Every Google family has a preview entry')
  const firstFace = foldTwins(original.matches, readCatalog(await read(data.catalogs[0].file), data), { script: original.verdict?.script?.[0]?.label, limit: 1 })[0].face
  assert.deepEqual(await page.locator('.result-preview').first().evaluate(i => [i.style.fontWeight, i.style.fontStyle]), [String(firstFace.weight ?? 400), firstFace.style ?? 'normal'])
  assert.equal(await page.locator('header .header-icon[aria-label="GitHub repository"]').count(), 1)
  assert.equal(await page.locator('header #backend').getAttribute('class'), 'sr-only')
  assert.equal(await page.locator('footer [aria-label="GitHub repository"]').count(), 0)
  assert.equal(await page.locator('footer .footer-license').textContent(), 'MIT')
  assert.equal(await page.locator('footer .footer-license').getAttribute('rel'), 'license')
  assert.equal(await page.locator('footer .footer-license').getAttribute('href'), 'https://github.com/dy/gpu-font/blob/main/LICENSE')
  const headerRight = await page.locator('header, header .header-icon').evaluateAll(elements => elements.map(element => element.getBoundingClientRect().right))
  assert.ok(Math.abs(headerRight[0] - headerRight[1]) < 1)
  const footerCenters = await page.locator('footer .footer-license, footer .omkara').evaluateAll(elements => elements.map(element => {
    const rect = element.getBoundingClientRect(); return rect.top + rect.height / 2
  }))
  assert.ok(Math.abs(footerCenters[0] - footerCenters[1]) < 1)
  assert.equal(await page.locator('.source-panel > .panel-head > #sample').count(), 1)
  assert.equal(await page.locator('#clear').count(), 0, 'The workbench keeps a source; there is no close')
  // Every text grey meets WCAG AA on every ground it sits on.
  const contrast = await page.evaluate(() => {
    const c = document.createElement('canvas').getContext('2d', { willReadFrequently: true }), style = getComputedStyle(document.documentElement)
    const lum = token => { c.fillStyle = style.getPropertyValue(token); c.fillRect(0, 0, 1, 1); const rgb = [...c.getImageData(0, 0, 1, 1).data].slice(0, 3).map(v => { v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4 }); return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722 }
    return [['--color-faint', '--color-paper'], ['--color-faint', '--color-well'], ['--color-faint', '--color-surface'], ['--color-muted', '--color-well'], ['--color-code', '--color-well']].map(([a, b]) => ({ pair: [a, b], ratio: (Math.max(lum(a), lum(b)) + .05) / (Math.min(lum(a), lum(b)) + .05) }))
  })
  assert.ok(contrast.every(c => c.ratio >= 4.5), JSON.stringify(contrast))
  // Links, popovers and code follow the tokens.
  assert.deepEqual(await page.evaluate(() => [...new Set([...document.querySelectorAll('.popover')].map(p => getComputedStyle(p).borderRadius))]), ['6px'], 'Popovers share one radius')
  assert.equal(await page.locator('.explanation-copy a, .related a').evaluateAll(links => new Set(links.map(a => getComputedStyle(a).textDecorationColor)).size), 1, 'Prose and footer links share one underline')
  assert.ok(await page.locator('.code .tok-k').count() > 0 && await page.locator('.code .tok-c').count() > 0, 'The code block is highlighted')
  assert.deepEqual(await page.locator('.pairs > div').evaluateAll(rows => rows.map(r => [r.children[0].tagName, r.children[1].tagName])).then(pairs => [...new Set(pairs.map(String))]), ['DT,DD'], 'Each question sits beside its answer')
  assert.ok(await page.locator('.goals > div').evaluateAll(rows => rows.length > 0 && rows.every(r => (r.dataset.state === 'not-yet') === r.querySelector('dd').textContent.startsWith('Not yet'))), 'Each goal’s bullet agrees with its status')
  const use = await page.locator('#how-to-use').evaluate(s => [s.querySelector('.use-text'), s.querySelector('.code')].map(e => e.getBoundingClientRect()).map(r => ({ left: r.left, right: r.right, top: r.top })))
  assert.ok(use[0].right <= use[1].left && Math.abs(use[0].top - use[1].top) < 1, 'How to use sets its text beside the code')
  // Only sources whose recorded terms allow it ship; banned and unverified ones stay local.
  const terms = new Map((await read('bench/foundries.json')).sources.map(source => [source.id, source.terms?.status]))
  assert.deepEqual(data.catalogs.filter(option => !['permitted', 'none-found'].includes(terms.get(option.id))).map(option => option.id), [], 'Every shipped catalog has cleared terms')
  // Nothing heavy ships: fonts come from Google Fonts, and the page asks its own origin for no font or image.
  const own = new URL(base).host, fetched = await page.evaluate(() => performance.getEntriesByType('resource').map(r => r.name))
  assert.deepEqual(fetched.filter(u => new URL(u).host === own && /\.(ttf|otf|woff2?|png|jpe?g|webp)(\?|$)/.test(u)), [], 'No font or image is served by the site')
  assert.ok(fetched.some(u => u.startsWith('https://fonts.googleapis.com/css2?family=Lora')), 'Faces are requested from Google Fonts')
  assert.ok(await page.evaluate(() => document.fonts.check('16px Inter') && [...document.fonts].some(f => f.family.replaceAll('"', '') === 'Lora' && f.status === 'loaded')), 'Google Fonts faces load')
  // Facts about the run sit under the model input; facts about the model sit with How it works; nothing is folded away.
  assert.equal(await page.locator('#model-input .input-meta').evaluate(row => ['detection-time', 'result-summary', 'save'].every(id => row.querySelector('#' + id))), true)
  assert.equal(await page.locator('#how-it-works .facts #parameters, #how-it-works .facts #device, #how-it-works .facts #catalog-size').count(), 3)
  assert.equal(await page.locator('details.about, #about').count(), 0)
  assert.deepEqual(await page.locator('.site-nav a').evaluateAll(links => links.map(a => a.getAttribute('href'))), ['./#how-to-use', './#faq', './sources.html'])
  // Sources page: one row per ledger entry; a ban or restriction always shows its quoted clause and where it was read; held-locally families never read as coverage.
  const ledgerData = await read('bench/foundries.json'), sourcesPage = await browser.newPage()
  await sourcesPage.goto(`${base}/sources.html`); await sourcesPage.waitForFunction(n => document.querySelectorAll('#sources-rows tr').length === n, ledgerData.sources.length)
  assert.equal(await sourcesPage.locator('.site-nav [aria-current="page"]').textContent(), 'Sources')
  assert.ok(await sourcesPage.locator('[data-terms="ban"], [data-terms="restricted"]').evaluateAll(marks => marks.every(m => { const d = m.closest('td').querySelector('details'); return d?.querySelector('blockquote')?.textContent && /^https?:/.test(d.querySelector('a')?.href) })), 'Every ban shows its clause and source')
  assert.equal(await sourcesPage.locator('.held').evaluateAll(held => held.filter(h => !h.textContent.endsWith('held locally, not shipped')).length), 0)
  assert.equal(await sourcesPage.locator('#sources-updated').textContent(), ledgerData.updated)
  await sourcesPage.close()
  assert.equal(await page.locator('#legal a[href="./sources.html"]').count(), 1, 'Licenses links to Sources')
  // Dropdowns are absolutely positioned: they keep their place against the trigger while the page scrolls.
  await page.locator('#catalog-button').click()
  const gap = () => page.evaluate(() => Math.round(document.querySelector('#catalog-menu').getBoundingClientRect().top - document.querySelector('#catalog-button').getBoundingClientRect().bottom))
  const opened = await gap(); await page.mouse.wheel(0, 150); await page.waitForTimeout(200)
  assert.equal(await gap(), opened, 'An open dropdown scrolls with its trigger')
  await page.keyboard.press('Escape'); await page.evaluate(() => scrollTo(0, 0))
  // More matches: ranks 06–99, numbered on from the five and under the same rank column, emptied again on close.
  const all = foldTwins(original.matches, readCatalog(await read(data.catalogs[0].file), data), { script: original.verdict?.script?.[0]?.label, limit: 99 }), rest = all.slice(5)
  await page.locator('#more-matches').click()
  assert.equal(await page.locator('#more-list li').count(), rest.length)
  assert.deepEqual(await page.locator('#more-list .rank').evaluateAll(r => [r[0].textContent, r.at(-1).textContent]), ['06', String(5 + rest.length).padStart(2, '0')])
  assert.deepEqual(await page.locator('#more-list .result-score').evaluateAll(s => s.slice(0, 8).map(e => e.textContent)), rest.slice(0, 8).map(m => m.score.toFixed(3)))
  const columns = await page.evaluate(() => ['.results .rank', '#more-matches', '#more-list .rank', '.results .font-name', '#more-list .font-name'].map(q => document.querySelector(q).getBoundingClientRect().left))
  assert.ok(Math.abs(columns[0] - columns[1]) < 1 && Math.abs(columns[0] - columns[2]) < 1 && Math.abs(columns[3] - columns[4]) < 1, `The list sits under the rank column: ${columns}`)
  assert.ok((await page.locator('#more-list .font-twins').allTextContents()).every(t => t.startsWith('≈ ')), 'Twins read as ≈')
  await page.keyboard.press('Escape'); assert.equal(await page.locator('#more-list li').count(), 0)
  // Every quoted figure names a shipped metric: fractions read as percentages, data-format="number" as whole numbers.
  const figures = await page.locator('[data-metric]').evaluateAll(elements => elements.map(e => ({ key: e.dataset.metric, format: e.dataset.format, text: e.textContent })))
  assert.ok(figures.some(f => f.key === 'top1') && figures.every(f => Number.isFinite(data.metrics[f.key])), 'Every quoted figure is a shipped metric')
  assert.deepEqual(figures.map(f => f.text), figures.map(f => f.format === 'number' ? String(Math.round(data.metrics[f.key])) : `${(data.metrics[f.key] * 100).toFixed(1)}%`), 'FAQ and aims figures come from the shipped metrics')
  for (const twin of await page.locator('.font-twins').all()) {
    await twin.hover(); assert.equal(await page.locator('#twins-tip').evaluate(t => t.matches(':popover-open')), true)
    assert.deepEqual(await page.locator('#twins-tip li').allTextContents(), JSON.parse(await twin.getAttribute('data-twins')), 'The tooltip lists every twin')
    await page.mouse.move(1, 1); assert.equal(await page.locator('#twins-tip').evaluate(t => t.matches(':popover-open')), false)
  }
  assert.equal(await page.locator('.results-column > #catalog-control > .panel-head > #catalog-button').count(), 1)
  assert.equal(await page.locator('#model-input > #normalized + .input-meta > #detection-time').count(), 1)
  assert.deepEqual(await page.locator('#source-head').evaluate(h => [...h.children].filter(c => !c.matches('.sr-only')).map(c => c.id || c.className)), ['source-font', 'sample', 'source-spacer', 'draw', 'choose-image'], 'The current kind leads its name; the others follow')
  assert.equal(await page.locator('#tools').getAttribute('data-tool'), 'crop'); assert.ok(await page.locator('#undo').isDisabled())
  assert.ok(await page.locator('.pen-size').isHidden(), 'Brush size shows only with a drawing tool')
  assert.equal(await page.locator('#results-title').getAttribute('class'), 'sr-only')
  assert.deepEqual(await page.locator('.workbench .panel-head > :first-child').allTextContents(), ['Source', 'Model input', 'Catalog'])
  assert.deepEqual(await page.locator('.workbench .panel-head').evaluateAll(heads => heads.map(h => getComputedStyle(h).borderBottomStyle)), ['none', 'none', 'none'], 'Workbench heads carry no rule')
  assert.equal(await page.locator('.intro').evaluate(e => getComputedStyle(e).borderBottomStyle), 'solid', 'A rule separates the introduction from the workbench')
  assert.deepEqual(await page.locator('.workbench .panel-head > h2').evaluateAll(hs => hs.map(h => h.className)), ['sr-only', 'sr-only'], 'Source and Catalog are named for assistive technology only')
  assert.equal(await page.locator('#sample-label').textContent(), 'Lora')
  assert.deepEqual(await page.evaluate(() => ['source-font', 'draw', 'choose-image'].map(id => document.getElementById(id).getAttribute('aria-pressed'))), ['true', 'false', 'false'])
  const layout = await page.evaluate(() => Object.fromEntries(Object.entries({
    source: '.source-panel > .panel-head', catalog: '#catalog-control > .panel-head', input: '#model-input > .panel-head',
    selector: '#catalog-button', kind: '#source-font', name: '#sample', draw: '#draw', open: '#choose-image', inputs: '#normalized', time: '#detection-time', stage: '#stage', match: '.result:first-child'
  }).map(([name, query]) => {
    const rect = document.querySelector(query).getBoundingClientRect()
    return [name, { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }]
  })))
  assert.ok(Math.abs(layout.source.top - layout.catalog.top) < 1, 'Both columns open on the same line')
  assert.ok(Math.abs(layout.source.bottom - layout.catalog.bottom) < 1, 'Both column heads have the same height')
  assert.ok(Math.abs(layout.catalog.right - layout.selector.right) < 1, 'The catalog selector ends its column')
  assert.ok(Math.abs(layout.source.left - layout.kind.left) < 1, 'The source switch starts its column')
  assert.ok(Math.abs(layout.source.right - layout.open.right) < 1, 'The other kinds end the source column, level with the catalog selector')
  assert.ok(layout.kind.right <= layout.name.left && layout.name.right <= layout.draw.left && layout.draw.right <= layout.open.left && layout.draw.top >= layout.source.top - 1 && layout.draw.bottom <= layout.source.bottom + 1, 'Aa and its name lead the line; the sheet and image kinds follow on the right')
  assert.ok(Math.abs(layout.stage.top - layout.match.top) < 1, 'The first match starts level with the preview')
  assert.ok(layout.time.top >= layout.inputs.bottom && Math.abs(layout.time.left - layout.input.left) < 1, 'Recognition time sits under the model input')
  assert.equal(await page.locator('#normalized').evaluate(e => getComputedStyle(e).height), '96px', 'Model input keeps its height for every crop')
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
    if (option.id !== 'google-fonts') {
      assert.equal(await page.locator('.result-preview').count(), 0)
      assert.equal(await page.locator('.reference-preview').count(), 0, 'Captured specimens are never shipped')
      assert.ok((await page.locator('.result .font-name a').evaluateAll(links => links.map(a => a.href))).every(href => /^https:\/\//.test(href)), 'Each match links to its source instead')
    }
  }
  assert.deepEqual((await result(page)).matches, original.matches)
  // Failed imports preserve the selected catalog and current answer.
  for (const payload of ['{', '{}', JSON.stringify({ ...await read(data.catalogs[0].file), encoderSha256: 'wrong' })]) {
    await page.locator('#catalog-file').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from(payload) })
    await page.waitForFunction(() => !document.querySelector('#catalog-error').hidden)
    assert.deepEqual((await result(page)).matches, original.matches)
  }
  const custom = await read(data.catalogs[0].file)
  // Smallest compatible catalog, retaining one real vector and exact owner.
  custom.faces = custom.faces.slice(0, 1)
  custom.vectors = { ...custom.vectors, shape: [1, 128], data: Buffer.from(custom.vectors.data, 'base64').subarray(0, 128).toString('base64'), scales: custom.vectors.scales.slice(0, 1), owners: [0] }
  await page.locator('#catalog-file').setInputFiles({ name: 'myfonts.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(custom)) })
  await page.waitForFunction(() => document.querySelector('#catalog-label').textContent === 'myfonts.json')
  const imported = await result(page)
  assert.deepEqual(await intro(page), expectedIntro, 'Importing a custom catalog changes neither the shipped figures nor the benchmark')
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
  // Narrowing the crop, while updates run and after they settle, moves nothing below the workbench.
  const below = () => page.locator('#how-it-works').evaluate(e => e.getBoundingClientRect().top)
  const settled = await below(), during = new Set()
  for (let i = 0; i < 80; i++) { await page.keyboard.press('Shift+ArrowLeft'); if (i % 8 === 0) during.add(await below()) }
  await result(page); during.add(await below())
  assert.deepEqual([...during], [settled], 'A narrow crop does not move the page')
  await page.keyboard.press('Home'); assert.deepEqual((await result(page)).inputs, original.inputs)
  await choose(page, data.catalogs.at(-1).id)
  // The image button opens the file picker; the chosen file replaces the source.
  const picked = page.waitForEvent('filechooser')
  await page.locator('#choose-image').click()
  await (await picked).setFiles({ name: 'picked.png', mimeType: 'image/png', buffer: Buffer.from((await page.locator('#source').evaluate(c => c.toDataURL())).split(',')[1], 'base64') })
  const pickedResult = await result(page)
  assert.equal(pickedResult.source.name, 'picked.png'); assert.deepEqual(pickedResult.inputs, original.inputs)
  assert.equal(await page.locator('#sample-label').textContent(), 'picked.png')
  await verify(page, pickedResult, data.catalogs.at(-1))
  // Aa returns to the last font sample without opening the picker; the name opens it.
  await page.locator('#source-font').click()
  const refont = await result(page)
  assert.equal(refont.source.known, 'lora'); assert.equal(await page.locator('#sample-menu').evaluate(m => m.matches(':popover-open')), false)
  await page.locator('#source-font').click(); assert.equal((await result(page)).source.known, 'lora', 'Aa in font mode changes nothing')
  await page.locator('#sample').click(); assert.equal(await page.locator('#sample-menu').evaluate(m => m.matches(':popover-open')), true)
  await page.keyboard.press('Escape')
  // The pencil edits any source; undo restores the exact pixels, so the exact tensors.
  const glyphs = await page.locator('#source').boundingBox(), on = (x, y) => [glyphs.x + glyphs.width * x, glyphs.y + glyphs.height * y]
  await page.locator('#pen-tool').click(); assert.equal(await page.locator('#tools').getAttribute('data-tool'), 'pen')
  assert.ok(await page.locator('.crop-handle[data-handle="se"]').isHidden(), 'Crop handles step aside while drawing'); assert.ok(await page.locator('#selection').isVisible())
  await page.mouse.move(...on(.1, .5)); await page.mouse.down(); await page.mouse.move(...on(.9, .5), { steps: 8 }); await page.mouse.up()
  const struck = await result(page)
  assert.equal(struck.source.known, 'lora'); assert.equal(struck.source.strokes, 1); assert.notDeepEqual(struck.inputs, refont.inputs)
  await page.keyboard.press('ControlOrMeta+z'); const unstruck = await result(page)
  assert.deepEqual(unstruck.inputs, refont.inputs, 'Undo restores the original pixels exactly'); assert.ok(await page.locator('#undo').isDisabled())
  await page.locator('#pen-tool').click(); assert.equal(await page.locator('#tools').getAttribute('data-tool'), 'pen', 'The pencil stays selected until another tool is chosen')
  await page.locator('#crop-tool').click(); assert.deepEqual(await page.locator('#crop-tool, #pen-tool, #erase').evaluateAll(bs => bs.map(b => b.getAttribute('aria-pressed'))), ['true', 'false', 'false'], 'Crop is a tool of its own')
  // The sheet starts blank with the pencil active; its strokes are the whole input.
  await page.locator('#draw').click()
  assert.equal(await page.locator('#sample').getAttribute('data-source'), 'drawing'); assert.ok(await page.locator('#sample').isHidden(), 'A drawing needs no name')
  assert.deepEqual(await page.evaluate(() => ['source-font', 'draw', 'choose-image'].map(id => document.getElementById(id).getAttribute('aria-pressed'))), ['false', 'true', 'false'])
  assert.deepEqual(await page.locator('#source-head').evaluate(h => [...h.children].slice(1).map(c => c.id || c.className)), ['draw', 'sample', 'source-spacer', 'source-font', 'choose-image'], 'The sheet moves to the front while drawing')
  assert.equal(await page.evaluate(() => document.activeElement.id), 'draw', 'The moved button keeps focus')
  assert.equal(await page.locator('#tools').getAttribute('data-tool'), 'pen'); assert.equal(await page.locator('#pen-tool').getAttribute('aria-pressed'), 'true')
  assert.equal(await page.locator('.result').count(), 0); assert.ok(await page.locator('#save').isDisabled())
  assert.equal(await page.locator('#message').textContent(), '')
  assert.ok(await page.locator('#model-input').isVisible(), 'Model input keeps its place on an empty sheet')
  const sheet = await page.locator('#source').boundingBox(), at = (x, y) => [sheet.x + sheet.width * x, sheet.y + sheet.height * y]
  for (const [from, to] of [[[.25, .2], [.25, .8]], [[.45, .2], [.45, .8]], [[.25, .5], [.45, .5]], [[.6, .4], [.6, .8]]]) {
    await page.mouse.move(...at(...from)); await page.mouse.down(); await page.mouse.move(...at(...to), { steps: 6 }); await page.mouse.up()
  }
  const drawn = await result(page)
  assert.equal(drawn.source.drawing, true); assert.equal(drawn.source.strokes, 4); assert.deepEqual(drawn.crop, { x: 0, y: 0, width: 720, height: 240 })
  await verify(page, drawn, data.catalogs.at(-1))
  await page.mouse.move(...at(.8, .3)); await page.mouse.down(); await page.mouse.move(...at(.9, .7), { steps: 4 }); await page.mouse.up(); await result(page)
  await page.locator('#undo').click(); const redrawn = await result(page)
  assert.equal(redrawn.source.strokes, 4); assert.deepEqual(redrawn.inputs, drawn.inputs, 'Undo on a sheet restores the exact drawing')
  // The wedge sets the stroke width in sheet pixels; the eraser paints the sheet back to blank.
  await page.locator('#pen-size').fill('40')
  await page.mouse.move(...at(.8, .2)); await page.mouse.down(); await page.mouse.move(...at(.8, .8), { steps: 6 }); await page.mouse.up(); await result(page)
  const inked = await page.locator('#source').evaluate(c => { const d = c.getContext('2d').getImageData(Math.round(c.width * .7), 120, Math.round(c.width * .2), 1).data; let n = 0; for (let i = 0; i < d.length; i += 4) n += d[i] < 128; return n })
  assert.ok(Math.abs(inked - 40) <= 1, `A 40px pen draws a 40px stroke, not ${inked}`)
  const brush = await page.locator('#image-frame').evaluate(f => ({ cursor: f.style.cursor, onScreen: Math.round(40 * document.querySelector('#source').getBoundingClientRect().width / 720) }))
  assert.match(decodeURIComponent(brush.cursor), new RegExp(`width="${brush.onScreen + 2}"`), 'The cursor is the brush at its on-screen size')
  await page.locator('#erase').click(); assert.deepEqual(await page.locator('#pen-tool, #erase').evaluateAll(bs => bs.map(b => b.getAttribute('aria-pressed'))), ['false', 'true'], 'Pencil and eraser are exclusive')
  for (let y = .05; y < 1; y += .1) { await page.mouse.move(...at(.02, y)); await page.mouse.down(); await page.mouse.move(...at(.98, y), { steps: 10 }); await page.mouse.up() }
  await page.waitForFunction(() => !document.querySelector('.result') && !document.querySelector('#results').hasAttribute('aria-busy'))
  assert.equal(await page.locator('#message').textContent(), '', 'An erased sheet is not an error')
  assert.ok(await page.locator('#model-input').isVisible()); assert.equal(await page.locator('#normalized canvas').count(), 0)
  assert.match(decodeURIComponent(await page.locator('#image-frame').evaluate(f => f.style.cursor)), /stroke-dasharray/, 'The eraser cursor is dashed')
  await page.locator('#draw').click(); assert.equal(await page.locator('#tools').getAttribute('data-tool'), 'pen', 'A new sheet starts with the pencil')
  assert.equal(await page.locator('#pen-size').inputValue(), '40', 'Pen width carries over to a new sheet')
  // Before a first source loads, the workbench offers both entries.
  const blank = await browser.newPage({ viewport: { width: 320, height: 900 } }); let releaseSample
  const sampleGate = new Promise(r => { releaseSample = r })
  await blank.route(/fonts\.googleapis\.com\/css2\?family=Lora&/, async route => { await sampleGate; await route.continue() })
  // The held stylesheet would hold the load event too, so wait for the DOM instead.
  await blank.goto(base, { waitUntil: 'domcontentloaded' }); await blank.waitForSelector('body[data-ready="true"]')
  assert.equal(await blank.locator('.result, #normalized canvas').count(), 0); assert.ok(await blank.locator('#save').isDisabled())
  assert.ok(await blank.locator('#model-input').isVisible(), 'Model input is always in place')
  assert.deepEqual(await blank.locator('#input-count, #detection-time').allTextContents(), ['', ''])
  assert.ok(await blank.locator('#empty-sample').isVisible())
  assert.ok(await blank.locator('#sample').isHidden()); assert.ok(await blank.locator('#choose-image').isHidden())
  assert.equal(await blank.locator('#empty').getByText('or', { exact: true }).count(), 1)
  await blank.locator('#empty-sample').click()
  const sampleMenu = await blank.locator('#sample-menu').boundingBox()
  assert.ok(sampleMenu.x >= 0 && sampleMenu.y >= 0 && sampleMenu.x + sampleMenu.width <= 321 && sampleMenu.y + sampleMenu.height <= 901)
  await blank.keyboard.press('Escape')
  assert.ok(await blank.locator('#empty-sample').evaluate(e => e === document.activeElement))
  const other = data.fonts.find(f => f.id !== 'lora')
  await blank.locator('#empty-sample').click(); await blank.locator(`[data-font="${other.id}"]`).click()
  assert.equal((await result(blank)).source.known, other.id, 'Choosing from the empty state supersedes the pending first sample')
  releaseSample(); await blank.waitForTimeout(200)
  assert.equal(await blank.locator('#sample-label').textContent(), other.name, 'A late first sample must not overwrite the chosen one')
  await blank.close()
  // Corrupt images and blank pixels do not retain stale matches; replacement recovers.
  await page.locator('#file').setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('bad') })
  await page.waitForFunction(() => document.querySelector('#message').textContent.includes('Could not read'))
  assert.equal(await page.locator('.result').count(), 0)
  await page.locator('#source-font').click(); assert.equal((await result(page)).source.known, 'lora', 'Aa recovers the last sample from a drawing')
  await choose(page, data.catalogs[0].id)
  for (const width of [320, 375, 414, 768, 1440]) {
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
  assert.deepEqual(await intro(early), ['Finds the closest font families to any line of text.', 'Recognition is experimental.'], 'No figure is claimed before loading its measured artifacts')
  await early.locator('#file').setInputFiles({ name: 'early.png', mimeType: 'image/png', buffer: sourcePng })
  await early.waitForFunction(() => !document.querySelector('#image-frame').hidden)
  await early.waitForTimeout(100)
  assert.equal(await early.locator('#sample-label').textContent(), 'early.png')
  assert.ok(await early.locator('.panel-head > #sample').isVisible())
  assert.deepEqual(await early.evaluate(() => ['source-font', 'draw', 'choose-image'].map(id => document.getElementById(id).getAttribute('aria-pressed'))), ['false', 'false', 'true'])
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
    detectionMilliseconds: original.milliseconds, checks: ['native CPU/PyTorch/WebGPU projections', 'exact displayed tensors and input stats', 'catalog A → A → B → A', 'cached embedding reuse', 'one-face JSON import', 'invalid imports preserve result', 'stale catalog response', 'resolution and crop round trip', 'image button/switch/replacement', 'crop, pencil and eraser tools on any source, exact undo, brush size and cursor', 'twin tooltip and matches 06–99', 'token contrast, links, popovers, code, FAQ and goals layout, dropdowns follow scroll', 'no captured specimens shipped', 'only cleared sources shipped', 'no fonts or images served; fonts from Google', 'sources page', 'Aa returns to the last sample', 'narrow crop keeps layout', 'empty image/sample entry', 'corrupt image recovery', 'keyboard and responsive selectors', 'image before initial catalog', 'CPU fallback'],
    scope: 'Runtime correctness and UI lifecycle only; not recognition accuracy.' }
  await writeFile('bench/catalog-demo.json', JSON.stringify(report, null, 2) + '\n'); console.log(report)
} finally { await browser.close() }
