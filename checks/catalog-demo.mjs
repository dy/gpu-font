import assert from 'node:assert/strict'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { chromium } from 'playwright'
import { readCatalog, matchCatalog, foldTwins, embedWindows, unit } from '../src/catalog.mjs'
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
  assert.equal(await page.locator('#detection-time').textContent(), value.cachedEmbedding ? `ranked in ${ms}ms` : `${ms}ms`, 'A catalog switch says it only re-ranked')
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
  assert.deepEqual(await page.locator('footer .footer-license, footer .omkara').evaluateAll(links => links.map(a => getComputedStyle(a).textDecorationLine)), ['underline', 'underline'], 'MIT and the Krishnized license are both underlined links')
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
  // How it works and How to use: each heading with its text under it, its figure beside them from the heading's top.
  const splits = await page.locator('.split').evaluateAll(blocks => blocks.map(block => { const [heading, text, figure] = [...block.children], [h, t, f] = [heading, text, figure].map(e => e.getBoundingClientRect())
    return [heading.textContent, h.bottom <= t.top && f.left >= t.right && Math.abs(f.top - h.top) < 1] }))
  assert.deepEqual(splits, ['How it works', 'How to use'].map(heading => [heading, true]), 'Split layout')
  // How it works ends on accuracy and speed side by side, bars without text, on the split's columns.
  const measures = await page.locator('#how-it-works .metrics > div').evaluateAll(cols => cols.map(c => [c.id, Math.round(c.getBoundingClientRect().left), Math.round(c.getBoundingClientRect().top), c.querySelectorAll('p').length]))
  const [text, facts] = await page.locator('#how-it-works .split > :nth-child(2), #how-it-works .split > :nth-child(3)').evaluateAll(e => e.map(x => Math.round(x.getBoundingClientRect().left)))
  assert.deepEqual(measures.map(([id, left, top, prose]) => [id, left, top === measures[0][2], prose]), [['how-accurate', text, true, 0], ['how-fast', facts, true, 0]], 'Accuracy and speed sit in How it works, side by side')
  // How does it compare is its table alone; Goals are gone, answered by the questions; How to use follows them.
  assert.deepEqual(await page.locator('#compare').evaluate(s => [...s.children].map(e => e.tagName)), ['H2', 'DIV'])
  assert.equal(await page.locator('#faq + #how-to-use + #compare').count(), 1, 'How to use follows the questions; the comparison follows How to use')
  assert.equal(await page.locator('#inspiration, .goals').count(), 0)
  // The How to use sample runs as printed: its import becomes a dynamic import, its log a return, on the page's own crop.
  const sample = (await page.locator('#how-to-use .code').textContent()).replace(/^import \{([^}]*)\} from ('[^']*')/m, 'const {$1} = await import($2)').replace(/console\.log\((.*)\)\s*$/, 'return [$1]')
  const [family, score] = await page.evaluate(async code => {
    const source = document.getElementById('source'), imageData = source.getContext('2d').getImageData(0, 0, source.width, source.height)
    return new (async () => {}).constructor('imageData', code)(imageData)
  }, sample)
  assert.ok(typeof family === 'string' && family && Number.isFinite(score), `The How to use sample runs: ${family}, ${score}`)
  // The other questions flow in two columns, each a heading over its answer, text only.
  const questions = await page.locator('.questions > .qa').evaluateAll(items => items.map(q => [q.children[0].tagName, q.children[1].tagName, Math.round(q.getBoundingClientRect().left), q.children.length]))
  assert.ok(questions.length >= 7 && questions.every(([h, p, , count]) => h === 'H3' && p === 'P' && count === 2) && new Set(questions.map(q => q[2])).size === 2, JSON.stringify(questions))
  // Accuracy: two bars, the first result and the top 5, each filled to its shipped metric; no title over the other questions.
  assert.deepEqual(await page.locator('#how-accurate .meter').evaluateAll(m => m.map(e => [e.previousElementSibling.textContent, e.dataset.metric, Number(e.style.getPropertyValue('--value'))])), [['1st result', 'top1', data.metrics.top1], ['Top 5', 'top5Accuracy', data.metrics.top5Accuracy]], 'Accuracy names the first result and the top five plainly')
  assert.equal(await page.locator('#faq h2').count(), 0)
  // Only sources whose recorded terms allow it ship; banned and unverified ones stay local.
  const terms = new Map((await read('bench/foundries.json')).sources.map(source => [source.id, source.terms?.status]))
  // A grouped catalog (Other) is cleared only when every source in it is.
  assert.deepEqual(data.catalogs.filter(option => !(option.sources ?? [option.id]).every(id => ['permitted', 'none-found'].includes(terms.get(id)))).map(option => option.id), [], 'Every shipped catalog has cleared terms')
  assert.equal(data.catalogs.at(-1).id, 'other', 'Small sources are searched together as Other, last in the menu'); assert.ok(data.catalogs.every(c => c.sources || c.families >= 10))
  // Nothing heavy ships: fonts come from Google Fonts, and the page asks its own origin for no font or image.
  const own = new URL(base).host, fetched = await page.evaluate(() => performance.getEntriesByType('resource').map(r => r.name))
  assert.deepEqual(fetched.filter(u => new URL(u).host === own && /\.(ttf|otf|woff2?|png|jpe?g|webp)(\?|$)/.test(u)), [], 'No font or image is served by the site')
  assert.ok(fetched.some(u => u.startsWith('https://fonts.googleapis.com/css2?family=Lora')), 'Faces are requested from Google Fonts')
  assert.ok(await page.evaluate(() => document.fonts.check('16px Inter') && [...document.fonts].some(f => f.family.replaceAll('"', '') === 'Lora' && f.status === 'loaded')), 'Google Fonts faces load')
  // Facts about the run sit under the model input; facts about the model sit with How it works; nothing is folded away.
  assert.deepEqual(await page.locator('#model-input .input-meta').evaluate(row => [...row.children].map(e => e.id)), ['result-summary', 'detection-time'], 'Under the model input: what the crop was detected as, then its time')
  assert.equal(await page.locator('.results-panel .results-foot > #more-matches + #save').count(), 1, 'The export sits with the matches it exports')
  assert.deepEqual(await page.locator('#save').evaluate(b => [b.textContent, b.getAttribute('aria-label'), !!b.querySelector('svg')]), ['JSON', 'Download JSON', true], 'An icon and JSON; its name says what it does')
  assert.match(await page.locator('#result-summary').textContent(), /^[^·]+, (upright|italic), Latin$/, 'The verdict is a comma list; its script a word, not an ISO 15924 code')
  assert.equal(await page.locator('#how-it-works .facts #parameters, #how-it-works .facts #device, #how-it-works .facts #catalog-size').count(), 3)
  assert.doesNotMatch(await page.locator('main').innerText(), /·/, 'Lists read with commas, not middle dots')
  assert.equal(await page.locator('details.about, #about').count(), 0)
  assert.deepEqual(await page.locator('.site-nav a').evaluateAll(links => links.map(a => [a.getAttribute('href'), a.textContent])), [['./catalogs.html', 'Catalogs']])
  // Catalogs page: the searched catalogs as site.json ships them, then one row per other
  // ledger entry; a ban or restriction always shows its quoted clause and where it was read; held-locally families never read as coverage.
  const ledgerData = await read('bench/foundries.json'), sourcesPage = await browser.newPage()
  await sourcesPage.goto(`${base}/catalogs.html`)
  // Icon sets stay in the ledger but off the page: they are glyph collections, not fonts to find.
  const others = ledgerData.sources.filter(s => !data.catalogs.some(c => (c.sources ?? [c.id]).includes(s.id)) && s.kind !== 'icons')
  await sourcesPage.waitForFunction(n => document.querySelectorAll('#sources tbody tr').length === n, others.length)
  // Included and Not included are plain titled tables, one after the other, both shown.
  assert.deepEqual(await sourcesPage.locator('#included > h2, #sources > h2').allTextContents(), ['Included', 'Not included'])
  assert.equal(await sourcesPage.locator('[role="tab"], [role="tabpanel"]').count(), 0)
  const includedWidths = await sourcesPage.locator('#included thead th').evaluateAll(t => t.map(e => Math.round(e.getBoundingClientRect().width)))
  // Sortable by any column: Families sorts by number, largest first, then reverses; sources without a count stay last.
  const families = sourcesPage.locator('#sources').getByRole('button', { name: 'Families' }), familyColumn = () => sourcesPage.locator('#sources tbody tr').evaluateAll(rows => rows.map(r => r.lastElementChild.textContent))
  // Approximate sizes a source advertises read ≈N; worked counts read exactly.
  await families.click()
  const size = s => s.work.familiesIndexed ?? s.work.familiesInventoried ?? s.familiesAvailable, shown = s => size(s) == null ? null : (s.work.familiesIndexed ?? s.work.familiesInventoried) != null ? size(s).toLocaleString('en-US') : `≈${size(s).toLocaleString('en-US')}${(s.availableUnit ?? 'families') === 'families' ? '' : ` ${s.availableUnit}`}`
  const counted = others.filter(s => size(s) != null).toSorted((a, b) => size(b) - size(a) || a.name.localeCompare(b.name))
  assert.deepEqual((await familyColumn()).slice(0, counted.length), counted.map(shown)); assert.ok((await familyColumn()).slice(counted.length).every(t => t === '–'))
  assert.equal(await sourcesPage.locator('#sources th[aria-sort]').evaluate(th => [th.textContent, th.getAttribute('aria-sort')].join()), 'Families,descending')
  // Fixed layout: sorting never moves a column, and both tables share the same widths; Source is as wide as the longest names need.
  const widths = () => sourcesPage.locator('#sources thead th').evaluateAll(t => t.map(e => Math.round(e.getBoundingClientRect().width)))
  // Included adds a JSON column; Source, Terms, Licence and Families keep their widths across both tables, Status gives way.
  const before = await widths(), shared = w => [w[0], w[1], w[2], w[4]]
  assert.deepEqual(shared(before), shared(includedWidths)); assert.equal(includedWidths.length, before.length + 1); assert.ok(before[0] > before[3], 'Source is wider than Status')
  await families.click()
  assert.deepEqual((await familyColumn()).slice(0, counted.length), counted.toSorted((a, b) => size(a) - size(b) || a.name.localeCompare(b.name)).map(shown))
  assert.deepEqual(await widths(), before)
  await sourcesPage.locator('#sources').getByRole('button', { name: 'Source' }).click()
  assert.deepEqual(await sourcesPage.locator('#sources tbody th').allTextContents(), others.map(s => s.name).toSorted((a, b) => a.toLowerCase() < b.toLowerCase() ? -1 : a.toLowerCase() > b.toLowerCase() ? 1 : a.localeCompare(b)))
  // Tables end on their last row, with no rule under it.
  assert.deepEqual(await sourcesPage.locator('.sources-table tbody tr:last-child > *').evaluateAll(cells => [...new Set(cells.map(c => getComputedStyle(c).borderBottomStyle))]), ['none'])
  // Included: every shipped catalog, largest first, with the families and faces it searches; Other names its sources.
  const ledgerName = id => ledgerData.sources.find(s => s.id === id).name, rowName = c => c.sources ? c.name : ledgerName(c.id)
  const faces = c => `${c.faces.toLocaleString('en-US')} ${c.faces === 1 ? 'face' : 'faces'}`
  assert.deepEqual(await sourcesPage.locator('#included tbody tr').evaluateAll(rows => rows.map(r => [r.id, r.children[4].textContent, r.querySelector('td .detail').textContent, r.querySelector('th .detail')?.textContent ?? null, r.lastElementChild.querySelector('a[download]')?.getAttribute('href')])),
    data.catalogs.toSorted((a, b) => b.families - a.families || rowName(a).localeCompare(rowName(b))).map(c => [c.id, c.families.toLocaleString('en-US'), faces(c), c.sources ? c.sources.map(ledgerName).join(', ') : null, c.file]))
  assert.equal(await sourcesPage.locator('#sources a[download]').count(), 0, 'Only shipped catalogs download')
  assert.equal(await sourcesPage.locator('.site-nav [aria-current="page"]').textContent(), 'Catalogs')
  assert.ok(await sourcesPage.locator('[data-terms="ban"], [data-terms="restricted"]').evaluateAll(marks => marks.every(m => { const d = m.closest('td').querySelector('details'); return d?.querySelector('blockquote')?.textContent && /^https?:/.test(d.querySelector('a')?.href) })), 'Every ban shows its clause and source')
  assert.equal(await sourcesPage.locator('.held').evaluateAll(held => held.filter(h => !h.textContent.endsWith('held locally, not shipped')).length), 0)
  await sourcesPage.close()
  assert.equal(await page.locator('#legal a[href="./catalogs.html"]').count(), 1, 'Licenses links to Catalogs')
  // Dropdowns are absolutely positioned: they keep their place against the trigger while the page scrolls.
  await page.locator('#catalog-button').click()
  const gap = () => page.evaluate(() => Math.round(document.querySelector('#catalog-menu').getBoundingClientRect().top - document.querySelector('#catalog-button').getBoundingClientRect().bottom))
  const opened = await gap(); await page.mouse.wheel(0, 150); await page.waitForTimeout(200)
  assert.equal(await gap(), opened, 'An open dropdown scrolls with its trigger')
  await page.keyboard.press('Escape'); await page.evaluate(() => scrollTo(0, 0))
  // More matches grows the list in place to the top 50, previews and all; Fewer matches folds it back to five.
  const all = foldTwins(original.matches, readCatalog(await read(data.catalogs[0].file), data), { script: original.verdict?.script?.[0]?.label, limit: 50 })
  // A face Google Fonts refuses says so instead of leaving a gap (the sixth row's subset request is made to fail).
  const refused = url => url.href.includes('&text=') && url.href.includes(`family=${encodeURIComponent(all[5].face.family).replaceAll('%20', '+')}`)
  await page.route(refused, route => route.fulfill({ status: 400, body: '' }))
  await page.locator('#more-matches').click()
  await page.waitForFunction(n => document.querySelectorAll('#results .result').length === n, all.length)
  assert.deepEqual(await page.locator('#more-matches').evaluate(b => [b.firstChild.textContent, b.getAttribute('aria-expanded')]), ['Fewer matches', 'true'])
  assert.deepEqual(await page.locator('#results .rank').evaluateAll(r => [r[0].textContent, r.at(-1).textContent]), ['01', String(all.length).padStart(2, '0')])
  assert.deepEqual(await page.locator('#results .result-score').allTextContents(), all.map(m => m.score.toFixed(3)))
  // The crop stays in view while the long list scrolls beside it.
  await page.locator('#results .result').nth(30).scrollIntoViewIfNeeded()
  assert.equal(await page.locator('.source-panel').evaluate(panel => Math.round(panel.getBoundingClientRect().top)), 0, 'The input panel sticks to the top')
  // A window shorter than the panel pins it by its bottom edge, and the verdict stays in the panel, under its windows, in view.
  const stuck = () => page.evaluate(() => { const panel = document.querySelector('.source-panel').getBoundingClientRect(), verdict = document.querySelector('#result-summary').getBoundingClientRect(), windows = document.querySelector('#normalized').getBoundingClientRect()
    return { bottom: Math.round(panel.bottom - innerHeight), top: Math.round(panel.top), verdict: verdict.top >= windows.bottom && verdict.bottom <= panel.bottom && verdict.bottom <= innerHeight && verdict.top >= 0 } })
  await page.setViewportSize({ width: 1440, height: 560 }); await page.locator('#results .result').nth(40).scrollIntoViewIfNeeded()
  const short = await stuck(); assert.ok(short.bottom === 0 && short.top < 0 && short.verdict, `Short window: ${JSON.stringify(short)}`)
  await page.setViewportSize({ width: 1440, height: 1000 }); await page.locator('#results .result').nth(30).scrollIntoViewIfNeeded()
  const tall = await stuck(); assert.ok(tall.top === 0 && tall.verdict, `Tall window: ${JSON.stringify(tall)}`)
  // Rows past the fifth preview through a subset face of their own, read-only; the refused one names its absence.
  await page.waitForFunction(() => [...document.querySelectorAll('#results .result')].slice(5).every(r => r.querySelector('.result-unavailable') || r.querySelector('.result-preview')?.style.visibility === ''))
  assert.equal(await page.locator('#results .result').nth(5).locator('.result-unavailable').textContent(), 'No preview available')
  assert.ok(await page.locator('#results .result').evaluateAll(rows => rows.slice(6).every(r => r.querySelector('.result-unavailable') || (r.querySelector('.result-preview').readOnly && /^"subset-\d+"$/.test(r.querySelector('.result-preview').style.getPropertyValue('--font-specimen'))))), 'Rows past the fifth use subset faces')
  assert.ok((await page.locator('#results .font-twins').allTextContents()).every(t => t.startsWith('≈ ')), 'Twins read as ≈')
  await page.unroute(refused)
  await page.locator('#more-matches').click(); assert.equal(await page.locator('#results .result').count(), 5)
  // Every quoted figure names a shipped metric: fractions read as percentages, data-format="number" as whole numbers.
  const figures = await page.locator('[data-metric]').evaluateAll(elements => elements.map(e => ({ key: e.dataset.metric, format: e.dataset.format, text: e.textContent })))
  assert.ok(figures.some(f => f.key === 'top1') && figures.every(f => Number.isFinite(data.metrics[f.key])), 'Every quoted figure is a shipped metric')
  const formats = { number: v => String(Math.round(v)), seconds: v => `${v < 1 ? v.toFixed(2) : Math.round(v)} s`, megabytes: v => `${Math.round(v)} MB` }
  assert.deepEqual(figures.map(f => f.text), figures.map(f => formats[f.format]?.(data.metrics[f.key]) ?? `${(data.metrics[f.key] * 100).toFixed(1)}%`), 'FAQ and aims figures come from the shipped metrics')
  // Speed, download size and accuracy all come from site.json: no model-dependent figure is typed into the page.
  assert.ok(['matchWebgpuSeconds', 'matchCpuSeconds', 'downloadMegabytes', 'top1'].every(key => figures.some(f => f.key === key)), 'Speed, size and accuracy are bound')
  const bars = await page.locator('#how-fast .meter').evaluateAll(meters => meters.map(m => Number(m.style.getPropertyValue('--value'))))
  assert.deepEqual(bars, [data.metrics.matchWebgpuSeconds / data.metrics.matchCpuSeconds, 1], 'Time bars are drawn against the slower backend')
  // The comparison is a table: gpu-font counts fonts as other finders do, every shipped face, and names the scripts the model detects.
  const compared = data.catalogs.reduce((n, c) => n + c.faces, 0)
  assert.equal(await page.locator('#compare #compared-fonts').textContent(), compared.toLocaleString('en-US'))
  if (artifact.heads) assert.equal(await page.locator('#compare #compared-scripts').textContent(), String(artifact.heads.script.labels.length))
  assert.deepEqual(await page.locator('#compare thead th').allTextContents(), ['Font finder', 'Fonts', 'Your fonts', 'Weight, italic', 'Subsets', 'Runs on', 'Open source', 'Cost'])
  assert.ok(await page.locator('#compare tbody tr').evaluateAll(rows => rows.every(r => r.cells.length === 8 && r.cells[0].querySelector('.detail')?.textContent && r.cells[1].querySelector('.detail')?.textContent)), 'Each finder names its owner, and each count its sources')
  assert.equal(await page.locator('#compare a[href="./catalogs.html#included"], #compare a[href="./catalogs.html#my-fonts"]').count(), 2)
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
  // Each column names itself, Source and Matches, in the same quiet label as Model input.
  assert.deepEqual(await page.locator('.workbench .column-title').evaluateAll(t => t.map(e => [e.tagName, e.textContent, e.checkVisibility()])), [['H2', 'Source', true], ['H2', 'Matches', true]])
  assert.deepEqual(await page.locator('#model-input .panel-head > :first-child, #catalog-control .panel-head > :first-child').allTextContents(), ['Model input', 'Catalog'])
  assert.deepEqual(await page.locator('.workbench .panel-head').evaluateAll(heads => heads.map(h => getComputedStyle(h).borderBottomStyle)), ['none', 'none', 'none'], 'Workbench heads carry no rule')
  assert.equal(await page.locator('.intro').evaluate(e => getComputedStyle(e).borderBottomStyle), 'solid', 'A rule separates the introduction from the workbench')
  assert.deepEqual(await page.locator('.workbench .panel-head > h2').evaluateAll(hs => hs.map(h => h.className)), ['sr-only'], 'Catalog is named for assistive technology only; Source and Matches are visible column titles')
  assert.equal(await page.locator('#sample-label').textContent(), 'Lora')
  assert.deepEqual(await page.evaluate(() => ['source-font', 'draw', 'choose-image'].map(id => document.getElementById(id).getAttribute('aria-pressed'))), ['true', 'false', 'false'])
  const layout = await page.evaluate(() => Object.fromEntries(Object.entries({
    source: '.source-panel > .panel-head', catalog: '#catalog-control > .panel-head', input: '#model-input > .panel-head',
    selector: '#catalog-button', kind: '#source-font', name: '#sample', draw: '#draw', open: '#choose-image', inputs: '#normalized', summary: '#result-summary', time: '#detection-time', more: '#more-matches', stage: '#stage', match: '.result:first-child'
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
  assert.ok(layout.input.top - layout.stage.bottom < 40 && layout.summary.top >= layout.inputs.bottom, 'Model input follows the preview; the verdict sits under its windows')
  assert.ok(Math.abs(layout.summary.top + layout.summary.bottom - layout.more.top - layout.more.bottom) < 2, 'The verdict shares a line with More matches and JSON')
  assert.ok(Math.abs(layout.summary.left - layout.input.left) < 1 && Math.abs(layout.time.right - layout.input.right) < 1 && Math.abs(layout.time.top - layout.summary.top) < 1, 'The verdict starts its row and the time ends it')
  const row = await page.locator('#normalized').evaluate(e => ({ height: e.getBoundingClientRect().height, width: e.clientWidth, windows: [...e.children].map(c => c.getBoundingClientRect().height) }))
  assert.ok(Math.abs(row.height - Math.min(96, (row.width - 24) / 8)) < .5, 'Model input keeps one height for every crop: a full-width window\'s, at most 2×')
  assert.ok(row.windows.every(h => h <= row.height + .5), 'Every window fits the row')
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
  // Windows fit their third of the row and its height, and keep fitting when the page narrows after they are drawn.
  const fits = () => page.locator('#normalized').evaluate(row => { const box = row.getBoundingClientRect(), third = (row.clientWidth - 2 * parseFloat(getComputedStyle(row).columnGap)) / 3
    return [...row.children].every(c => { const r = c.getBoundingClientRect(); return r.height <= box.height + .5 && r.width <= third + .5 && r.width <= c.width * 2 + .5 }) })
  assert.ok(await fits(), 'Windows fit the row')
  await page.setViewportSize({ width: 375, height: 1000 }); assert.ok(await fits(), 'Windows still fit after the page narrows')
  await page.setViewportSize({ width: 1440, height: 1000 })
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
  assert.deepEqual(await page.locator('#source-head').evaluate(h => [...h.children].map(c => c.id || c.className)), ['draw', 'sample', 'source-spacer', 'source-font', 'choose-image'], 'The sheet moves to the front while drawing')
  assert.equal(await page.evaluate(() => document.activeElement.id), 'draw', 'The moved button keeps focus')
  assert.equal(await page.locator('#tools').getAttribute('data-tool'), 'pen'); assert.equal(await page.locator('#pen-tool').getAttribute('aria-pressed'), 'true')
  assert.equal(await page.locator('.result').count(), 0); assert.ok(await page.locator('#save').isDisabled()); assert.ok(await page.locator('#save').isHidden(), 'No matches, no export')
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
    assert.equal(await page.locator('#add-my-fonts').evaluate(e => e === document.activeElement), true, 'End reaches the last entry, Index my fonts')
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
  assert.equal(await early.locator('#workbench').getAttribute('aria-busy'), 'true', 'The early image waits behind the loader until the first catalog lands')
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
  // My fonts, indexed in the page as the repository indexes Google Fonts: Lora Regular's static catalog instance, drawn and
  // embedded here, must equal the shipped encoder run on the repository's stored reference windows of lora/400 (the same
  // lines, rasterizer and preparation, pixel for pixel). The shipped rows themselves sit 0.999 away: the catalog export
  // embeds those windows with the unquantized checkpoint, not the int8 encoder that the page and this check run.
  const instances = await read('.data/style/reference-instances.json').catch(() => null), myFaces = ['lora/400', 'lora/700i', 'roboto/400', 'playfairdisplay/400', 'inter/400']
  if (instances && myFaces.every(id => instances[id])) {
    // Before indexing there is no status line and no table; after, a table of families, their styles and face counts.
    const context = await browser.newContext(), mine = await context.newPage()
    const table = () => mine.locator('.my-fonts-table tbody tr').evaluateAll(rows => rows.map(r => [...r.children].map(c => c.textContent)))
    await mine.goto(`${base}/catalogs.html`); await mine.waitForFunction(() => document.querySelectorAll('#sources tbody tr').length > 0)
    assert.equal(await mine.locator('#my-fonts-status').textContent(), ''); assert.equal(await mine.locator('.my-fonts-table').count(), 0)
    await mine.locator('#files-input').setInputFiles(await Promise.all(myFaces.map(async id => ({ name: `${id.replace('/', '-')}.ttf`, mimeType: 'font/ttf', buffer: await readFile(instances[id].path) }))))
    await mine.waitForFunction(() => document.querySelectorAll('.my-fonts-table tbody tr').length, null, { timeout: 60000 })
    const indexedRows = await table()
    assert.deepEqual(indexedRows.map(r => r[0]), indexedRows.map(r => r[0]).toSorted((a, b) => a.localeCompare(b))); assert.equal(indexedRows.length, 4)
    assert.equal(indexedRows.reduce((sum, r) => sum + Number(r[2]), 0), 5); const lora = indexedRows.find(r => r[0] === 'Lora'); assert.ok(lora[1].startsWith('Regular, ') && lora[2] === '2', `Lora row: ${lora}`) // styles as the files name them, regular first
    const stored = () => mine.evaluate(() => import('./my-fonts.mjs').then(m => m.readMyFonts()))
    const indexed = readCatalog((await stored()).catalog, data), shipped = readCatalog(await read(data.catalogs[0].file), data)
    const rowsOf = (catalog, test) => catalog.owners.flatMap((owner, r) => test(catalog.faces[owner]) ? [catalog.vectors.subarray(r * 128, r * 128 + 128)] : [])
    const ours = rowsOf(indexed, f => f.family === 'Lora' && f.weight === 400 && f.style === 'normal'), theirs = rowsOf(shipped, f => f.id === 'lora/400')
    const refs = await read('.data/style/references-browser.json'), refPixels = await readFile('.data/style/references-browser.u8')
    const faceIndex = (await read('.data/style/faces.json')).faces.findIndex(f => f.id === 'lora/400')
    const expected = ['Latn-lower', 'Latn-upper'].map(kind => unit(refs.samples.flatMap((sample, i) => sample.face === faceIndex && sample.kind === kind
      ? [embedWindows(refs.windows.filter(w => w.source === i).map(w => inferCPU(model, { width: w.width, height: w.height, pixels: Float32Array.from(refPixels.subarray(w.offset, w.offset + w.width * w.height), v => v / 255) })))] : [])
      .reduce((sum, e) => sum.map((v, d) => v + e[d]), new Float32Array(128))))
    const cosine = (a, b) => a.reduce((sum, v, d) => sum + v * b[d], 0), same = ours.map((row, i) => cosine(row, expected[i])), near = ours.map(row => Math.max(...theirs.map(other => cosine(row, other))))
    assert.equal(ours.length, 2, 'Lowercase and capitals')
    assert.ok(same.every(c => c > .9999), `Browser-indexed Lora against the encoder on the stored reference windows: ${same}`)
    assert.ok(near.every(c => c > .998), `Browser-indexed Lora against the shipped rows: ${near}`)
    // Installed faces render by PostScript name, join the stored faces and preview in the results; file faces have no preview.
    if (process.platform === 'darwin') {
      await context.grantPermissions(['local-fonts'], { origin: base })
      await mine.evaluate(() => { const all = window.queryLocalFonts; window.queryLocalFonts = () => all.call(window, { postscriptNames: ['Georgia', 'Georgia-Bold'] }) })
      await mine.locator('#index-installed').click()
      await mine.waitForFunction(() => document.querySelectorAll('.my-fonts-table tbody tr').length === 5, null, { timeout: 60000 })
      assert.deepEqual((await table()).find(r => r[0] === 'Georgia').slice(1), ['Regular, Bold', '2'])
      assert.deepEqual((await stored()).catalog.faces.filter(f => f.local).map(f => f.local).sort(), ['Georgia', 'Georgia-Bold'])
    }
    const search = await context.newPage(); await search.goto(`${base}/?catalog=my-fonts`)
    const found = await result(search)
    assert.equal(found.catalog.id, 'my-fonts'); assert.equal(await search.locator('#catalog-label').textContent(), 'My fonts')
    assert.equal(found.matches[0].face.family, 'Lora', 'A Lora sample finds Lora among my fonts')
    assert.ok(await search.locator('#add-my-fonts').isHidden())
    if (process.platform === 'darwin') {
      await search.waitForFunction(() => [...document.querySelectorAll('#results .result-preview')].some(p => p.style.visibility === ''))
      assert.deepEqual(await search.locator('#results .result').evaluateAll(rows => rows.filter(r => r.querySelector('.result-preview')).map(r => r.querySelector('.font-name a').textContent)), ['Georgia'])
    }
    await search.close()
    await mine.locator('#my-fonts-remove').click(); await mine.waitForFunction(() => !document.querySelector('.my-fonts-table') && document.querySelector('#my-fonts-stored').hidden)
    const after = await context.newPage(); await after.goto(base); await after.waitForFunction(() => document.body.dataset.ready === 'true')
    assert.ok(await after.locator('#add-my-fonts').evaluate(a => !a.hidden) && await after.locator('[data-catalog="my-fonts"]').count() === 0, 'Without my fonts the menu links to indexing')
    await context.close()
  } else console.log('My fonts differential skipped: no local catalog instances in .data/style')
  const report = { encoderSha256: data.encoderSha256, catalogs: data.catalogs.map(o => ({ id: o.id, sha256: o.sha256, families: o.families, faces: o.faces })), cpuError, gpuError,
    detectionMilliseconds: original.milliseconds, checks: ['native CPU/PyTorch/WebGPU projections', 'exact displayed tensors and input stats', 'catalog A → A → B → A', 'cached embedding reuse', 'one-face JSON import', 'invalid imports preserve result', 'stale catalog response', 'resolution and crop round trip', 'image button/switch/replacement', 'crop, pencil and eraser tools on any source, exact undo, brush size and cursor', 'twin tooltip and the list grown to 50 with subset previews', 'token contrast, links, popovers, code, section and question layout, dropdowns follow scroll', 'no captured specimens shipped', 'only cleared sources shipped', 'no fonts or images served; fonts from Google', 'catalogs page', 'my fonts: browser-indexed Lora equals the encoder on stored references, join, installed faces, previews, removal', 'Aa returns to the last sample', 'narrow crop keeps layout', 'empty image/sample entry', 'corrupt image recovery', 'keyboard and responsive selectors', 'image before initial catalog', 'CPU fallback'],
    scope: 'Runtime correctness and UI lifecycle only; not recognition accuracy.' }
  await writeFile('bench/catalog-demo.json', JSON.stringify(report, null, 2) + '\n'); console.log(report)
} finally { await browser.close() }
