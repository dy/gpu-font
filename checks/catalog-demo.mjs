import assert from 'node:assert/strict'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { openSync, readSync, closeSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { chromium } from 'playwright'
import { readCatalog, subsetCatalog, matchCatalog, foldTwins, embedWindows, unit, rowBytes, unpack, readHeads, verdict, writeStyle, readStyle } from '../src/catalog.mjs'
import { packVectors } from '../src/references.mjs'
import { previewPath } from '../previews.mjs'
import { readNetwork, inferCPU } from '../src/network.mjs'

const read = async p => JSON.parse(await readFile(p, 'utf8'))
// The site runs from the repository: site.json names the model and catalogs in place.
const data = await read('site.json'), artifact = await read(data.model), model = readNetwork(artifact)
// Faces scripts/previews.py found nothing to draw for; every other non-Google face has a picture stored with the site.
const pictureless = new Set((await read('bench/previews.json')).missing.map(m => m.id))
// Shipped distinct families and catalogs, fixed at load; accuracy is not credited to a selected or imported catalog.
const expectedIntro = [`Finds the closest of ${data.families.toLocaleString('en-US')} font families, from ${data.catalogs.length} catalogs, to any line of text.`,
  `Experimental: on rendered crops of every Google Fonts family, on text it never trained on, it names the right family first ${(data.metrics.top1 * 100).toFixed(1)}% of the time.`]
const intro = page => page.locator('#intro-lead, #intro-scope').allTextContents()
// The verdict: one pill per label, category, then an optional fine class, posture and script as a word, not an ISO 15924 code.
const verdictPills = async page => (await page.locator('#result-summary > .pill').allTextContents()).join('|'), VERDICT = /^[^|]+(\|[^|]+)?\|(Upright|Italic)\|Latin$/
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
  // The page shows one row per family (its kinds with identical letters in the crop's script folded); the saved result keeps every family.
  const shown = foldTwins(value.matches, catalog, { script, limit: 5 })
  assert.equal(await page.locator('.result').count(), Math.min(5, shown.length))
  assert.deepEqual(await page.locator('.result-score').allTextContents(), shown.map(m => m.score.toFixed(3)))
  assert.deepEqual(await page.locator('.font-twins').evaluateAll(notes => notes.map(n => JSON.parse(n.dataset.twins))), shown.filter(m => m.siblings.length).map(m => m.siblings), 'Each note carries its full twin list')
  const canvases = await page.locator('#normalized canvas').evaluateAll(cs => cs.map(c => ({ width: c.width, height: c.height, pixels: Array.from(c.getContext('2d').getImageData(0, 0, c.width, c.height).data).filter((_, i) => i % 4 === 0) })))
  assert.deepEqual(canvases, value.inputs.map(i => ({ width: i.width, height: i.height, pixels: i.pixels.map(p => Math.round(p * 255)) })))
  assert.equal(await page.locator('#input-count').textContent(), `${value.inputs.length} window${value.inputs.length === 1 ? '' : 's'}`)
  assert.match(await verdictPills(page), VERDICT, 'A catalog switch keeps the verdict pills')
  const ms = value.milliseconds.toFixed(1)
  assert.equal(await page.locator('#detection-time').textContent(), value.cachedEmbedding ? `Ranked in ${ms}ms` : `${ms}ms`, 'A catalog switch says it only re-ranked')
}
// What a row's preview settled on: the image it shows, its font, or its note; null while one is still loading.
const previewState = row => {
  const image = row.querySelector('.result-image'), input = row.querySelector('.result-preview'), note = row.querySelector('.result-unavailable')
  // A font is the family the field renders in: a name CSS cannot read leaves the page's own font there.
  return image ? (image.complete && image.naturalWidth ? `image ${image.src}` : null) : input ? (input.style.visibility === '' ? `font ${getComputedStyle(input).fontFamily.replaceAll('"', '')}` : null) : note?.textContent ?? null
}
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.on('pageerror', e => issues.push(e.message))
  await page.addInitScript(`globalThis.previewState = ${previewState}`)  // the page's own copy, for the checks below
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
  assert.deepEqual([await page.locator('footer .credit').textContent(), await page.locator('footer .credit a').getAttribute('href')], ['Made by dy.', 'https://github.com/dy'])
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
  assert.equal(await page.locator('.source-body > .panel-head > #sample').count(), 1)
  assert.equal(await page.locator('#clear').count(), 0, 'The workbench keeps a source; there is no close')
  // Every text grey meets WCAG AA on every ground it sits on.
  const contrast = await page.evaluate(() => {
    const c = document.createElement('canvas').getContext('2d', { willReadFrequently: true }), style = getComputedStyle(document.documentElement)
    const lum = token => { c.fillStyle = style.getPropertyValue(token); c.fillRect(0, 0, 1, 1); const rgb = [...c.getImageData(0, 0, 1, 1).data].slice(0, 3).map(v => { v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4 }); return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722 }
    return [['--color-muted', '--color-paper'], ['--color-muted', '--color-well'], ['--color-muted', '--color-surface'], ['--color-ink', '--color-well']].map(([a, b]) => ({ pair: [a, b], ratio: (Math.max(lum(a), lum(b)) + .05) / (Math.min(lum(a), lum(b)) + .05) }))
  })
  assert.ok(contrast.every(c => c.ratio >= 4.5), JSON.stringify(contrast))
  // Links and popovers follow the tokens.
  assert.deepEqual(await page.evaluate(() => [...new Set([...document.querySelectorAll('.popover')].map(p => getComputedStyle(p).borderRadius))]), ['6px'], 'Popovers share one radius')
  assert.equal(await page.locator('.explanation-copy a, .credit a').evaluateAll(links => new Set(links.map(a => getComputedStyle(a).textDecorationColor)).size), 1, 'Prose and footer links share one underline')
  // How it works: its heading with its text under it, its figure beside them from the heading's top.
  const splits = await page.locator('.split').evaluateAll(blocks => blocks.map(block => { const [heading, text, figure] = [...block.children], [h, t, f] = [heading, text, figure].map(e => e.getBoundingClientRect())
    return [heading.textContent, h.bottom <= t.top && f.left >= t.right && Math.abs(f.top - h.top) < 1] }))
  assert.deepEqual(splits, [['How it works', true]], 'Split layout')
  // How it works: the text with the model's facts under it; accuracy over speed beside them, bars without text.
  const measures = await page.locator('#how-it-works .metrics > div').evaluateAll(cols => cols.map(c => { const r = c.getBoundingClientRect(); return [c.id, Math.round(r.left), Math.round(r.top), Math.round(r.bottom), c.querySelectorAll('p').length] }))
  const [text, copy, facts, figure] = await page.locator('#how-it-works :is(.split > :nth-child(2), .explanation-copy, .facts, .split > :nth-child(3))').evaluateAll(e => e.map(x => x.getBoundingClientRect()).map(r => ({ left: Math.round(r.left), top: Math.round(r.top), bottom: Math.round(r.bottom) })))
  assert.deepEqual(measures.map(([id, left, , , prose]) => [id, left, prose]), [['how-accurate', figure.left, 0], ['how-fast', figure.left, 0]], 'Accuracy and speed sit in the right column of How it works')
  assert.ok(measures[1][2] >= measures[0][3], 'Speed sits under accuracy')
  assert.ok(facts.left === text.left && facts.top >= copy.bottom, 'The model\'s facts sit under the text')
  // How does it compare is its table alone and follows the questions; Goals are gone, answered by the questions; the API lives in the README.
  assert.deepEqual(await page.locator('#compare').evaluate(s => [...s.children].map(e => e.tagName)), ['H2', 'DIV'])
  assert.equal(await page.locator('#faq + #compare').count(), 1, 'The comparison follows the questions')
  assert.equal(await page.locator('#inspiration, .goals').count(), 0)
  // The README's usage sample runs as printed, on the page's own crop: its package import becomes a dynamic import of the module.
  const usage = (await readFile('README.md', 'utf8')).match(/```js\n([^`]*)```/)[1].replace(/^import \{([^}]*)\} from 'gpu-font'/m, "const {$1} = await import('./src/match.mjs')")
  const [family, styleName, score] = await page.evaluate(async code => {
    const source = document.getElementById('source'), imageData = source.getContext('2d').getImageData(0, 0, source.width, source.height)
    return new (async () => {}).constructor('imageData', `${code}\nreturn [best.face.family, best.face.styleName, best.score]`)(imageData)
  }, usage)
  assert.ok(family && styleName && Number.isFinite(score), `The README sample runs: ${family} ${styleName}, ${score}`)
  // The other questions flow in two columns, each a heading over its answer, text only.
  const questions = await page.locator('.questions > .qa').evaluateAll(items => items.map(q => [q.children[0].tagName, q.children[1].tagName, Math.round(q.getBoundingClientRect().left), q.children.length]))
  assert.ok(questions.length >= 7 && questions.every(([h, p, , count]) => h === 'H3' && p === 'P' && count === 2) && new Set(questions.map(q => q[2])).size === 2, JSON.stringify(questions))
  // Accuracy: two bars, the first result and the top 5, each filled to its shipped metric; no title over the other questions.
  assert.deepEqual(await page.locator('#how-accurate .meter').evaluateAll(m => m.map(e => [e.previousElementSibling.textContent, e.dataset.metric, Number(e.style.getPropertyValue('--value'))])), [['1st result', 'top1', data.metrics.top1], ['Top 5', 'top5Accuracy', data.metrics.top5Accuracy]], 'Accuracy names the first result and the top five plainly')
  assert.equal(await page.locator('#faq h2').count(), 0)
  // A source ships when its recorded terms allow it, or by a decision recorded beside them in the ledger; the rest stay local.
  const ledger = new Map((await read('bench/foundries.json')).sources.map(source => [source.id, source])), cleared = id => ['permitted', 'none-found'].includes(ledger.get(id)?.terms?.status) || ledger.get(id)?.decision?.ship === true
  // A grouped catalog (Other) is cleared only when every source in it is.
  assert.deepEqual(data.catalogs.filter(option => !(option.sources?.map(s => s.id) ?? [option.id]).every(cleared)).map(option => option.id), [], 'Every shipped catalog has cleared terms or a recorded decision')
  assert.equal(data.catalogs.at(-1).id, 'other', 'Small sources are searched together as Other, last in the menu'); assert.ok(data.catalogs.every(c => c.sources || c.families >= 100))
  // No font ships: fonts come from Google Fonts. The only images the page asks its own origin for are stored previews.
  const own = new URL(base).host, fetched = await page.evaluate(() => performance.getEntriesByType('resource').map(r => r.name))
  assert.deepEqual(fetched.filter(u => new URL(u).host === own && /\.(ttf|otf|woff2?)(\?|$)/.test(u)), [], 'No font is served by the site')
  assert.deepEqual(fetched.filter(u => new URL(u).host === own && /\.(png|jpe?g|webp)(\?|$)/.test(u) && !/\/previews\/[0-9a-f]{2}\/[0-9a-f]{14}\.webp$/.test(u)), [], 'The site serves no image but stored previews')
  assert.ok(fetched.some(u => u.startsWith('https://fonts.googleapis.com/css2?family=Lora')), 'Faces are requested from Google Fonts')
  assert.ok(await page.evaluate(() => document.fonts.check('16px Inter') && [...document.fonts].some(f => f.family.replaceAll('"', '') === 'Lora' && f.status === 'loaded')), 'Google Fonts faces load')
  // The run's input windows fold under Model, closed at first, with their outlines on the preview. The time stays on
  // Model's row, on the right; the JSON export sits on More matches' row, on the right.
  assert.equal(await page.locator('.results-column > .column-head + #result-summary.input-meta').count(), 1, 'Under Matches: what the crop was detected as')
  const endsRow = (selector, row) => page.locator(selector).evaluate((e, row) => { const box = e.closest(row).getBoundingClientRect(), own = e.getBoundingClientRect(); return own.width > 0 && Math.abs(own.right - box.right) < 1 && own.top >= box.top && own.bottom <= box.bottom }, row)
  assert.equal(await page.locator('.source-body > #model-details > summary > .input-time:last-child > svg + #detection-time').count(), 1, 'Model’s row holds the time, after a clock')
  assert.ok(await endsRow('#detection-time', 'summary'), 'The time ends Model’s row, folded')
  assert.deepEqual(await page.locator('.results-panel .results-foot > *').evaluateAll(es => es.map(e => e.id)), ['more-matches', 'copy-link', 'save'], 'Under the matches: More matches, then Link and JSON')
  assert.ok(await endsRow('#save', '.results-foot'), 'JSON ends More matches’ row')
  const outlines = () => page.locator('#input-regions').evaluate(e => getComputedStyle(e).display)
  assert.deepEqual([await page.locator('#model-details').evaluate(d => d.open), await page.locator('#model-input').isVisible(), await outlines()], [false, false, 'none'], 'Model is folded, with the outlines')
  await page.locator('#model-details > summary').click()
  assert.deepEqual([await page.locator('#model-details').evaluate(d => d.open), await page.locator('#model-input').isVisible(), await outlines()], [true, true, 'block'], 'Model unfolds, with the outlines')
  assert.deepEqual(await page.locator('#save').evaluate(b => [b.textContent, b.getAttribute('aria-label'), !!b.querySelector('svg')]), ['JSON', 'Download JSON', true], 'An icon and JSON; its name says what it does')
  assert.match(await verdictPills(page), VERDICT, 'The verdict is one pill per label')
  assert.equal(await page.locator('#how-it-works .facts #parameters, #how-it-works .facts #device, #how-it-works .facts #catalog-size').count(), 3)
  assert.doesNotMatch(await page.locator('main').innerText(), /·/, 'Lists read with commas, not middle dots')
  assert.equal(await page.locator('details.about, #about').count(), 0)
  assert.equal(await page.locator('.site-nav, a[href="./catalogs.html"], a[href="./catalogs.html#sources"]').count(), 0, 'Catalogs stays out of the menu and the page; only My fonts links to it')
  // Catalogs page: one Sources table. The catalogs a search covers as site.json ships them, each downloadable; then the
  // sources not searched, grouped. Each group folds open into its sources, sorted by the chosen column inside it. A ban or
  // restriction always shows its quoted clause and where it was read; held-locally families never read as coverage.
  const ledgerData = await read('bench/foundries.json'), sourcesPage = await browser.newPage()
  await sourcesPage.goto(`${base}/catalogs.html`)
  await sourcesPage.waitForFunction(n => document.querySelectorAll('#sources tbody tr').length >= n, data.catalogs.length + 1)
  assert.equal(await sourcesPage.locator('#sources > h2').textContent(), 'Sources')
  // A rights holder can ask to remove a font or change its link: an issue opened from the sources or from Licenses.
  const optOut = /^https:\/\/github\.com\/dy\/gpu-font\/issues\/new\?/
  assert.deepEqual(await sourcesPage.locator('#sources > h2 + .explanation-copy').evaluate(e => [e.textContent, e.querySelector('a').href]), ['Is your font here? Ask to remove it or change its link.', await page.locator('#legal a', { hasText: 'Ask to remove' }).evaluate(a => a.href)], 'Rights holders are asked before the table, where Licenses sends them')
  assert.match(await page.locator('#legal a', { hasText: 'Ask to remove' }).getAttribute('href'), optOut)
  assert.equal(await sourcesPage.locator('#sources .sources-table, [role="tab"]').count(), 1, 'One table, no tabs')
  const table = () => sourcesPage.locator('#sources tbody tr').evaluateAll(rows => rows.map(r => ({ id: r.id, member: r.classList.contains('member-row'), group: !!r.querySelector('.fold'), families: [...r.children[4].querySelectorAll(':scope > :not(.json-link)')].map(e => e.textContent).join(' '), json: r.children[4].querySelector('a[download]')?.getAttribute('href') ?? null })))
  // Folded, the table lists the shipped catalogs in site.json's order, each with its JSON under its family count, then the unsearched groups.
  const folded = await table()
  assert.deepEqual(folded.slice(0, data.catalogs.length).map(r => [r.id, r.families, r.json]), data.catalogs.map(c => [c.id, c.families.toLocaleString('en-US'), c.file]))
  assert.ok(folded.slice(data.catalogs.length).every(r => r.group && !r.json && !r.member), 'After the catalogs come the unsearched groups, folded')
  // Unfolded, Other lists exactly the sources it groups, and the unsearched groups together hold every other source but icon sets.
  for (const id of folded.filter(r => r.group).map(r => r.id)) await sourcesPage.locator(`#${id} .fold`).click()
  const open = await table(), groupOf = new Map()
  let current = null
  for (const row of open) row.member ? groupOf.set(row.id, current) : (current = row.id)
  assert.deepEqual(open.filter(r => groupOf.get(r.id) === 'other').map(r => r.id).toSorted(), data.catalogs.find(c => c.id === 'other').sources.map(s => s.id).toSorted())
  const unsearched = ledgerData.sources.filter(s => !data.catalogs.some(c => (c.sources?.map(m => m.id) ?? [c.id]).includes(s.id)) && s.kind !== 'icons')
  assert.deepEqual(open.filter(r => r.member && groupOf.get(r.id) !== 'other').map(r => r.id).toSorted(), unsearched.map(s => s.id).toSorted(), 'Every unsearched source sits in one group; icon sets stay off the page')
  assert.ok(open.every(r => !r.member || !r.json), 'Only whole catalogs download')
  // Sortable by any column inside each group: Families sorts by number, largest first, then reverses; sources without a
  // count stay last. Approximate sizes a source advertises read ≈N; worked counts read exactly. Sorting moves no column.
  const size = s => s.work.familiesIndexed ?? s.work.familiesInventoried ?? s.familiesAvailable, shown = s => size(s) == null ? '–' : (s.work.familiesIndexed ?? s.work.familiesInventoried) != null ? size(s).toLocaleString('en-US') : `≈${size(s).toLocaleString('en-US')}${(s.availableUnit ?? 'families') === 'families' ? '' : ` ${s.availableUnit}`}`
  const commercial = unsearched.filter(s => groupOf.get(s.id) === 'commercial')
  const byFamilies = dir => commercial.toSorted((a, b) => (size(a) == null) - (size(b) == null) || dir * ((size(a) ?? 0) - (size(b) ?? 0)) || a.name.localeCompare(b.name)).map(shown)
  const order = async () => (await table()).filter(r => groupOf.get(r.id) === 'commercial').map(r => r.families)
  const widths = () => sourcesPage.locator('#sources thead th').evaluateAll(t => t.map(e => Math.round(e.getBoundingClientRect().width)))
  const before = await widths(); assert.ok(Math.max(...before.slice(1, 4)) - Math.min(...before.slice(1, 4)) <= 1, 'Terms, Licence and Status are equally wide')
  assert.equal(await sourcesPage.locator('#sources th[aria-sort]').evaluate(th => [th.textContent, th.getAttribute('aria-sort')].join()), 'Families,descending')
  assert.deepEqual(await order(), byFamilies(-1))
  await sourcesPage.locator('#sources').getByRole('button', { name: 'Families' }).click()
  assert.deepEqual(await order(), byFamilies(1)); assert.deepEqual(await widths(), before)
  // Tables end on their last row, with no rule under it.
  assert.deepEqual(await sourcesPage.locator('.sources-table tbody tr:last-child > *').evaluateAll(cells => [...new Set(cells.map(c => getComputedStyle(c).borderBottomStyle))]), ['none'])
  assert.ok(await sourcesPage.locator('[data-terms="ban"], [data-terms="restricted"]').evaluateAll(marks => marks.every(m => { const td = m.closest('td'); return td.querySelector('blockquote')?.textContent && /^https?:/.test(td.querySelector('.checked a')?.href) })), 'Every ban shows its clause and source')
  assert.equal(await sourcesPage.locator('.held').evaluateAll(held => held.filter(h => !h.textContent.endsWith('held locally, not shipped')).length), 0)
  // One pill everywhere: the verdict and a match's face share one look, in the secondary grey the sources table uses.
  const look = ['backgroundColor', 'color', 'borderRadius', 'fontSize', 'lineHeight', 'paddingLeft']
  const pills = await page.locator('#result-summary > .pill, .result .font-style').evaluateAll((es, look) => es.map(e => look.map(k => getComputedStyle(e)[k])), look), [allowed] = pills
  assert.ok(pills.length > 5); for (const pill of pills) assert.deepEqual(pill, allowed)
  assert.equal(await page.evaluate(() => { const pill = document.createElement('span'); pill.className = 'pill'; document.body.append(pill); const display = getComputedStyle(pill).display; pill.remove(); return display }), 'none', 'A face without a name shows no empty pill')
  // The sources table has two text sizes: each cell's leading word or phrase, and small text for everything under it.
  assert.deepEqual(await sourcesPage.locator('#sources tbody :is(th > a:not([download]), th > .fold, .cell-main), #sources tbody :is(.json-link, .detail, .held, blockquote, .checked)').evaluateAll(es => [...new Set(es.map(e => getComputedStyle(e).fontSize))].sort()), ['13px', '16px'])
  assert.equal(await sourcesPage.locator('#sources .pill').count(), 0, 'Terms read as plain words, not pills')
  // Each lead is one whole line, in ink: at full width no lead is cut, and at the narrowest the table keeps every terms
  // and status word and every count. Every note shows in full; only the group rows fold.
  const leads = () => sourcesPage.locator('#sources .cell-main').evaluateAll(es => es.map(e => e.querySelector('span') ?? e).map(e => ({ text: e.textContent, cut: e.scrollWidth > e.clientWidth + .5, column: e.closest('td').cellIndex, height: Math.round(e.closest('.cell-main').getBoundingClientRect().height), color: getComputedStyle(e.closest('.cell-main')).color })))
  const full = await leads()
  assert.deepEqual(full.filter(l => l.cut).map(l => l.text), [], 'At full width every lead fits its column')
  assert.deepEqual([...new Set(full.map(l => l.height))].length, 1, 'Every lead is one line')
  assert.deepEqual([...new Set(full.map(l => l.color))], [await sourcesPage.locator('#sources tbody th').first().evaluate(e => getComputedStyle(e).color)], 'Every lead is ink, folding or not')
  await sourcesPage.setViewportSize({ width: 360, height: 720 })
  assert.deepEqual((await leads()).filter(l => l.cut && l.column !== 2).map(l => l.text), [], 'Narrow, the table scrolls and keeps every terms and status word whole')
  await sourcesPage.setViewportSize({ width: 1280, height: 720 })
  assert.equal(await sourcesPage.locator('#sources :is(details, summary)').count(), 0, 'Notes show unfolded')
  // A licence splits without loss or repetition: the lead is the ledger's text, or its start with the rest under it;
  // only a lead cut mid-clause (…) repeats the whole text under it.
  const licences = new Map(ledgerData.sources.map(s => [s.id, s.licence]))
  for (const [id, lead, rest] of await sourcesPage.locator('#sources tbody tr:not(.group-row)').evaluateAll(rows => rows.map(r => [r.id, ...[...r.children[2].children].map(e => e.textContent)]))) {
    const text = licences.get(id)
    if (!text) { assert.equal(lead, '–', id); continue }
    const upper = s => s.replace(/^./, c => c.toUpperCase())
    if (rest == null) assert.equal(lead, upper(text), id)
    else if (lead.endsWith('…')) assert.ok(rest === text && text.startsWith(lead.slice(0, -1)), id)
    else assert.ok(upper(text).startsWith(lead) && text.endsWith(rest) && text.slice(lead.length, text.length - rest.length).trim().replace(/^[,:;]$/, '') === '', `${id}: ${lead} | ${rest}`)
  }
  // Source names link the way matched fonts do, with the same underline, and read as the name alone, without an arrow.
  const linkLook = es => [...new Set(es.map(e => ['color', 'textDecorationLine', 'textDecorationColor', 'textUnderlineOffset'].map(k => getComputedStyle(e)[k]).join()))]
  const [matchLink] = await page.locator('.result .font-name a').evaluateAll(linkLook)
  assert.deepEqual(await sourcesPage.locator('#sources tbody th > a:not([download])').evaluateAll(linkLook), [matchLink])
  assert.equal(await sourcesPage.locator('#sources tbody th > a:not([download])').evaluateAll(links => links.filter(a => /[↗↓]/.test(a.textContent)).length), 0, 'Source names carry no arrow')
  // A catalog's size, the JSON download, sits under its family count, flush right; a group's arrow stands before its
  // name, in the indent, so the name lines up with its sources' names.
  assert.ok(await sourcesPage.locator('#sources tbody td:has(> .json-link)').evaluateAll(tds => tds.every(td => { const count = td.querySelector('.cell-main').getBoundingClientRect(), json = td.querySelector('.json-link').getBoundingClientRect(); return json.top >= count.bottom && Math.abs(json.right - count.right) < 1 && /^\d[\d.]* [KM]B$/.test(td.querySelector('.json-link').textContent) })), 'Each size sits under its family count')
  assert.ok(await sourcesPage.locator('.json-link').evaluateAll(links => links.every(a => a.children.length === 2 && a.firstElementChild.tagName === 'svg' && a.firstElementChild.getAttribute('aria-hidden') === 'true')), 'A download icon leads each size')
  assert.deepEqual(await sourcesPage.locator('#other').evaluate(row => {
    const left = e => { const r = document.createRange(); r.selectNodeContents(e); return Math.round(r.getBoundingClientRect().left) }, fold = row.querySelector('.fold')
    return [Math.round(fold.getBoundingClientRect().left) === Math.round(document.querySelector('#google-fonts > th > a').getBoundingClientRect().left), left(fold) === left(row.nextElementSibling.querySelector('th > a')), getComputedStyle(fold, '::after').content]
  }), [true, true, 'none'], 'A group arrow leads in the indent; its name lines up with its sources')
  assert.match(matchLink, /underline/)
  assert.equal(allowed[1], await sourcesPage.locator('.sources-table tbody td').first().evaluate(e => getComputedStyle(e).color))
  await sourcesPage.close()
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
  assert.deepEqual(await page.locator('.source-panel').evaluate(panel => [Math.round(panel.querySelector('.source-body').getBoundingClientRect().top), panel.querySelector('.column-head').getBoundingClientRect().bottom < 0]), [24, true], 'The preview and model input stick a padding from the top; the Source title scrolls away')
  // A window shorter than the preview and model input pins them by their bottom edge, a padding above it, so the model input stays in view.
  const stuck = () => page.evaluate(() => { const panel = document.querySelector('.source-body').getBoundingClientRect(), windows = document.querySelector('#normalized').getBoundingClientRect()
    return { bottom: Math.round(panel.bottom - innerHeight), top: Math.round(panel.top), windows: windows.top >= 0 && windows.bottom <= innerHeight } })
  await page.setViewportSize({ width: 1440, height: 560 }); await page.locator('#results .result').nth(40).scrollIntoViewIfNeeded()
  const short = await stuck(); assert.ok(short.bottom === -32 && short.top < 0 && short.windows, `Short window: ${JSON.stringify(short)}`)
  await page.setViewportSize({ width: 1440, height: 1000 }); await page.locator('#results .result').nth(30).scrollIntoViewIfNeeded()
  const tall = await stuck(); assert.ok(tall.top === 24 && tall.windows, `Tall window: ${JSON.stringify(tall)}`)
  // Rows past the fifth preview through a subset face of their own, read-only; the refused one names its absence.
  await page.waitForFunction(() => [...document.querySelectorAll('#results .result')].slice(5).every(r => r.querySelector('.result-unavailable') || r.querySelector('.result-preview')?.style.visibility === ''))
  // Each preview sets its line by its own font's ascent and descent, in one height, so none scrolls inside its field.
  const previews = await page.locator('#results .result-preview').evaluateAll(inputs => inputs.map(i => [i.closest('.result').querySelector('.font-name a').textContent, i.scrollHeight > i.clientHeight, i.clientHeight]))
  assert.deepEqual(previews.filter(p => p[1]).map(p => p[0]), [], 'No preview scrolls')
  assert.equal(new Set(previews.map(p => p[2])).size, 1, 'Every preview has one height')
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
  // All searches every shipped catalog as one; a Google family keeps its specimen link there. Menu rows centre their text.
  assert.ok(await page.locator('.catalog-option').evaluateAll(options => options.every(o => getComputedStyle(o).alignItems === 'center')), 'Catalog menu rows centre their name and count')
  await choose(page, 'all')
  const shippedFaces = (await Promise.all(data.catalogs.map(c => read(c.file)))).flatMap(c => c.faces)
  assert.equal(await page.locator('#catalog-size').textContent(), `All: ${new Set(shippedFaces.map(f => f.familyId)).size.toLocaleString('en-US')} families, ${shippedFaces.length.toLocaleString('en-US')} faces`)
  await page.waitForFunction(() => document.querySelectorAll('#results .result').length >= 5)
  assert.ok((await page.locator('#results .font-name a').first().getAttribute('href')).startsWith('https://fonts.google.com/specimen/'), 'A Google family links to its specimen under All')
  await choose(page, 'google-fonts')
  // The comparison is a table: gpu-font counts fonts as other finders do, every shipped face, and names the scripts the model detects.
  const compared = data.catalogs.reduce((n, c) => n + c.faces, 0)
  assert.equal(await page.locator('#compare #compared-fonts').textContent(), compared.toLocaleString('en-US'))
  if (artifact.heads) assert.equal(await page.locator('#compare #compared-scripts').textContent(), String(artifact.heads.script.labels.length))
  assert.deepEqual(await page.locator('#compare thead th').allTextContents(), ['Font finder', 'Fonts', 'Your fonts', 'Styles', 'Subsets', 'Runs on', 'Open source', 'Cost'])
  assert.ok(await page.locator('#compare tbody tr').evaluateAll(rows => rows.every(r => r.cells.length === 8 && r.cells[0].querySelector('.detail')?.textContent && r.cells[1].querySelector('.detail')?.textContent)), 'Each finder names its owner, and each count its sources')
  assert.equal(await page.locator('#compare a[href="./catalogs.html#my-fonts"]').count(), 1)
  for (const twin of await page.locator('.font-twins').all()) {
    await twin.hover(); assert.equal(await page.locator('#twins-tip').evaluate(t => t.matches(':popover-open')), true)
    assert.deepEqual(await page.locator('#twins-tip li').allTextContents(), JSON.parse(await twin.getAttribute('data-twins')), 'The tooltip lists every twin')
    await page.mouse.move(1, 1); assert.equal(await page.locator('#twins-tip').evaluate(t => t.matches(':popover-open')), false)
  }
  assert.equal(await page.locator('.results-column > .column-head > #matches-title + #catalog-control > #catalog-button').count(), 1, 'The catalog selector shares the Matches line')
  assert.deepEqual(await page.locator('#source-head').evaluate(h => [...h.children].filter(c => !c.matches('.sr-only')).map(c => c.id || c.className)), ['source-font', 'sample', 'source-spacer', 'draw', 'choose-image'], 'The current kind leads its name; the others follow')
  assert.equal(await page.locator('#tools').getAttribute('data-tool'), 'crop'); assert.ok(await page.locator('#undo').isDisabled())
  assert.ok(await page.locator('.pen-size').isHidden(), 'Brush size shows only with a drawing tool')
  // Each column names itself, Source and Matches; Model names its folded details and Input their windows.
  assert.deepEqual(await page.locator('.workbench .column-title').evaluateAll(t => t.map(e => [e.tagName, e.textContent, e.checkVisibility()])), [['H2', 'Source', true], ['H2', 'Matches', true]])
  assert.equal(new Set(await page.locator('.column-title, .metrics h3, .qa h3').evaluateAll(es => es.map(e => ['fontFamily', 'fontSize', 'color'].map(k => getComputedStyle(e)[k]).join()))).size, 1, 'Source and Matches share the serif subtitle of Accuracy and the questions')
  assert.deepEqual(await page.locator('#model-details > summary, #model-input .panel-head > :first-child, #catalog-control > :first-child').evaluateAll(es => es.map(e => e.firstChild.textContent)), ['Model', 'Input', 'Catalog'])
  assert.deepEqual(await page.locator('.workbench :is(.column-head, .panel-head, .input-meta)').evaluateAll(heads => heads.map(h => getComputedStyle(h).borderBottomStyle)), ['none', 'none', 'none', 'none', 'none'], 'Workbench heads carry no rule')
  assert.equal(await page.locator('.intro').evaluate(e => getComputedStyle(e).borderBottomStyle), 'solid', 'A rule separates the introduction from the workbench')
  assert.equal(await page.locator('#catalog-title').getAttribute('class'), 'sr-only', 'Catalog is named for assistive technology only; Source and Matches are visible column titles')
  assert.equal(await page.locator('#sample-label').textContent(), 'Lora')
  assert.deepEqual(await page.evaluate(() => ['source-font', 'draw', 'choose-image'].map(id => document.getElementById(id).getAttribute('aria-pressed'))), ['true', 'false', 'false'])
  const layout = await page.evaluate(() => Object.fromEntries(Object.entries({
    title: '.source-panel > .column-head', head: '.results-column > .column-head', source: '#source-head', verdict: '.results-column > .input-meta', model: '#model-details > summary',
    selector: '#catalog-button', kind: '#source-font', name: '#sample', draw: '#draw', open: '#choose-image', summary: '#result-summary', time: '#detection-time', stage: '#stage', match: '.result:first-child'
  }).map(([name, query]) => {
    const rect = document.querySelector(query).getBoundingClientRect()
    return [name, { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }]
  })))
  // Both columns open on two rows of one height: Source over the sample row; Matches with its catalog over the verdict.
  assert.ok(Math.abs(layout.title.top - layout.head.top) < 1 && Math.abs(layout.title.bottom - layout.head.bottom) < 1, 'Both column heads open on the same line, at one height')
  assert.ok(Math.abs(layout.head.right - layout.selector.right) < 1, 'The catalog selector ends the Matches line')
  assert.ok(Math.abs(layout.source.top - layout.verdict.top) < 1 && Math.abs(layout.source.bottom - layout.verdict.bottom) < 1, 'The verdict row sits level with the sample row')
  assert.ok(Math.abs(layout.summary.right - layout.head.right) < 1, 'The verdict ends its row, on the right')
  assert.ok(Math.abs(layout.time.right - layout.source.right) < 1, 'In Model, the time ends the source column, on the right')
  assert.ok(Math.abs(layout.source.left - layout.kind.left) < 1, 'The source switch starts its column')
  assert.ok(Math.abs(layout.source.right - layout.open.right) < 1, 'The other kinds end the source column')
  assert.ok(layout.kind.right <= layout.name.left && layout.name.right <= layout.draw.left && layout.draw.right <= layout.open.left && layout.draw.top >= layout.source.top - 1 && layout.draw.bottom <= layout.source.bottom + 1, 'Aa and its name lead the line; the sheet and image kinds follow on the right')
  assert.ok(Math.abs(layout.stage.top - layout.match.top) < 1, 'The first match starts level with the preview')
  assert.ok(layout.model.top - layout.stage.bottom < 40, 'Model follows the preview')
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
      // Each match shows the picture of its face stored with the site, or says it has none.
      const faces = foldTwins(value.matches, readCatalog(await read(option.file), data), { script: value.verdict?.script?.[0]?.label, limit: 5 }).map(m => m.face)
      await page.waitForFunction(() => [...document.querySelectorAll('#results .result')].every(previewState))
      assert.deepEqual(await page.locator('#results .result').evaluateAll(rows => rows.map(previewState)), await Promise.all(faces.map(async f => pictureless.has(f.id) ? 'No preview available' : `image ${base}/${await previewPath(f.id)}`)), option.id)
    }
  }
  assert.deepEqual((await result(page)).matches, original.matches)
  // A stored picture that fails to load gives way to the note, never to an empty row.
  await page.route('**/previews/**', route => route.fulfill({ status: 404, body: '' }))
  await choose(page, 'dafont'); await result(page)
  await page.waitForFunction(() => [...document.querySelectorAll('#results .result')].every(previewState))
  assert.deepEqual(await page.locator('#results .result').evaluateAll(rows => rows.map(previewState)), Array(5).fill('No preview available'))
  await page.unroute('**/previews/**'); await choose(page, data.catalogs[0].id)
  // Failed imports preserve the selected catalog and current answer.
  for (const payload of ['{', '{}', JSON.stringify({ ...await read(data.catalogs[0].file), encoderSha256: 'wrong' })]) {
    await page.locator('#catalog-file').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from(payload) })
    await page.waitForFunction(() => !document.querySelector('#catalog-error').hidden)
    assert.deepEqual((await result(page)).matches, original.matches)
  }
  const custom = await read(data.catalogs[0].file)
  // Smallest compatible catalog, retaining one real vector and exact owner.
  custom.faces = custom.faces.slice(0, 1)
  custom.vectors = { ...custom.vectors, shape: [1, custom.dimensions], data: Buffer.from(custom.vectors.data, 'base64').subarray(0, rowBytes(custom.vectors, custom.dimensions)).toString('base64'), scales: custom.vectors.scales.slice(0, 1), owners: [0] }
  await page.locator('#catalog-file').setInputFiles({ name: 'myfonts.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(custom)) })
  await page.waitForFunction(() => document.querySelector('#catalog-label').textContent === 'myfonts.json')
  const imported = await result(page)
  assert.deepEqual(await intro(page), expectedIntro, 'Importing a custom catalog changes neither the shipped figures nor the benchmark')
  assert.equal(imported.matches.length, 1); assert.equal(imported.matches[0].family, custom.faces[0].familyId)
  assert.equal(await page.locator('.result-preview, .reference-preview').count(), 0, 'Imported metadata must not claim a local specimen')
  assert.equal(await page.locator('#results .result').evaluate(previewState), 'No preview available', 'An imported face has no stored picture, and says so')
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
  assert.equal(await page.locator('.result').count(), 0); assert.ok(await page.locator('#save').isDisabled()); assert.ok(await page.locator('#save').isHidden(), 'No matches, no export'); assert.ok(await page.locator('.input-time').isHidden(), 'No time, no clock')
  assert.equal(await page.locator('#message').textContent(), '')
  assert.ok(await page.locator('#model-input').isVisible(), 'Unfolded Model keeps its input on an empty sheet')
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
  assert.ok(await blank.locator('#model-details > summary').isVisible(), 'Model is always in place')
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
  // A shared address shows the same matches without the image: ?style= carries the crop's style at int8 and re-encodes to
  // itself, so the opened page keeps the address it came from. Link copies the address; the default catalog is not named.
  const heads = readHeads(artifact), shared = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
  shared.on('page', p => p.on('pageerror', e => issues.push(e.message)))
  await shared.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: base })
  const sender = await shared.newPage(); await sender.goto(base); const sent = await result(sender)
  await sender.waitForFunction(() => new URLSearchParams(location.search).has('style'))
  const link = sender.url(), style = new URL(link).searchParams.get('style'), decoded = readStyle(style).embedding
  assert.equal(style, writeStyle(sent.embedding, data.modelSha256)); assert.equal(new URL(link).searchParams.has('catalog'), false, 'The default catalog needs no name')
  await sender.locator('#copy-link').click(); assert.equal(await sender.evaluate(() => navigator.clipboard.readText()), link, 'Link copies the address')
  const receiver = await shared.newPage(); await receiver.goto(link); const got = await result(receiver), google = readCatalog(await read(data.catalogs[0].file), data)
  assert.equal(got.shared, true); assert.deepEqual(got.embedding, Array.from(decoded)); assert.deepEqual(got.matches, matchCatalog(decoded, google, verdict(decoded, heads)))
  const top = value => foldTwins(value.matches, google, { script: value.verdict?.script?.[0]?.label, limit: 5 }).map(m => m.family)
  assert.deepEqual(top(got), top(sent), 'The receiver sees the sender\'s five families')
  assert.deepEqual([await receiver.locator('#message').textContent(), await receiver.locator('#empty').isVisible()], ['Matches from a shared link: it holds the crop’s style, not its image.', true])
  await receiver.waitForTimeout(700); assert.equal(receiver.url(), link, 'An opened link keeps its address')
  // Another catalog re-ranks the shared style, and the address names it.
  await choose(receiver, data.catalogs[1].id); const moved = await result(receiver)
  assert.deepEqual(moved.matches, matchCatalog(decoded, readCatalog(await read(data.catalogs[1].file), data), moved.verdict))
  await receiver.waitForFunction(id => new URLSearchParams(location.search).get('catalog') === id, data.catalogs[1].id)
  // One source of a grouped catalog searches alone, by its id: its faces only, under its name.
  const group = data.catalogs.find(c => c.sources?.some(s => s.id === 'collletttivo')), alone = await shared.newPage()
  await alone.goto(`${base}/?catalog=collletttivo&style=${style}`); const lone = await result(alone)
  assert.deepEqual(lone.matches, matchCatalog(decoded, subsetCatalog(readCatalog(await read(group.file), data), f => f.sourceId === 'collletttivo'), lone.verdict))
  assert.ok(lone.matches.length && lone.matches.every(m => m.face.sourceId === 'collletttivo'))
  assert.deepEqual([await alone.locator('#catalog-label').textContent(), lone.catalog.id], [group.sources.find(s => s.id === 'collletttivo').name, 'collletttivo'])
  // Embedded, the matcher alone: no masthead, introduction or sections; the catalog a plain label; links open new tabs,
  // and Link copies the whole page's address.
  const embed = await shared.newPage(); await embed.goto(`${base}/?catalog=collletttivo&embed`); await result(embed)
  assert.deepEqual(await embed.locator('.masthead, main > section').evaluateAll(es => es.filter(e => e.checkVisibility()).length), 0)
  assert.deepEqual([await embed.locator('#catalog-button').isDisabled(), await embed.locator('#catalog-button').innerText(), await embed.evaluate(() => document.querySelector('base')?.target)], [true, 'Collletttivo', '_blank'])
  assert.ok(await embed.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
  await embed.locator('#copy-link').click(); const copiedLink = new URL(await embed.evaluate(() => navigator.clipboard.readText()))
  assert.deepEqual([copiedLink.searchParams.get('catalog'), copiedLink.searchParams.has('style'), copiedLink.searchParams.has('embed')], ['collletttivo', true, false])
  // In another site's iframe, which denies clipboard access unless it allows it, the embed still runs and Link still copies.
  const site = base.replace('127.0.0.1', 'localhost'); await shared.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: site })
  const host = await shared.newPage(); await host.goto(`${site}/og.png`)
  await host.evaluate(src => document.body.append(Object.assign(document.createElement('iframe'), { src, style: 'width: 1100px; height: 950px' })), `${base}/?catalog=collletttivo&embed`)
  const framed = host.frameLocator('iframe'); await framed.locator('#copy-link:not([disabled])').waitFor({ timeout: 60000 })
  assert.equal(await host.frames()[1].evaluate(() => document.featurePolicy.allowsFeature('clipboard-write')), false)
  await framed.locator('#copy-link').click(); await framed.locator('#copy-link', { hasText: 'Copied' }).waitFor()
  assert.equal(new URL(await host.evaluate(() => navigator.clipboard.readText())).searchParams.get('catalog'), 'collletttivo')
  // A link it can't read, or made with another model, says so and shows no matches; an unknown catalog names what is searched instead.
  for (const [query, text] of [['style=abc', 'This link’s matches can’t be read.'], [`style=00000000.${style.slice(9)}`, 'This link comes from another version of the model, so its matches can’t be shown.']]) {
    const bad = await shared.newPage(); await bad.goto(`${base}/?${query}`); await bad.waitForFunction(() => document.body.dataset.ready === 'true')
    assert.deepEqual([await bad.locator('#message').textContent(), await bad.locator('.result').count(), await bad.locator('#empty').isVisible()], [text, 0, true], query)
  }
  const unknown = await shared.newPage(); await unknown.goto(`${base}/?catalog=nope`); await result(unknown)
  assert.equal(await unknown.locator('#catalog-error').textContent(), `No catalog “nope”; searching ${data.catalogs[0].name}.`)
  await shared.close()
  const cpu = await browser.newPage()
  await cpu.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }))
  await cpu.goto(base); const fallback = await result(cpu)
  assert.equal(fallback.backend, 'CPU'); assert.ok(error(fallback.embedding, original.embedding) < 1e-4)
  await cpu.close(); assert.deepEqual(issues, [])
  // My fonts, indexed in the page as the repository indexes Google Fonts: Lora Regular's static catalog instance, drawn and
  // embedded here, must equal the shipped encoder run on the repository's stored reference windows of lora/400 (the same
  // lines, rasterizer and preparation, pixel for pixel). The shipped rows sit about a hundredth away: they and the page's
  // rows are rounded to 4 bits separately, and a catalog exported before this encoder shipped embeds the checkpoint.
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
    // Download JSON hands over the stored catalog, the file the catalog menu and createMatcher() open.
    const [download] = await Promise.all([mine.waitForEvent('download'), mine.locator('#my-fonts-download').click()])
    assert.equal(download.suggestedFilename(), 'my-fonts.json'); assert.deepEqual(JSON.parse(await readFile(await download.path(), 'utf8')), (await stored()).catalog)
    const indexed = readCatalog((await stored()).catalog, data), shipped = readCatalog(await read(data.catalogs[0].file), data)
    const rowsOf = (catalog, test) => catalog.owners.flatMap((owner, r) => test(catalog.faces[owner]) ? [catalog.vectors.subarray(r * catalog.dimensions, (r + 1) * catalog.dimensions)] : [])
    const ours = rowsOf(indexed, f => f.family === 'Lora' && f.weight === 400 && f.style === 'normal'), theirs = rowsOf(shipped, f => f.id === 'lora/400')
    // The window file is too large to read whole; each needed window is read at its offset.
    const refs = await read('.data/style/references-browser.json'), refFile = openSync('.data/style/references-browser.u8', 'r')
    const refPixels = w => { const bytes = Buffer.alloc(w.width * w.height); readSync(refFile, bytes, 0, bytes.length, w.offset); return bytes }
    const faceIndex = (await read('.data/style/faces.json')).faces.findIndex(f => f.id === 'lora/400')
    const expected = ['Latn-lower', 'Latn-upper'].map(kind => unit(refs.samples.flatMap((sample, i) => sample.face === faceIndex && sample.kind === kind
      ? [embedWindows(refs.windows.filter(w => w.source === i).map(w => inferCPU(model, { width: w.width, height: w.height, pixels: Float32Array.from(refPixels(w), v => v / 255) })))] : [])
      .reduce((sum, e) => sum.map((v, d) => v + e[d]), new Float32Array(model.outputs))))
    closeSync(refFile)
    const cosine = (a, b) => a.reduce((sum, v, d) => sum + v * b[d], 0), near = ours.map(row => Math.max(...theirs.map(other => cosine(row, other))))
    assert.equal(ours.length, 2, 'Lowercase and capitals')
    // The browser packs its rows at 4 bits, so its integers are compared with the encoder's vector packed the same way:
    // equal, except where a value sits on a rounding boundary and float noise tips it one step.
    const saved = (await stored()).catalog, raw = saved.vectors, dims = saved.dimensions, packed = unpack(Buffer.from(raw.data, 'base64'), 4, raw.shape[0] * dims)
    const oursPacked = raw.owners.flatMap((owner, r) => { const f = saved.faces[owner]; return f.family === 'Lora' && f.weight === 400 && f.style === 'normal' ? [{ values: packed.subarray(r * dims, (r + 1) * dims), scale: raw.scales[r] }] : [] })
    expected.forEach((vector, i) => {
      const cpu = packVectors([Array.from(vector)]), values = unpack(Buffer.from(cpu.data, 'base64'), 4, dims), mine = oursPacked[i]
      assert.ok(Math.abs(mine.scale - cpu.scales[0]) <= cpu.scales[0] * 1e-4, `Browser-indexed Lora row ${i} scale ${mine.scale}, encoder ${cpu.scales[0]}`)
      const steps = Array.from(values, (v, d) => Math.abs(v - mine.values[d]))
      assert.ok(Math.max(...steps) <= 1 && steps.filter(Boolean).length <= 4, `Browser-indexed Lora row ${i} against the encoder on the stored reference windows: ${steps.filter(Boolean).length} values differ, by up to ${Math.max(...steps)} steps`)
    })
    // Two 4-bit roundings, the browser's and the shipped catalog's, each cost about a hundredth of cosine.
    assert.ok(near.every(c => c > .98), `Browser-indexed Lora against the shipped rows: ${near}`)
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
      // The installed face renders by a family CSS can read (its key, local:…, escaped), not the page's own font.
      assert.match(await search.locator('#results .result-preview').first().evaluate(input => getComputedStyle(input).fontFamily), /^"?specimen-local_3a_Georgia/)
    }
    await search.close()
    await mine.locator('#my-fonts-remove').click(); await mine.waitForFunction(() => !document.querySelector('.my-fonts-table') && document.querySelector('#my-fonts-stored').hidden)
    // A link to my fonts, opened where none are indexed (someone else's browser), says so and searches the default.
    const after = await context.newPage(); await after.goto(`${base}/?catalog=my-fonts`); await after.waitForFunction(() => document.body.dataset.ready === 'true')
    assert.ok(await after.locator('#add-my-fonts').evaluate(a => !a.hidden) && await after.locator('[data-catalog="my-fonts"]').count() === 0, 'Without my fonts the menu links to indexing')
    assert.deepEqual([await after.locator('#catalog-error').textContent(), await after.locator('#catalog-label').textContent()], [`My fonts aren’t indexed in this browser; searching ${data.catalogs[0].name}.`, data.catalogs[0].name])
    await context.close()
  } else console.log('My fonts differential skipped: no local catalog instances in .data/style')
  const report = { encoderSha256: data.encoderSha256, catalogs: data.catalogs.map(o => ({ id: o.id, sha256: o.sha256, families: o.families, faces: o.faces })), cpuError, gpuError,
    detectionMilliseconds: original.milliseconds, checks: ['native CPU/PyTorch/WebGPU projections', 'exact displayed tensors and input stats', 'catalog A → A → B → A', 'cached embedding reuse', 'one-face JSON import', 'invalid imports preserve result', 'stale catalog response', 'resolution and crop round trip', 'image button/switch/replacement', 'crop, pencil and eraser tools on any source, exact undo, brush size and cursor', 'twin tooltip and the list grown to 50 with subset previews', 'stored previews and their fallback', 'token contrast, links, popovers, code, section and question layout, dropdowns follow scroll', 'no captured specimens shipped', 'only cleared sources shipped', 'no fonts served, images only stored previews; fonts from Google', 'catalogs page', 'my fonts: browser-indexed Lora equals the encoder on stored references, join, installed faces, previews, removal', 'Aa returns to the last sample', 'narrow crop keeps layout', 'share links: address, copy, reopen, re-rank; one source alone; embed; unreadable and other-model links', 'my fonts download', 'removal requests', 'empty image/sample entry', 'corrupt image recovery', 'keyboard and responsive selectors', 'image before initial catalog', 'CPU fallback'],
    scope: 'Runtime correctness and UI lifecycle only; not recognition accuracy.' }
  await writeFile('bench/catalog-demo.json', JSON.stringify(report, null, 2) + '\n'); console.log(report)
} finally { await browser.close() }
