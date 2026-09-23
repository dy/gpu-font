// Does the text-style encoder already tell icons apart? A measurement, not a model.
//
// Six openly licensed icon sets, twenty shared concepts. References are drawn at
// 64 px, queries at 24 px (UI size) on a 2x screen, all rendered by Chromium from
// each set's public CSS. Every query is ranked against all references with the
// deployed encoder and, as a floor, with plain downsampled pixels.
import { chromium } from 'playwright'
import { readFile, writeFile } from 'node:fs/promises'
import { decodePng } from './preview-lib.mjs'
import { prepareLine } from '../src/line.mjs'
import { readNetwork, inferCPU } from '../src/network.mjs'
import { embedWindows } from '../src/catalog.mjs'

const SETS = {
  'material-outlined': { css: 'https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined', html: n => `<span style="font-family:'Material Symbols Outlined'">${n}</span>` },
  'material-rounded': { css: 'https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded', html: n => `<span style="font-family:'Material Symbols Rounded'">${n}</span>` },
  'material-sharp': { css: 'https://fonts.googleapis.com/css2?family=Material+Symbols+Sharp', html: n => `<span style="font-family:'Material Symbols Sharp'">${n}</span>` },
  'font-awesome': { css: 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css', html: n => `<i class="fa-solid fa-${n}"></i>` },
  bootstrap: { css: 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css', html: n => `<i class="bi bi-${n}"></i>` },
  phosphor: { css: 'https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.1/src/regular/style.css', html: n => `<i class="ph ph-${n}"></i>` },
}
// concept: [material ligature, Font Awesome, Bootstrap, Phosphor]
const CONCEPTS = {
  home: ['home', 'house', 'house', 'house'], search: ['search', 'magnifying-glass', 'search', 'magnifying-glass'],
  settings: ['settings', 'gear', 'gear', 'gear'], user: ['person', 'user', 'person', 'user'],
  heart: ['favorite', 'heart', 'heart', 'heart'], star: ['star', 'star', 'star', 'star'],
  trash: ['delete', 'trash', 'trash', 'trash'], edit: ['edit', 'pen', 'pencil', 'pencil'],
  menu: ['menu', 'bars', 'list', 'list'], close: ['close', 'xmark', 'x', 'x'],
  check: ['check', 'check', 'check', 'check'], back: ['arrow_back', 'arrow-left', 'arrow-left', 'arrow-left'],
  download: ['download', 'download', 'download', 'download-simple'], upload: ['upload', 'upload', 'upload', 'upload-simple'],
  mail: ['mail', 'envelope', 'envelope', 'envelope'], phone: ['call', 'phone', 'telephone', 'phone'],
  calendar: ['calendar_today', 'calendar', 'calendar', 'calendar'], camera: ['photo_camera', 'camera', 'camera', 'camera'],
  lock: ['lock', 'lock', 'lock', 'lock'], bell: ['notifications', 'bell', 'bell', 'bell'],
}
const nameFor = (set, concept) => CONCEPTS[concept][set.startsWith('material') ? 0 : { 'font-awesome': 1, bootstrap: 2, phosphor: 3 }[set]]

const encoder = JSON.parse(await readFile('models/encoder/encoder.json', 'utf8'))
const model = readNetwork(encoder)
const embed = image => {
  const prepared = prepareLine(image, { deskew: true, sampler: 'windows' })
  if (prepared.status !== 'ok' || !prepared.windows.length) return null
  return embedWindows(prepared.windows.slice(0, 3).map(window => inferCPU(model, window)))
}
// Floor: the ink box resampled to 32×32 grey, mean-removed and normalised.
const pixels = ({ width, height, channels, pixels: data }) => {
  const gray = (x, y) => { const i = (y * width + x) * channels; return channels >= 3 ? (data[i] + data[i + 1] + data[i + 2]) / 3 : data[i] }
  let x0 = width, y0 = height, x1 = -1, y1 = -1
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (gray(x, y) < 200) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y) }
  if (x1 < 0) return null
  const side = Math.max(x1 - x0, y1 - y0) + 1, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, out = new Float32Array(32 * 32)
  for (let v = 0; v < 32; v++) for (let u = 0; u < 32; u++) {
    const x = Math.round(cx - side / 2 + (u + 0.5) * side / 32), y = Math.round(cy - side / 2 + (v + 0.5) * side / 32)
    out[v * 32 + u] = x < 0 || y < 0 || x >= width || y >= height ? 0 : (255 - gray(x, y)) / 255
  }
  const mean = out.reduce((a, b) => a + b, 0) / out.length, norm = Math.hypot(...out.map(v => v - mean)) || 1
  return out.map(v => (v - mean) / norm)
}
const toRgba = image => {
  const data = new Uint8ClampedArray(image.width * image.height * 4)
  for (let i = 0; i < image.width * image.height; i++) {
    const at = i * image.channels
    data.set(image.channels >= 3 ? [image.pixels[at], image.pixels[at + 1], image.pixels[at + 2]] : [image.pixels[at], image.pixels[at], image.pixels[at]], i * 4)
    data[i * 4 + 3] = image.channels === 4 ? image.pixels[at + 3] : 255
  }
  return { width: image.width, height: image.height, data }
}

const browser = await chromium.launch()
const page = await (await browser.newContext({ deviceScaleFactor: 2, viewport: { width: 400, height: 300 } })).newPage()
await page.setContent(`<!doctype html><meta charset=utf-8>${Object.values(SETS).map(set => `<link rel=stylesheet href="${set.css}">`).join('')}
<style>body{margin:0;background:#fff;color:#111}#box{display:inline-flex;align-items:center;justify-content:center;padding:24px;line-height:1}#box *{font-size:inherit}</style><div id=box></div>`)
await page.waitForLoadState('networkidle')

const shots = []
for (const set of Object.keys(SETS)) for (const concept of Object.keys(CONCEPTS)) for (const [role, size] of [['reference', 64], ['query', 24]]) {
  await page.evaluate(({ html, size }) => {
    const box = document.getElementById('box'); box.style.fontSize = `${size}px`; box.innerHTML = html
  }, { html: SETS[set].html(nameFor(set, concept)), size })
  await page.evaluate(() => document.fonts.ready)
  await page.waitForTimeout(80)
  const image = decodePng(await page.locator('#box').screenshot())
  shots.push({ set, concept, role, size, embedding: embed(toRgba(image)), pixels: pixels(image) })
}
await browser.close()

const blank = shots.filter(shot => !shot.pixels).map(shot => `${shot.set}/${shot.concept}/${shot.role}`)
const references = shots.filter(shot => shot.role === 'reference' && shot.embedding && shot.pixels)
const queries = shots.filter(shot => shot.role === 'query' && shot.embedding && shot.pixels)
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0)
const score = key => {
  const tally = { exactTop1: 0, exactTop5: 0, setTop1: 0, setTop5Majority: 0, conceptTop1: 0 }
  for (const query of queries) {
    const ranked = references.map(reference => ({ reference, similarity: dot(query[key], reference[key]) })).sort((a, b) => b.similarity - a.similarity)
    const same = r => r.set === query.set && r.concept === query.concept
    if (same(ranked[0].reference)) tally.exactTop1++
    if (ranked.slice(0, 5).some(r => same(r.reference))) tally.exactTop5++
    if (ranked[0].reference.set === query.set) tally.setTop1++
    if (ranked.slice(0, 5).filter(r => r.reference.set === query.set).length >= 3) tally.setTop5Majority++
    if (ranked[0].reference.concept === query.concept) tally.conceptTop1++
  }
  const rates = Object.fromEntries(Object.entries(tally).map(([k, v]) => [k, Math.round(v / queries.length * 1000) / 10]))
  // Where the misses go: the set a query's nearest reference came from.
  const confusions = {}
  for (const query of queries) {
    const nearest = references.map(reference => ({ reference, similarity: dot(query[key], reference[key]) })).sort((a, b) => b.similarity - a.similarity)[0].reference
    if (nearest.set === query.set && nearest.concept === query.concept) continue
    const pair = `${query.set} → ${nearest.set}`
    confusions[pair] = (confusions[pair] ?? 0) + 1
  }
  return { ...rates, confusions }
}
const report = {
  what: 'Icon retrieval with the deployed text-style encoder versus a raw-pixel floor. Chance: exact 1/120, set 1/6, concept 1/20.',
  sets: Object.keys(SETS), concepts: Object.keys(CONCEPTS).length, references: references.length, queries: queries.length, blank,
  encoderSha256: (await import('node:crypto')).createHash('sha256').update(await readFile('models/encoder/encoder.json')).digest('hex'),
  encoder: score('embedding'), pixels: score('pixels'),
}
await writeFile('bench/glyph-probe.json', JSON.stringify(report, null, 2) + '\n')
console.log(JSON.stringify({ references: report.references, queries: report.queries, blank: report.blank.length, encoder: report.encoder, pixels: report.pixels }, null, 1))
