// Web pages as people screenshot them: the crops scripts/web_screens.mjs cut from the homepages of the Chrome UX Report's
// top sites, each labelled by the font file Chromium drew it with (CDP CSS.getPlatformFontsForNode), so the label is the
// browser's own and not a guess. gpu-font reads every crop whose font a shipped catalog holds. No crop was trained on:
// the whole set is a test. The crops are other people's designs and stay in .data/screens.
//
//   node scripts/screens.mjs [--until 2026-09-27T12:00:00Z]     needs `npm run demo`
import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { matchPhotos, rankOf, catalogNames, normalize, accuracy } from './photos.mjs'

const ROOT = '.data/screens'
// What a font file adds to its family's name: a weight, a width, a slant, an optical size ("Google Sans 18pt").
const STYLE = /(\s+(thin|hairline|extra\s?light|ultra\s?light|light|regular|normal|book|medium|semi\s?bold|demi\s?bold|demi|bold|extra\s?bold|ultra\s?bold|black|heavy|italic|oblique|condensed|narrow|\d+pt|\d{3}))+$/i

const unnamed = name => !name || !name.trim() || /[\x00-\x1f\x7f]/.test(name)  // nothing, or bytes where a name should be: a subset web font
/** The catalog family a drawn font belongs to: its own name when indexed, else the name without its style words; null when
 * neither is indexed or the name is empty or garbled (a subset web font named by bytes). */
export function familyOf(name, names) {
  if (unnamed(name) || !normalize(name)) return null
  return names.get(normalize(name)) ?? names.get(normalize(name.replace(STYLE, ''))) ?? null
}

/** Records captured up to `until`, each with the catalog family of its font; the rest counted by why they were left. */
export function indexed(records, names, until = Infinity) {
  const kept = [], left = { later: 0, unnamed: 0, 'not indexed': 0 }
  for (const record of records) {
    if (Date.parse(record.capturedAt) > until) { left.later++; continue }
    if (unnamed(record.font?.family)) { left.unnamed++; continue }
    const truth = familyOf(record.font.family, names)
    if (truth) kept.push({ record, truth }); else left['not indexed']++
  }
  return { kept, left }
}

/** Accuracy over crops and over families (scripts/photos.mjs), with the sites the crops came from. */
export const summary = list => { const { families, ...rest } = accuracy(list); return { crops: list.length, families, sites: new Set(list.map(s => s.record.origin)).size, ...rest } }

async function main() {
  const at = process.argv.indexOf('--until'), until = at < 0 ? Date.now() : Date.parse(process.argv[at + 1])
  if (!Number.isFinite(until)) throw new Error('--until takes a date, as 2026-09-27T12:00:00Z')
  const records = (await readFile(`${ROOT}/records.jsonl`, 'utf8')).trim().split('\n').map(line => JSON.parse(line))
  const { kept, left } = indexed(records, await catalogNames(), until), read = records.filter(r => Date.parse(r.capturedAt) <= until)
  const { browser, adapter, encoderSha256, results } = await matchPhotos(kept.map(({ record }) => ({ path: `${ROOT}/${record.image}`, box: [0, 0, 1e9, 1e9] })))
  const scored = kept.map((item, i) => ({ ...item, ...results[i], rank: rankOf(results[i].rows, item.truth.family) }))
  const slices = key => Object.fromEntries([...new Set(scored.map(key))].sort().map(value => [value, summary(scored.filter(s => key(s) === value))]))
  const size = px => px < 14 ? 'under 14 px' : px < 20 ? '14–19 px' : px < 32 ? '20–31 px' : '32 px and over'
  const times = scored.map(s => s.ms).sort((a, b) => a - b), time = p => Math.round(times[Math.floor(p * (times.length - 1))])
  const report = {
    set: { name: 'Web screens', collector: 'scripts/web_screens.mjs', sites: 'homepages of the Chrome UX Report\'s top origins, where robots.txt allows',
      label: 'the font file Chromium drew the line with (CSS.getPlatformFontsForNode), one font a line',
      until: new Date(until).toISOString(), recordsSha256: createHash('sha256').update(read.map(r => JSON.stringify(r)).join('\n')).digest('hex'),
      crops: read.length, sites_visited: new Set(read.map(r => r.origin)).size, indexed: { crops: kept.length, families: new Set(kept.map(k => k.truth.family)).size }, left },
    rule: 'Right when the drawn font\'s family, or a family the page folds into that row, is among the first rows the page shows (5); any weight counts.',
    crop: 'The whole crop: one line of text as the collector cut it.',
    catalogs: 'All: every shipped catalog searched as one.', browser, adapter, encoderSha256, date: new Date().toISOString().slice(0, 10),
    timeMs: { median: time(0.5), p95: time(0.95), from: 'image bytes to ranked rows, warm' },
    unread: Object.fromEntries([...new Set(scored.map(s => s.status))].filter(s => s !== 'ok').map(s => [s, scored.filter(x => x.status === s).length])),
    results: { all: summary(scored), catalog: slices(s => s.truth.catalog), font: slices(s => s.record.font.custom ? 'web font' : 'installed font'),
      size: slices(s => size(s.record.css.fontSize)), length: slices(s => s.record.text.length < 10 ? 'under 10 characters' : s.record.text.length < 20 ? '10–19 characters' : '20 characters and over') }
  }
  // One crop a line, so the report stays readable and diffs stay small; the site and the text stay out: they are the page's.
  const rows = scored.map(s => JSON.stringify({ family: s.truth.family, px: s.record.css.fontSize, ...(s.status !== 'ok' && { status: s.status }), rank: s.rank, first: s.rows[0]?.name ?? null,
    score: s.rows[0] ? Math.round(s.rows[0].score * 1e3) / 1e3 : null }))
  await writeFile('bench/screens.json', JSON.stringify(report, null, 1).slice(0, -2) + `,\n "crops": [\n  ${rows.join(',\n  ')}\n ]\n}\n`)
  const pct = v => v === null ? '–' : `${(100 * v).toFixed(1)}%`
  for (const [name, r] of [['all', report.results.all], ...Object.entries(report.results.font), ...Object.entries(report.results.size)])
    console.log(`${name}: ${r.crops} crops, ${r.families} families: top-1 ${pct(r.top1)}, top-5 ${pct(r.top5)}; by family top-1 ${pct(r.byFamily.top1)}, top-5 ${pct(r.byFamily.top5)}`)
  console.log(`median ${report.timeMs.median} ms, p95 ${report.timeMs.p95} ms on ${adapter}`)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
