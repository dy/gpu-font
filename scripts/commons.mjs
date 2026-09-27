// Typefaces in use, photographed: signs, packaging and print from Wikimedia Commons (scripts/commons_typefaces.py), every
// photo under a free licence and filed by Commons under its typeface's category. The label is that category, so it names
// the photo and not a line: gpu-font reads the tallest line of text found in each photo (scripts/text_boxes.py), taking
// the most prominent lettering to be what the photo was filed for. A weaker label than WhatFontIs-Bench's, from pictures
// nobody made for a benchmark. No photo was trained on: the whole set is a test.
//
//   node scripts/commons.mjs     needs `npm run demo`
import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { matchPhotos, rankOf, catalogNames, normalize, accuracy } from './photos.mjs'

const ROOT = '.data/photos/commons', CONFIDENCE = .5, LETTERS = 3

/** The line a photo is read by: the tallest one found with some confidence and at least three characters; null without one. */
export function line(photo) {
  const lines = photo.lines.filter(l => l.confidence >= CONFIDENCE && [...l.text.replace(/\s/g, '')].length >= LETTERS)
  return lines.reduce((best, l) => !best || l.box[3] - l.box[1] > best.box[3] - best.box[1] ? l : best, null)
}

/** Photos whose typeface a shipped catalog holds and that show a line to read; the rest counted by why they were left. */
export function indexed(photos, names) {
  const kept = [], left = { 'not indexed': 0, 'no line': 0 }
  for (const photo of photos) {
    const truth = names.get(normalize(photo.typeface)), read = truth && line(photo)
    if (!truth) left['not indexed']++; else if (!read) left['no line']++; else kept.push({ photo, truth, line: read })
  }
  return { kept, left }
}

export const summary = list => { const { families, ...rest } = accuracy(list); return { photos: list.length, typefaces: families, ...rest } }

async function main() {
  const text = await readFile(`${ROOT}/boxes.jsonl`, 'utf8'), photos = text.trim().split('\n').map(l => JSON.parse(l))
  const info = new Map((await readFile(`${ROOT}/info.jsonl`, 'utf8')).trim().split('\n').map(l => JSON.parse(l)).map(i => [i.title, i]))
  const { kept, left } = indexed(photos, await catalogNames())
  const { browser, adapter, encoderSha256, results } = await matchPhotos(kept.map(k => ({ path: `${ROOT}/images/${k.photo.file}`, box: k.line.box })))
  const scored = kept.map((item, i) => ({ ...item, ...results[i], rank: rankOf(results[i].rows, item.truth.family) }))
  const slices = (key, list = scored) => Object.fromEntries([...new Set(list.map(key))].sort().map(value => [value, summary(list.filter(s => key(s) === value))]))
  const often = new Set([...scored.reduce((m, s) => m.set(s.truth.family, (m.get(s.truth.family) || 0) + 1), new Map())].filter(([, n]) => n >= 10).map(([family]) => family))
  const times = scored.map(s => s.ms).sort((a, b) => a - b), time = p => Math.round(times[Math.floor(p * (times.length - 1))])
  const report = {
    set: { name: 'Wikimedia Commons, typefaces by name', source: 'https://commons.wikimedia.org/wiki/Category:Typefaces_by_name', collector: 'scripts/commons_typefaces.py',
      licence: 'each photo\'s own free licence, named on its Commons page', boxesSha256: createHash('sha256').update(text).digest('hex'),
      photos: photos.length, typefaces: new Set(photos.map(p => p.typeface)).size, indexed: { photos: kept.length, typefaces: new Set(kept.map(k => k.truth.family)).size }, left },
    label: 'The Commons category the photo is filed under. It names the photo, not the line read: a weak label.',
    rule: 'Right when the typeface, or a family the page folds into that row, is among the first rows the page shows (5); any weight counts.',
    crop: `The tallest line of text found in the photo with confidence ${CONFIDENCE} or more and at least ${LETTERS} characters.`,
    catalogs: 'All: every shipped catalog searched as one.', browser, adapter, encoderSha256, date: new Date().toISOString().slice(0, 10),
    timeMs: { median: time(0.5), p95: time(0.95), from: 'image bytes to ranked rows, warm' },
    unread: Object.fromEntries([...new Set(scored.map(s => s.status))].filter(s => s !== 'ok').map(s => [s, scored.filter(x => x.status === s).length])),
    results: { all: summary(scored), catalog: slices(s => s.truth.catalog), typeface: slices(s => s.truth.family, scored.filter(s => often.has(s.truth.family))) }
  }
  // One photo a line, named by its Commons title, which is also where its author and licence are.
  const rows = scored.map(s => JSON.stringify({ title: s.photo.title, licence: info.get(s.photo.title)?.licence ?? null, typeface: s.truth.family, ...(s.status !== 'ok' && { status: s.status }), rank: s.rank,
    first: s.rows[0]?.name ?? null, score: s.rows[0] ? Math.round(s.rows[0].score * 1e3) / 1e3 : null }))
  await writeFile('bench/commons.json', JSON.stringify(report, null, 1).slice(0, -2) + `,\n "photos": [\n  ${rows.join(',\n  ')}\n ]\n}\n`)
  const pct = v => v === null ? '–' : `${(100 * v).toFixed(1)}%`
  for (const [name, r] of [['all', report.results.all], ...Object.entries(report.results.catalog)])
    console.log(`${name}: ${r.photos} photos, ${r.typefaces} typefaces: top-1 ${pct(r.top1)}, top-5 ${pct(r.top5)}; by typeface top-1 ${pct(r.byFamily.top1)}, top-5 ${pct(r.byFamily.top5)}`)
  console.log(`median ${report.timeMs.median} ms, p95 ${report.timeMs.p95} ms on ${adapter}`)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
