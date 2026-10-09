// The same-text stage and the line reader on the final photographs: WhatFontIs-Bench pictures of held-out Google families,
// each ranked three ways, by the catalog alone, by the catalog plus the shortlist set in the picture's own text (the
// label's), and plus the shortlist set in the text the shipped reader read from the crop. The reader is read too: its
// character error against the label, with accents and characters outside its classes folded away as its labels are.
//
//   node scripts/sametext.mjs [--sample 400] [--seed 1]     needs `npm run demo`
import { writeFile } from 'node:fs/promises'
import { matchPhotos, rankOf, twinSets, accuracy } from './photos.mjs'
import { whatfontisSet } from './whatfontis.mjs'

const CLASSES = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

/** The label's text as the reader could read it: accents folded to their letters, characters outside its classes dropped. */
export const readable = text => [...text.normalize('NFKD')].filter(c => !/\p{M}/u.test(c)).filter(c => CLASSES.includes(c)).join('').trim()

export function distance(a, b) {
  let row = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    const next = [i]
    for (let j = 1; j <= b.length; j++) next.push(Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)))
    row = next
  }
  return row[b.length]
}

/** A seeded shuffle's first `count` items (Fisher–Yates with a small linear congruential generator). */
export function sample(items, count, seed = 1) {
  const out = items.slice(); let state = seed >>> 0 || 1
  const next = () => (state = (1664525 * state + 1013904223) >>> 0) / 2 ** 32
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(next() * (i + 1)); [out[i], out[j]] = [out[j], out[i]] }
  return out.slice(0, count)
}

export const summary = list => { const { families, ...rest } = accuracy(list); return { photographs: list.length, families, ...rest } }

async function main() {
  const arg = (name, fallback) => { const at = process.argv.indexOf(name); return at < 0 ? fallback : Number(process.argv[at + 1]) }
  const count = arg('--sample', 400), seed = arg('--seed', 1)
  const finals = (await whatfontisSet()).indexed.filter(item => item.final && item.truth.catalog === 'google-fonts' && item.label.text)
  const items = sample(finals, count, seed).map(item => ({ ...item, text: item.label.text }))
  const { browser, adapter, encoderSha256, results } = await matchPhotos(items, { limit: 20, text: true })
  const twins = await twinSets()
  const scored = items.map((item, i) => {
    const r = results[i], credit = twins.get(item.truth.familyId), rank = rows => rankOf(rows ?? [], item.truth.family, credit)
    const want = readable(item.text), read = r.read ?? ''
    return { ...item, ...r, ranks: { catalog: rank(r.rows), byText: rank(r.byText), byRead: rank(r.byRead) }, want, cer: want.length ? distance(read, want) / want.length : null, exact: read === want }
  })
  const by = key => summary(scored.map(s => ({ ...s, rank: s.ranks[key] })))
  const read = scored.filter(s => s.cer !== null), cer = read.reduce((sum, s) => sum + s.cer, 0) / Math.max(1, read.length)
  const ms = key => { const times = scored.map(s => s[key]).filter(Number.isFinite).sort((a, b) => a - b); return times.length ? Math.round(times[Math.floor(times.length / 2)]) : null }
  const report = {
    set: { name: 'WhatFontIs-Bench final photographs of Google families with a text label', sampled: items.length, of: finals.length, seed },
    rule: 'Right when the family, a family folded into its row, or a twin the teacher cannot tell from it, is among the first rows (5); ranked three ways over the same crop and embedding.',
    stages: { catalog: 'the catalog\'s rows alone', byText: 'the first 30 rows re-sorted by catalog score plus the cosine with the label\'s text drawn in each face Google Fonts serves (src/sametext.mjs)', byRead: 'the same, with the text the line reader read from the crop (models/reader/reader.json)' },
    browser, adapter, encoderSha256, date: new Date().toISOString().slice(0, 10),
    timeMs: { match: ms('ms'), read: ms('readMs'), text: ms('textMs'), from: 'per photograph, warm; `text` covers both same-text rankings' },
    reader: { characterError: Math.round(cer * 1e4) / 1e4, exact: Math.round(1e4 * read.filter(s => s.exact).length / Math.max(1, read.length)) / 1e4, labels: read.length },
    results: { catalog: by('catalog'), byText: by('byText'), byRead: by('byRead') }
  }
  const rows = scored.map(s => JSON.stringify({ id: s.label.id, family: s.truth.family, text: s.want, read: s.read, ...(s.status !== 'ok' && { status: s.status }), ...s.ranks }))
  await writeFile('bench/sametext.json', JSON.stringify(report, null, 1).slice(0, -2) + `,\n "photographs": [\n  ${rows.join(',\n  ')}\n ]\n}\n`)
  const pct = v => v === null ? '–' : `${(100 * v).toFixed(1)}%`
  for (const [name, r] of Object.entries(report.results)) console.log(`${name}: ${r.photographs} photographs, ${r.families} families: top-1 ${pct(r.top1)}, top-5 ${pct(r.top5)}, top-20 ${pct(r.top20)}; by family top-1 ${pct(r.byFamily.top1)}, top-5 ${pct(r.byFamily.top5)}`)
  console.log(`reader: character error ${pct(report.reader.characterError)}, exact ${pct(report.reader.exact)} on ${report.reader.labels} labels; median match ${report.timeMs.match} ms, read ${report.timeMs.read} ms, text ${report.timeMs.text} ms on ${adapter}`)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
