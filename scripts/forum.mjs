// The DaFont forum's identification requests as a read: every thread scripts/dafont_forum.py scraped whose first post the
// forum settled on a font, the picture the asker posted, the line the answer names (scripts/forum_boxes.py's Vision
// boxes; the line sharing most words with the text the answerer previewed the font with, else the tallest confident
// horizontal one), labelled by the answer when a shipped catalog holds it. Threads split by the hash of their id: the
// held-out tenth is never trained on and is the read; the rest is development, which train.style_forum packs as views.
// Max Halford's older threads (scripts/dafont.mjs) are a read of their own, hand-cropped.
//
//   node scripts/forum.mjs [--limit 1000]     needs `npm run demo`; the newest threads first
import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { cleanFlag, matchPhotos, rankOf, twinSets, catalogNames, accuracy } from './photos.mjs'
import { familyOf } from './screens.mjs'

const ROOT = '.data/forum/dafont'
const CONFIDENT = .5, PAD = .25

/** A thread's role: one in ten by the hash of its id is `held-out`, never trained on; the rest `development`, which
 * train.style_forum packs as training views. The same rule as the sites' and the open families'. */
export const threadRole = id => parseInt(createHash('sha1').update(String(id)).digest('hex').slice(0, 8), 16) % 10 === 0 ? 'held-out' : 'development'

const words = text => new Set((text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []))
const height = line => line.box[3] - line.box[1]

/** The boxed line the answer is about: among the confident horizontal lettered ones, the line sharing most words with the
 * text the answerer previewed the font with (the link's text=), else the tallest; null when no line qualifies. */
export function lineOf(lines, hint = '') {
  const kept = lines.filter(l => l.confidence >= CONFIDENT && l.box[2] - l.box[0] >= height(l) && height(l) > 0 && words(l.text).size)
  if (!kept.length) return null
  const want = words(hint), shared = kept.map(l => [...words(l.text)].filter(w => want.has(w)).length), most = Math.max(...shared)
  return most ? kept[shared.indexOf(most)] : kept.reduce((a, b) => height(b) > height(a) ? b : a)
}

/** The line's box with a little of the picture around it, as a person would cut it. */
export const padded = ([x0, y0, x1, y1]) => { const pad = Math.round((y1 - y0) * PAD); return [x0 - pad, y0 - pad, x1 + pad, y1 + pad] }

export const previewText = url => { try { return new URL(url).searchParams.get('text') ?? '' } catch { return '' } }

/** Held-out identified threads with a boxed picture and an answer a catalog holds; the rest counted by why they were left. */
export function indexed(requests, boxes, names) {
  const kept = [], left = { development: 0, unboxed: 0, 'no line': 0, 'not indexed': 0 }
  for (const request of requests) {
    if (!request.identified?.length || !request.files?.length) continue
    if (threadRole(request.id) !== 'held-out') { left.development++; continue }
    const found = boxes.get(request.files[0].file)
    if (!found?.lines) { left.unboxed++; continue }
    const answer = request.identified[0], line = lineOf(found.lines, previewText(answer.url))
    if (!line) { left['no line']++; continue }
    const truth = familyOf(answer.name, names)
    if (truth) kept.push({ request, answer, line, truth }); else left['not indexed']++
  }
  return { kept, left }
}

export const summary = list => { const { families, ...rest } = accuracy(list); return { threads: list.length, families, ...rest } }

const host = url => { try { return new URL(url).host } catch { return 'none' } }

async function main() {
  const at = process.argv.indexOf('--limit'), limit = at < 0 ? Infinity : Number(process.argv[at + 1])
  if (!(limit > 0)) throw new Error('--limit takes a count of threads')
  const jsonl = async file => (await readFile(`${ROOT}/${file}`, 'utf8')).trim().split('\n').filter(Boolean).map(line => JSON.parse(line))
  const requests = await jsonl('requests.jsonl'), boxes = new Map((await jsonl('boxes.jsonl')).map(b => [b.file, b]))
  const all = indexed(requests, boxes, await catalogNames()), kept = all.kept.slice(0, limit)
  const { clean, report: target } = cleanFlag('forum'), { browser, adapter, encoderSha256, results } = await matchPhotos(kept.map(({ request, line }) => ({ path: `${ROOT}/images/${request.files[0].file}`, box: padded(line.box) })), { clean })
  const twins = await twinSets(), scored = kept.map((item, i) => ({ ...item, ...results[i], rank: rankOf(results[i].rows, item.truth.family, twins.get(item.truth.familyId)) }))
  const slices = key => Object.fromEntries([...new Set(scored.map(key))].sort().map(value => [value, summary(scored.filter(s => key(s) === value))]))
  const size = px => px < 24 ? 'under 24 px' : px < 48 ? '24–47 px' : px < 96 ? '48–95 px' : '96 px and over'
  const times = scored.map(s => s.ms).sort((a, b) => a - b), time = p => Math.round(times[Math.floor(p * (times.length - 1))])
  const report = {
    set: { name: 'DaFont forum threads, held out', collector: 'scripts/dafont_forum.py', boxes: 'scripts/forum_boxes.py', forum: 'https://www.dafont.com/forum/?f=1',
      label: 'the font the thread settled on (the first post\'s "Identified font"), linked to its page by the answerer',
      requestsSha256: createHash('sha256').update(requests.map(r => JSON.stringify(r)).join('\n')).digest('hex'),
      threads: requests.length, identified: requests.filter(r => r.identified?.length).length, heldOut: all.kept.length + all.left.unboxed + all.left['no line'] + all.left['not indexed'],
      indexed: { threads: all.kept.length, families: new Set(all.kept.map(k => k.truth.family)).size }, read: kept.length, left: all.left },
    rule: 'Right when the answer\'s family, or a family the page folds into that row, or a twin the teacher cannot tell from it, is among the first rows the page shows (5); any weight counts.',
    crop: 'The Vision line box the answer names (the line sharing most words with the text the answerer previewed, else the tallest confident horizontal one), a quarter of its height around it.',
    catalogs: 'All: every shipped catalog searched as one.', ...(clean && { cleaner: clean }), browser, adapter, encoderSha256, date: new Date().toISOString().slice(0, 10),
    timeMs: { median: time(0.5), p95: time(0.95), from: 'image bytes to ranked rows, warm' },
    unread: Object.fromEntries([...new Set(scored.map(s => s.status))].filter(s => s !== 'ok').map(s => [s, scored.filter(x => x.status === s).length])),
    results: { all: summary(scored), catalog: slices(s => s.truth.catalog), host: slices(s => host(s.answer.url)), hint: slices(s => previewText(s.answer.url) ? 'preview text' : 'tallest line'),
      size: slices(s => size(height(s.line))), length: slices(s => s.line.text.length < 6 ? 'under 6 characters' : s.line.text.length < 12 ? '6–11 characters' : '12 characters and over') }
  }
  // One thread a line; the picture and its text stay out: they are the poster's.
  const rows = scored.map(s => JSON.stringify({ thread: s.request.id, family: s.truth.family, catalog: s.truth.catalog, px: height(s.line), ...(s.status !== 'ok' && { status: s.status }), rank: s.rank,
    first: s.rows[0]?.name ?? null, score: s.rows[0] ? Math.round(s.rows[0].score * 1e3) / 1e3 : null }))
  await writeFile(target, JSON.stringify(report, null, 1).slice(0, -2) + `,\n "threads": [\n  ${rows.join(',\n  ')}\n ]\n}\n`)
  const pct = v => v === null ? '–' : `${(100 * v).toFixed(1)}%`
  for (const [name, r] of [['all', report.results.all], ...Object.entries(report.results.catalog), ...Object.entries(report.results.hint), ...Object.entries(report.results.size)])
    console.log(`${name}: ${r.threads} threads, ${r.families} families: top-1 ${pct(r.top1)}, top-5 ${pct(r.top5)}; by family top-1 ${pct(r.byFamily.top1)}, top-5 ${pct(r.byFamily.top5)}`)
  console.log(`median ${report.timeMs.median} ms, p95 ${report.timeMs.p95} ms on ${adapter}`)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
