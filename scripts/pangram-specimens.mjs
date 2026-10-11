// Pangram Pangram's own product pictures (.data/pangram/specimens, 728 of 209 products, local only), read as the page reads
// a crop: the whole picture, and the tallest text line macOS Vision found in it (scripts/text_boxes.py) with a quarter of
// its height around, each searched across every shipped catalog with twin credit. A picture is right when any family of
// the product's design (the inventory's `design`) is among the first rows. The foundry's acceptance read: every picture in
// the first ten.
//
//   node scripts/pangram-specimens.mjs [--limit 100] [--clean <cleaner>]     with `npm run demo` running
import { readFile, writeFile } from 'node:fs/promises'
import { cleanFlag, matchPhotos, rankOf, twinSets, normalize } from './photos.mjs'

const root = '.data/pangram', arg = (name, fallback) => { const at = process.argv.indexOf(name); return at < 0 ? fallback : Number(process.argv[at + 1]) }
const limit = arg('--limit', 100), PAD = 0.25
const inventory = JSON.parse(await readFile('bench/open-fonts.json', 'utf8'))
const designs = new Map()
for (const f of inventory.families) if (f.id.startsWith('pangram-pangram') && f.design) designs.set(f.design, [...(designs.get(f.design) ?? []), f])
const designOf = handle => [...designs.keys()].filter(d => handle === d || handle.startsWith(d + '-')).sort((a, b) => b.length - a.length)[0]
const boxes = new Map((await readFile(`${root}/specimens-boxes.jsonl`, 'utf8').catch(() => '')).split('\n').filter(Boolean).map(JSON.parse).map(b => [b.image, b]))
const pictures = JSON.parse(await readFile(`${root}/specimens.json`, 'utf8')).images.filter(p => designOf(p.handle))
const padded = ([x0, y0, x1, y1], width, height) => { const pad = Math.round((y1 - y0) * PAD); return [Math.max(0, x0 - pad), Math.max(0, y0 - pad), Math.min(width, x1 + pad), Math.min(height, y1 + pad)] }
// The picture's own line finder (scripts/clean-candidates.mjs bands, written by a band script to specimens-bands.jsonl): the tallest
// band's box in the picture, read raw, and the cleaned band crop in .data/pangram/bands, read as a picture of its own.
const found = new Map((await readFile(`${root}/specimens-bands.jsonl`, 'utf8').catch(() => '')).split('\n').filter(Boolean).map(JSON.parse).map(b => [b.image, b]))
const items = pictures.flatMap(p => {
  const b = boxes.get(p.image), path = `${root}/specimens/${p.image}`, whole = { picture: p, kind: 'whole', path, box: [0, 0, 1e9, 1e9] }
  const top = b?.lines?.find(l => l.box[2] - l.box[0] >= 16 && l.box[3] - l.box[1] >= 8)  // the tallest line Vision found with room for letters in it
  const line = top ? { picture: p, kind: 'line', path, box: padded(top.box, b.width, b.height), text: top.text } : null
  const own = (found.get(p.image)?.boxes ?? []).map((box, k) => ({ picture: p, kind: k ? 'bands' : 'band', path, box }))  // the tallest band alone, and every band, merged below
  return [whole, ...(line ? [line] : []), ...own]
})
console.log(`${pictures.length} pictures of ${new Set(pictures.map(p => designOf(p.handle))).size} designs, ${items.length} reads`)
const { clean } = cleanFlag('pangram-specimens'), { results, adapter, encoderSha256 } = await matchPhotos(items, { limit, log: () => {}, clean })
const twins = await twinSets()
const scored = items.map((item, i) => {
  const families = designs.get(designOf(item.picture.handle)), rows = results[i].rows ?? []
  const ranks = families.map(f => rankOf(rows, f.family, twins.get(f.id))).filter(r => r !== null)
  return { image: item.picture.image, handle: item.picture.handle, design: designOf(item.picture.handle), kind: item.kind, status: results[i].status, rank: ranks.length ? Math.min(...ranks) : null, first: rows.slice(0, 3).map(r => r.name), ...(item.text && { text: item.text }) }
})
// Every band of a picture as one read: each family at its best score over the bands, ranked again.
for (const image of new Set(scored.filter(s => s.kind === 'band').map(s => s.image))) {
  const reads = items.map((item, i) => [item, i]).filter(([item]) => item.picture.image === image && (item.kind === 'band' || item.kind === 'bands'))
  const best = new Map()
  for (const [, i] of reads) for (const row of results[i].rows ?? []) if (!best.has(row.name) || best.get(row.name).score < row.score) best.set(row.name, row)
  const rows = [...best.values()].sort((a, b) => b.score - a.score), first = scored.find(s => s.image === image && s.kind === 'band')
  const ranks = designs.get(first.design).map(f => rankOf(rows, f.family, twins.get(f.id))).filter(r => r !== null)
  scored.push({ ...first, kind: 'all bands', rank: ranks.length ? Math.min(...ranks) : null, first: rows.slice(0, 3).map(r => r.name) })
}
const share = (list, test) => list.length ? Math.round(1e3 * list.filter(test).length / list.length) / 10 : null
const summary = kind => { const list = scored.filter(s => s.kind === kind); return { reads: list.length, top1: share(list, s => s.rank === 1), top5: share(list, s => s.rank && s.rank <= 5), top10: share(list, s => s.rank && s.rank <= 10), top100: share(list, s => s.rank !== null) } }
const byDesign = {}
for (const s of scored.filter(s => s.kind === 'line')) (byDesign[s.design] ??= []).push(s)
const report = { set: { pictures: pictures.length, designs: designs.size, source: 'https://pangrampangram.com/products.json, local only' }, rule: 'Right when any family of the product\'s design, a family folded into its row, or a twin, is among the first rows (10); the whole picture and the tallest Vision line with a quarter of its height around.',
  adapter, encoderSha256, date: new Date().toISOString().slice(0, 10), results: Object.fromEntries(['whole', 'line', 'band', 'all bands'].filter(k => scored.some(s => s.kind === k)).map(k => [k, summary(k)])),
  byDesign: Object.fromEntries(Object.entries(byDesign).map(([d, list]) => [d, { pictures: list.length, top10: share(list, s => s.rank && s.rank <= 10) }]).sort((a, b) => (a[1].top10 ?? -1) - (b[1].top10 ?? -1))), reads: scored }
await writeFile(clean ? `${root}/specimens-read-${clean.replace(/\W+/g, '-')}.json` : `${root}/specimens-read.json`, JSON.stringify(report, null, 1) + '\n')
for (const [kind, r] of Object.entries(report.results)) console.log(`${kind}: ${r.reads} reads, top-1 ${r.top1}%, top-5 ${r.top5}%, top-10 ${r.top10}%, top-${limit} ${r.top100}%`)
console.log('worst designs by line:', Object.entries(report.byDesign).slice(0, 12).map(([d, r]) => `${d} ${r.top10 ?? '-'}% of ${r.pictures}`).join('; '))
