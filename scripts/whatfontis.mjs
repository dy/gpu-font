// WhatFontIs-Bench v1.0 (CC BY 4.0): 11,995 photos of one word in a known font on real surfaces, walls and printed
// objects, made by WhatFontIs, which publishes its own API's results on the whole set. gpu-font reads every image whose
// font a shipped catalog holds, cut to the set's own text crop. Fonts that never entered training or checkpoint selection
// (the style split's test role, and every catalog besides Google Fonts) are the final read; the rest is development.
import { readFile, writeFile, access, readdir } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { matchPhotos, rankOf, catalogNames, normalize } from './photos.mjs'

export const REPO = 'https://github.com/whatfontis/WhatFontIs-Bench', COMMIT = 'f01d865380c160b5d54b07d0d1aba54d6ae3fefe'
const ROOT = '.data/whatfontis', SET = `${ROOT}/WhatFontIs-Bench/v1`
const exists = path => access(path).then(() => true, () => false)
const run = (command, ...args) => execFileSync(command, args, { stdio: 'inherit' })

// A font's family without the weight or style its title adds: "Heebo regular", "Teko 300", "Futura PT Book".
export const family = name => name.replace(/(\s+(regular|book|medium|roman|\d{3}))+$/i, '')

// The labels at the pinned commit, the four image archives of release v1.0 unpacked beside them, and every label whose
// font a shipped catalog holds, with that family's role in the style split.
export async function whatfontisSet() {
  if (!await exists(`${ROOT}/WhatFontIs-Bench`)) run('git', 'clone', '--quiet', REPO, `${ROOT}/WhatFontIs-Bench`)
  run('git', '-C', `${ROOT}/WhatFontIs-Bench`, 'checkout', '--quiet', COMMIT)
  if ((await readdir(`${SET}/scenes`).catch(() => [])).length < 11995) for (const n of [1, 2, 3, 4]) {
    const zip = `${ROOT}/WhatFontIs-Bench_v1.0_images_part${n}.zip`
    if (!await exists(zip)) run('curl', '-sSfL', '-o', zip, `${REPO}/releases/download/v1.0/WhatFontIs-Bench_v1.0_images_part${n}.zip`)
    run('unzip', '-oq', zip, '-d', SET)
  }
  const names = await catalogNames(), roles = JSON.parse(await readFile('bench/encoder-split.json')).families
  const labels = (await readFile(`${SET}/labels.jsonl`, 'utf8')).trim().split('\n').map(line => JSON.parse(line))
  const indexed = labels.flatMap(label => {
    const truth = names.get(normalize(family(label.family)))
    if (!truth) return []
    const role = truth.catalog === 'google-fonts' ? roles[truth.familyId] : 'other catalogs'
    return [{ label, truth, role, final: role === 'test' || role === 'other catalogs', path: `${SET}/${label.image}`, box: label.crop_box }]
  })
  return { labels, indexed, fonts: JSON.parse(await readFile(`${SET}/fonts.json`)), path: id => `${SET}/scenes/${String(id).padStart(5, '0')}.jpg` }
}

async function main() {
  const { labels, indexed: images, fonts } = await whatfontisSet()
  const { browser, adapter, encoderSha256, results } = await matchPhotos(images)
  const scored = images.map((image, i) => ({ ...image, ...results[i], rank: rankOf(results[i].rows, image.truth.family) }))
  const share = (list, test) => Math.round(1e4 * list.filter(test).length / list.length) / 1e4
  const summary = list => ({ images: list.length, fonts: new Set(list.map(s => s.truth.family)).size,
    top1: share(list, s => s.rank === 1), top5: share(list, s => s.rank && s.rank <= 5), top20: share(list, s => s.rank !== null) })
  const slices = key => Object.fromEntries([...new Set(scored.map(key))].sort().map(value => [value, summary(scored.filter(s => key(s) === value))]))
  const times = scored.map(s => s.ms).sort((a, b) => a - b), at = p => Math.round(times[Math.floor(p * (times.length - 1))])
  const report = {
    set: { name: 'WhatFontIs-Bench v1.0', repository: REPO, commit: COMMIT, license: 'CC BY 4.0', paper: 'https://doi.org/10.5281/zenodo.22876579',
      images: labels.length, fonts: fonts.length, indexed: { images: images.length, fonts: new Set(images.map(i => i.truth.family)).size } },
    rule: 'Right when the true family, or a family the page folds into that row, is among the first rows the page shows (5); any weight counts, as in the set\'s own results.',
    crop: 'The set\'s crop_box, the same for every finder.',
    catalogs: 'All: every shipped catalog searched as one.', browser, adapter, encoderSha256, date: new Date().toISOString().slice(0, 10),
    timeMs: { median: at(0.5), p95: at(0.95), from: 'image bytes to ranked rows, warm' },
    // Crops preparation turned away before the model: no answer at all.
    unread: Object.fromEntries([...new Set(scored.map(s => s.status))].filter(s => s !== 'ok').map(s => [s, scored.filter(x => x.status === s).length])),
    results: { all: summary(scored), final: summary(scored.filter(s => s.final)), development: summary(scored.filter(s => !s.final)),
      role: slices(s => s.role), type: slices(s => s.label.type), difficulty: slices(s => s.label.difficulty), category: slices(s => s.label.category),
      capitals: slices(s => s.label.caps_only ? 'capitals only' : 'mixed case'), effect: slices(s => s.label.hard_effects?.length ? s.label.hard_effects.join(' + ') : 'none') },
    // Published in the set's README, September 2026: all 11,995 images, searched among over 1.2 million fonts.
    whatfontis: { source: `${REPO}/blob/${COMMIT}/README.md`, searched: 'over 1.2 million fonts', images: 11995, top1: 0.837, top5: 0.933, top20: 0.965 }
  }
  // One image a line, so the report stays readable and diffs stay small.
  const rows = scored.map(s => JSON.stringify({ id: s.label.id, family: s.truth.family, final: s.final, ...(s.status !== 'ok' && { status: s.status }), rank: s.rank, first: s.rows[0]?.name ?? null,
    score: s.rows[0] ? Math.round(s.rows[0].score * 1e3) / 1e3 : null, ms: Math.round(s.ms) }))
  await writeFile('bench/whatfontis.json', JSON.stringify(report, null, 1).slice(0, -2) + `,\n "images": [\n  ${rows.join(',\n  ')}\n ]\n}\n`)
  const pct = v => `${(100 * v).toFixed(1)}%`
  for (const [name, r] of Object.entries({ all: report.results.all, final: report.results.final, development: report.results.development }))
    console.log(`${name}: ${r.images} images, ${r.fonts} fonts: top-1 ${pct(r.top1)}, top-5 ${pct(r.top5)}, top-20 ${pct(r.top20)}`)
  console.log(`median ${report.timeMs.median} ms, p95 ${report.timeMs.p95} ms on ${adapter}`)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
