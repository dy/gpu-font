// What costs accuracy on photos: the model or the picture? 200 WhatFontIs-Bench development photos are read as they are,
// through each candidate cleaner (scripts/clean-candidates.mjs: Otsu on luminance; the background flattened first; the
// picture's own ink colour; colour, flattening and a despeckle together), and as a clean Chromium render of the same word
// in the pinned instance of the same face. The cleaners are also read on the DaFont crops and the plain binarization on the
// clean renders, to see what it costs where photos are not the problem. Writes bench/photo-controls.json; renders and
// cleaned crops stay in .data/photo-controls/.
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { chromium } from 'playwright'
import { whatfontisSet } from './whatfontis.mjs'
import { matchPhotos, rankOf } from './photos.mjs'

const OUT = '.data/photo-controls', WHOLE = [0, 0, 1e9, 1e9], SAMPLE = 200
await mkdir(OUT, { recursive: true })
const { indexed } = await whatfontisSet()
// Every fifteenth development photo of a Google family, whose exact face is pinned locally.
const weight = title => Number(title.match(/(\d{3})$/)?.[1] ?? 400)
const photos = indexed.filter(i => !i.final && i.truth.catalog === 'google-fonts').filter((_, k) => k % 15 === 0).slice(0, SAMPLE)
const { crops } = JSON.parse(await readFile('bench/dafont-crops.json')), tasks = new Map(JSON.parse(await readFile('.data/dafont/tasks.json')).map(t => [t.task_id, t]))
const forum = Object.entries(crops).map(([id, box]) => { const t = tasks.get(id); return { truth: t.identified_font, path: `.data/dafont/images/${id}${new URL(t.img_url).pathname.match(/\.\w+$/)[0]}`, box } })

const browser = await chromium.launch(), page = await browser.newPage()
await page.goto(`http://127.0.0.1:${process.env.PORT || 4179}/`)  // the cleaners are served by the demo
const bytes = async path => (await readFile(path)).toString('base64')
// A crop through one of the candidate cleaners, written as a PNG.
const cleaned = async (path, box, cleaner, out) => page.evaluate(async ({ data, type, box, cleaner }) => {
  const { CLEANERS } = await import('/scripts/clean-candidates.mjs')
  const bitmap = await createImageBitmap(await (await fetch(`data:${type};base64,${data}`)).blob())
  const [x0, y0, x1, y1] = [Math.max(0, box[0]), Math.max(0, box[1]), Math.min(bitmap.width, box[2]), Math.min(bitmap.height, box[3])]
  const canvas = new OffscreenCanvas(x1 - x0, y1 - y0), ctx = canvas.getContext('2d'); ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0)
  const out = CLEANERS[cleaner](ctx.getImageData(0, 0, canvas.width, canvas.height)); ctx.putImageData(new ImageData(out.data, out.width, out.height), 0, 0); const png = new Uint8Array(await (await canvas.convertToBlob()).arrayBuffer()); let raw = ''
  for (const b of png) raw += String.fromCharCode(b)
  return btoa(raw)
}, { data: await bytes(path), type: path.endsWith('.png') ? 'image/png' : 'image/jpeg', box, cleaner }).then(png => writeFile(out, Buffer.from(png, 'base64')))
const CLEANERS = ['binarized', 'flattened', 'colour', 'cleaned', 'colour flattened', 'colour soft', 'colour flattened soft']
const render = (font, text, out) => page.evaluate(async ({ font, text }) => {
  const face = new FontFace('control', Uint8Array.from(atob(font), c => c.charCodeAt(0))); await face.load(); document.fonts.add(face)
  const probe = new OffscreenCanvas(1, 1).getContext('2d'); probe.font = '64px control'; const m = probe.measureText(text), pad = 20
  const canvas = new OffscreenCanvas(Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + 2 * pad, Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + 2 * pad)
  const ctx = canvas.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = '#000'; ctx.font = '64px control'
  ctx.fillText(text, pad + m.actualBoundingBoxLeft, pad + m.actualBoundingBoxAscent); document.fonts.delete(face)
  const png = new Uint8Array(await (await canvas.convertToBlob()).arrayBuffer()); let raw = ''
  for (const b of png) raw += String.fromCharCode(b)
  return btoa(raw)
}, { font, text }).then(png => writeFile(out, Buffer.from(png, 'base64')))

// Each case: its truth and the variants to read, in order.
const cases = []
try {
  for (const p of photos) {
    const id = p.label.id, clean = `${OUT}/${id}-clean.png`
    await render(await bytes(`.data/style/instances/${p.truth.familyId}--${weight(p.label.title)}.ttf`), p.label.text, clean)
    const variants = { photo: { path: p.path, box: p.box } }
    for (const c of CLEANERS) { await cleaned(p.path, p.box, c, `${OUT}/${id}-${c.replace(/ /g, '-')}.png`); variants[c] = { path: `${OUT}/${id}-${c.replace(/ /g, '-')}.png`, box: WHOLE } }
    await cleaned(clean, WHOLE, 'binarized', `${OUT}/${id}-clean-binary.png`); await cleaned(clean, WHOLE, 'colour soft', `${OUT}/${id}-clean-colour-soft.png`)
    cases.push({ set: 'photos', truth: p.truth.family, variants: { ...variants, clean: { path: clean, box: WHOLE }, 'clean binarized': { path: `${OUT}/${id}-clean-binary.png`, box: WHOLE }, 'clean colour soft': { path: `${OUT}/${id}-clean-colour-soft.png`, box: WHOLE } } })
  }
  for (const [k, f] of forum.entries()) {
    const variants = { photo: { path: f.path, box: f.box } }
    for (const c of CLEANERS) { await cleaned(f.path, f.box, c, `${OUT}/forum-${k}-${c.replace(/ /g, '-')}.png`); variants[c] = { path: `${OUT}/forum-${k}-${c.replace(/ /g, '-')}.png`, box: WHOLE } }
    cases.push({ set: 'forum', truth: f.truth, variants })
  }
} finally { await browser.close() }

const reads = cases.flatMap(c => Object.values(c.variants)), { results } = await matchPhotos(reads, { log: () => {} })
let next = 0
for (const c of cases) for (const name of Object.keys(c.variants)) c.variants[name] = rankOf(results[next++].rows, c.truth)
const share = (list, test) => Math.round(1e4 * list.filter(test).length / list.length) / 1e4
const table = set => { const list = cases.filter(c => c.set === set)
  return { cases: list.length, ...Object.fromEntries(Object.keys(list[0].variants).map(v => [v, { top1: share(list, c => c.variants[v] === 1), top5: share(list, c => c.variants[v] && c.variants[v] <= 5) }])) } }
const report = { photos: { sample: 'WhatFontIs-Bench development photos of Google families, every fifteenth', ...table('photos') },
  forum: { sample: 'the cropped DaFont requests', ...table('forum') },
  variants: { photo: 'the crop as the page reads it: its lines, cleaned unless two-tone (src/clean.mjs); every variant below is read the same way', clean: 'the word at 64 px in the pinned instance of the photo\'s face, black on white', binarized: 'Otsu\'s threshold on luminance as ink', flattened: 'the background under the letters taken away first (an opening of the ink), then Otsu',
    colour: 'ink as the position between the background\'s and the ink\'s colours (two means in RGB), then Otsu', cleaned: 'the colour axis, flattened, Otsu and a 3×3 majority despeckle',
    'colour flattened': 'the colour axis, flattened, Otsu', 'colour soft': 'the colour axis, the ink within 0.15 of Otsu\'s threshold keeping its grey', 'colour flattened soft': 'the colour axis flattened, with the same soft edge' } }
await writeFile('bench/photo-controls.json', JSON.stringify(report, null, 1) + '\n')
for (const [set, r] of Object.entries(report)) if (r.cases) console.log(set, r.cases, Object.entries(r).filter(([, v]) => v?.top5 !== undefined).map(([k, v]) => `${k} ${(100 * v.top5).toFixed(1)}%`).join(', '))
