// What costs accuracy on photos: the model or the picture? 200 WhatFontIs-Bench development photos are read three ways,
// as they are, binarized (Otsu), and as a clean Chromium render of the same word in the pinned instance of the same face.
// Binarization is also read on the DaFont crops and on the clean renders, to see what it costs where photos are not the
// problem. Writes bench/photo-controls.json; renders and binarized crops stay in .data/photo-controls/.
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
const bytes = async path => (await readFile(path)).toString('base64')
// Otsu's threshold on luminance; the minority side is ink, drawn black on white.
const binarize = async (path, box, out) => page.evaluate(async ({ data, type, box }) => {
  const bitmap = await createImageBitmap(await (await fetch(`data:${type};base64,${data}`)).blob())
  const [x0, y0, x1, y1] = [Math.max(0, box[0]), Math.max(0, box[1]), Math.min(bitmap.width, box[2]), Math.min(bitmap.height, box[3])]
  const canvas = new OffscreenCanvas(x1 - x0, y1 - y0), ctx = canvas.getContext('2d'); ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0)
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height), n = canvas.width * canvas.height, gray = new Float32Array(n), counts = new Array(256).fill(0)
  for (let i = 0; i < n; i++) counts[Math.round(gray[i] = image.data[4 * i] * .2126 + image.data[4 * i + 1] * .7152 + image.data[4 * i + 2] * .0722)]++
  let best = -1, threshold = 128, below = 0, sumBelow = 0; const sum = counts.reduce((s, c, v) => s + c * v, 0)
  for (let v = 0; v < 256; v++) {
    below += counts[v]; sumBelow += v * counts[v]
    if (!below || below === n) continue
    const between = below * (n - below) * (sumBelow / below - (sum - sumBelow) / (n - below)) ** 2
    if (between > best) { best = between; threshold = v }
  }
  const darkInk = gray.filter(v => v <= threshold).length < n / 2
  for (let i = 0; i < n; i++) { const v = (darkInk ? gray[i] <= threshold : gray[i] > threshold) ? 0 : 255; image.data.set([v, v, v, 255], 4 * i) }
  ctx.putImageData(image, 0, 0); const png = new Uint8Array(await (await canvas.convertToBlob()).arrayBuffer()); let raw = ''
  for (const b of png) raw += String.fromCharCode(b)
  return btoa(raw)
}, { data: await bytes(path), type: path.endsWith('.png') ? 'image/png' : 'image/jpeg', box }).then(png => writeFile(out, Buffer.from(png, 'base64')))
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
    await binarize(p.path, p.box, `${OUT}/${id}-binary.png`); await binarize(clean, WHOLE, `${OUT}/${id}-clean-binary.png`)
    cases.push({ set: 'photos', truth: p.truth.family, variants: { photo: { path: p.path, box: p.box }, binarized: { path: `${OUT}/${id}-binary.png`, box: WHOLE },
      clean: { path: clean, box: WHOLE }, 'clean binarized': { path: `${OUT}/${id}-clean-binary.png`, box: WHOLE } } })
  }
  for (const [k, f] of forum.entries()) {
    await binarize(f.path, f.box, `${OUT}/forum-${k}.png`)
    cases.push({ set: 'forum', truth: f.truth, variants: { photo: { path: f.path, box: f.box }, binarized: { path: `${OUT}/forum-${k}.png`, box: WHOLE } } })
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
  variants: { clean: 'the word at 64 px in the pinned instance of the photo\'s face, black on white', binarized: 'Otsu\'s threshold on luminance, the minority side drawn black on white' } }
await writeFile('bench/photo-controls.json', JSON.stringify(report, null, 1) + '\n')
for (const [set, r] of Object.entries(report)) if (r.cases) console.log(set, r.cases, Object.entries(r).filter(([, v]) => v?.top5 !== undefined).map(([k, v]) => `${k} ${(100 * v.top5).toFixed(1)}%`).join(', '))
