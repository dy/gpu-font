// The shared image set of https://github.com/dy/gpu-font/issues/1: forty crops every font finder reads, with licence, crop
// and true family, plus the weight and italic probe pairs. Photos are WhatFontIs-Bench v1.0 images (CC BY 4.0, CC0
// backgrounds); renders are drawn here from the pinned Google Fonts instances (SIL OFL) of families that never entered
// training or checkpoint selection. Each crop is written to .data/finders/<id>.png, the one file every finder gets.
// bench/finders.json keeps each finder's answers; this script fills gpu-font's, and keeps what was entered for the others.
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { whatfontisSet, REPO, COMMIT } from './whatfontis.mjs'
import { matchPhotos, rankOf } from './photos.mjs'

const OUT = '.data/finders', REPORT = 'bench/finders.json', INSTANCES = '.data/style/instances'
const PHOTO = { license: 'CC BY 4.0 (WhatFontIs-Bench v1.0; backgrounds CC0 1.0)', source: `${REPO}/tree/${COMMIT}/v1` }
const order = name => createHash('sha256').update(name).digest('hex')  // a fixed order that does not follow the alphabet

// Commercial classics the free finders can only answer with a look-alike, as the set names them.
const CLASSICS = ['Neue Haas Grotesk Text Pro 65 Medium', 'Futura PT Book', 'Adobe Garamond Pro Regular', 'DIN 2014 Regular', 'FranklinGothic URW Book',
  'Century Gothic Pro Regular', 'Rockwell Std Regular', 'Trajan Pro 3 Regular', 'Adobe Caslon Pro Regular', 'Brandon Grotesque Regular']
// Renders: [id, group, instance, family, text, css]. The CSS draws one property the photos lack.
const RENDERS = [
  ['hard-perspective', 'hard', 'elmssans--500', 'Elms Sans', 'Waterfront', 'font-size:72px;transform:perspective(420px) rotateY(32deg)'],
  ['hard-outline', 'hard', 'hubotsans--800', 'Hubot Sans', 'Outline Only', 'font-size:80px;color:transparent;-webkit-text-stroke:2.5px #151515'],
  ['hard-small', 'hard', 'atkinsonhyperlegiblenext--400', 'Atkinson Hyperlegible Next', 'Small print on a page', 'font-size:11px'],
  ['hard-condensed', 'hard', 'pragatinarrow--700', 'Pragati Narrow', 'NORTHBOUND', 'font-size:64px'],
  ['hard-script', 'hard', 'euphoriascript--400', 'Euphoria Script', 'Evening Garden', 'font-size:72px'],
  ['script-cyrillic', 'script', 'comfortaa--400', 'Comfortaa', 'Съешь же ещё', 'font-size:56px'],
  ['script-greek', 'script', 'caudex--400', 'Caudex', 'Ξεσκεπάζω την', 'font-size:56px'],
  ['script-arabic', 'script', 'cairo--400', 'Cairo', 'نص حكيم له سر قاطع', 'font-size:56px'],
  ['script-hebrew', 'script', 'frankruhllibre--400', 'Frank Ruhl Libre', 'דג סקרן שט בים', 'font-size:56px'],
  ['script-devanagari', 'script', 'hind--400', 'Hind', 'ऋषियों को सताने', 'font-size:56px'],
  ['script-chinese', 'script', 'chironheihk--400', 'Chiron Hei HK', '天地玄黃宇宙洪荒', 'font-size:56px'],
  ['probe-weight-300', 'probe', 'figtree--300', 'Figtree', 'Harbour lights', 'font-size:64px'],
  ['probe-weight-700', 'probe', 'figtree--700', 'Figtree', 'Harbour lights', 'font-size:64px'],
  ['probe-upright', 'probe', 'faustina--400', 'Faustina', 'Harbour lights', 'font-size:64px'],
  ['probe-italic', 'probe', 'faustina--400i', 'Faustina', 'Harbour lights', 'font-size:64px']
]
const PROBES = { weight: ['probe-weight-300', 'probe-weight-700'], italic: ['probe-upright', 'probe-italic'],
  subsets: RENDERS.filter(r => r[1] === 'script').map(r => r[0]) }
// Every finder the page compares, in its order, and how each may be run. tests/photos.test.mjs keeps the two lists equal.
const FINDERS = {
  WhatFontIs: 'Its API (https://www.whatfontis.com/API-identify-fonts-from-image.html): 200 requests a day, personal use with attribution.',
  WhatTheFont: 'By hand: MyFonts terms forbid scripted queries.',
  'Font Finder': 'By hand: it publishes no terms and no API.',
  'Photoshop Match Font': 'By hand: a desktop app.',
  Lens: 'From a script.',
  'What the Google Font': 'By hand: its terms forbid automated tools without authorization, and robots.txt disallows /api/.',
  Matcherator: 'By hand: Fontspring terms forbid scripted queries.',
  'gpu-font': 'From scripts/finders.mjs.'
}

// Photos, each family once: a cycle of surface and difficulty so every kind appears.
const KINDS = [['texture', 'easy'], ['scene', 'medium'], ['object', 'hard'], ['texture', 'hard'], ['scene', 'easy'], ['object', 'medium'], ['texture', 'medium'], ['scene', 'hard'], ['object', 'easy'], ['texture', 'hard']]
function photos(labels, finalFamilies) {
  const byTitle = Map.groupBy(labels, l => l.title), used = new Set(), pick = (list, i) => list.find(l => l.type === KINDS[i % 10][0] && l.difficulty === KINDS[i % 10][1]) || list[0]
  const google = [...finalFamilies].sort((a, b) => order(a) < order(b) ? -1 : 1), take = (category, n) => google.filter(t => byTitle.get(t)[0].category === category && !used.has(t)).slice(0, n)
  const control = [...take('Sans-Serif', 3), ...take('Serif', 3), ...take('Slab Serif', 2), ...take('Monospaced', 2)]
  control.forEach(t => used.add(t))
  // Hard photos: each effect the set applies, its faintest print, and a word in capitals.
  const rest = google.filter(t => !used.has(t)), hardest = key => (a, b) => a[key] - b[key], hard = [
    ['hard-shadow', l => l.hard_effects?.join() === 'cast shadow'], ['hard-glare', l => l.hard_effects?.join() === 'glare'],
    ['hard-wear', l => l.hard_effects?.includes('print wear')], ['hard-contrast', l => l.difficulty === 'hard', hardest('contrast')],
    ['hard-caps', l => l.difficulty === 'hard' && /^[A-Z]+$/.test(l.text), hardest('cap_height_px')]
  ].map(([id, test, sort]) => {
    const candidates = rest.filter(t => !used.has(t)).flatMap(t => byTitle.get(t).filter(test))
    const label = sort ? candidates.sort(sort)[0] : candidates[0]; used.add(label.title); return [id, label]
  })
  // One to three letters cut from a photo by the set's own letter boxes (which span each letter's advance), with a
  // fifth of the capital height above and below.
  const short = [1, 2, 3, 3].map((n, i) => {
    const title = rest.find(t => !used.has(t) && byTitle.get(t).some(l => l.char_quads)); used.add(title)
    const label = byTitle.get(title).find(l => l.char_quads), points = label.char_quads.slice(0, n).flat(), pad = Math.round(label.cap_height_px / 5)
    const xs = points.map(p => p[0]), ys = points.map(p => p[1])
    return [`short-${i + 1}`, label, [Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)) - pad, Math.ceil(Math.max(...xs)), Math.ceil(Math.max(...ys)) + pad], label.text.slice(0, n)]
  })
  return [...control.map((t, i) => [`control-${i + 1}`, 'control', pick(byTitle.get(t), i)]), ...CLASSICS.map((t, i) => [`classic-${i + 1}`, 'classic', pick(byTitle.get(t), i)]),
    ...hard.map(([id, label]) => [id, 'hard', label]), ...short.map(([id, label, box, text]) => [id, 'short', label, box, text])]
}

const { labels, indexed, path } = await whatfontisSet()
const finals = new Set(indexed.filter(i => i.final && i.truth.catalog === 'google-fonts').map(i => i.label.title))
const images = [
  ...photos(labels, finals).map(([id, group, label, box = label.crop_box, text = label.text]) => ({ id, group, family: label.family.replace(/(\s+(regular|book|medium|roman|\d{3}))+$/i, ''),
    face: label.title, text, source: `WhatFontIs-Bench v1.0, image ${label.id} (${label.type}, ${label.difficulty})`, link: `${PHOTO.source}`, license: PHOTO.license, box, from: path(label.id) })),
  ...RENDERS.map(([id, group, instance, family, text, css]) => ({ id, group, family, face: instance, text, css, source: 'Rendered by scripts/finders.mjs in Chromium',
    link: `https://fonts.google.com/specimen/${family.replaceAll(' ', '+')}`, license: 'SIL Open Font License (the font)' }))
]

// Every crop as the PNG each finder receives: photos cut to their box, renders drawn black on white with a margin.
await mkdir(OUT, { recursive: true })
const browser = await chromium.launch()
try {
  const page = await browser.newPage({ deviceScaleFactor: 1 })
  for (const image of images) {
    let png
    if (image.from) {
      const [x0, y0, x1, y1] = image.box
      png = Buffer.from(await page.evaluate(async ({ base64, x0, y0, x1, y1 }) => {
        const bitmap = await createImageBitmap(await (await fetch(`data:image/jpeg;base64,${base64}`)).blob())
        x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(bitmap.width, x1); y1 = Math.min(bitmap.height, y1)
        const canvas = new OffscreenCanvas(x1 - x0, y1 - y0); canvas.getContext('2d').drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0)
        const bytes = new Uint8Array(await (await canvas.convertToBlob({ type: 'image/png' })).arrayBuffer()); let raw = ''
        for (const b of bytes) raw += String.fromCharCode(b)
        return btoa(raw)
      }, { base64: (await readFile(image.from)).toString('base64'), x0, y0, x1, y1 }), 'base64')
    } else {
      const font = (await readFile(`${INSTANCES}/${image.face}.ttf`)).toString('base64')
      await page.setContent(`<style>@font-face{font-family:f;src:url(data:font/ttf;base64,${font})}body{margin:0;background:#fff}
        div{display:inline-block;padding:120px;color:#151515;font-family:f;line-height:1.3;white-space:nowrap}span{display:inline-block;${image.css}}</style>
        <div><span dir="auto">${image.text}</span></div>`)
      await page.evaluate(() => document.fonts.ready)
      // The text's box after any transform, with a margin of a third of the font size.
      const box = await page.evaluate(() => { const r = document.querySelector('span').getBoundingClientRect(), m = parseFloat(getComputedStyle(document.querySelector('span')).fontSize) / 3
        return { x: Math.floor(r.left - m), y: Math.floor(r.top - m), width: Math.ceil(r.width + 2 * m), height: Math.ceil(r.height + 2 * m) } })
      png = await page.screenshot({ clip: box })
    }
    await writeFile(`${OUT}/${image.id}.png`, png)
    image.file = `${OUT}/${image.id}.png`; image.sha256 = createHash('sha256').update(png).digest('hex')
  }
} finally { await browser.close() }

// gpu-font reads each PNG whole; the other finders keep whatever bench/finders.json already holds for them.
const { adapter, results } = await matchPhotos(images.map(image => ({ path: image.file, box: [0, 0, 1e9, 1e9] })), { log: () => {} })
const previous = JSON.parse(await readFile(REPORT, 'utf8').catch(() => '{}'))
const answers = Object.fromEntries(images.map((image, i) => {
  const rows = results[i].rows.slice(0, 5), first = rows[0]
  return [image.id, { top5: rows.map(r => r.name), rank: rankOf(rows, image.family), face: first ? `${first.weight} ${first.style}` : null, ms: Math.round(results[i].ms) }]
}))
const report = {
  issue: 'https://github.com/dy/gpu-font/issues/1',
  protocol: 'One fixed crop per image, the same file for every finder. Top 1 and top 5 at family level; identical designs count as right. Time per image from upload to result.',
  probes: PROBES,
  images: images.map(({ from, css, ...image }) => ({ ...image, ...(css && { css }) })),
  finders: Object.fromEntries(Object.entries(FINDERS).map(([name, how]) => [name, { ...previous.finders?.[name], how,
    ...(name === 'gpu-font' && { how: `${how} Catalog All, Chromium on ${adapter}.`, date: new Date().toISOString().slice(0, 10), answers }) }]))
}
await writeFile(REPORT, JSON.stringify(report, null, 1) + '\n')
const scored = Object.values(answers).filter((_, i) => images[i].group !== 'classic' && images[i].group !== 'probe')
console.log(`${images.length} crops in ${OUT}; gpu-font top-1 ${scored.filter(a => a.rank === 1).length}/${scored.length}, top-5 ${scored.filter(a => a.rank).length}/${scored.length} (classics and probes judged separately)`)
