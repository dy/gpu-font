// Catalogs from fonts the browser can draw: the Latin lines of the shipped catalogs (train/style_catalog.py), drawn by
// canvas at 48 px with 12 px margins (scripts/style-references-browser.mjs) and prepared as any crop, so their vectors
// rank alongside the shipped ones.
import { prepareLine } from './line.mjs'
import { embedWindows, unit, pack, rowBytes } from './catalog.mjs'

export const LATIN = { lower: ['hamburgefonts', 'quickjivexy', 'dwarflpzm'], upper: ['HAMBURGEFONTS', 'QUICKJIVEXY', 'DWARFLPZM'] }
const ALPHABET = { lower: 'abcdefghijklmnopqrstuvwxyz', upper: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' }, SIZE = 48, PAD = 12
// Every reference line is drawn at each of these sizes and the face's vector averages them all: small text upscaled to
// the model's window looks unlike large text shrunk to it, and references at 48 px alone missed 16 px crops one time
// in four (bench/style.md, Reference sizes). The repository's renderer (scripts/style-references-browser.mjs) uses the same list.
export const SIZES = [16, 24, 48]

// Reference lines per case. A letter the face lacks falls back to another font, so its width changes with the fallback;
// a face missing letters uses its own letters of that case, as the shipped catalogs do, and a case with fewer than 8 is skipped.
export function plan(ctx, family) {
  const width = (letter, fallback) => { ctx.font = `${SIZE}px "${family}", ${fallback}`; return ctx.measureText(letter).width }
  const out = {}
  for (const [kase, letters] of Object.entries(ALPHABET)) {
    const own = [...letters].filter(l => width(l, 'monospace') === width(l, 'serif')).join('')
    if (own.length === letters.length) out[`Latn-${kase}`] = LATIN[kase]
    else if (own.length >= 8) out[`Latn-${kase}`] = own.match(/.{1,12}/g).slice(0, 3)
  }
  return out
}

// One reference line at `size`, black on white, as the reference renders store it: the red channel as gray.
export function drawLine(ctx, family, text, size = SIZE) {
  const canvas = ctx.canvas
  ctx.font = `${size}px "${family}"`
  const b = ctx.measureText(text)
  canvas.width = Math.max(1, Math.ceil(b.actualBoundingBoxLeft + b.actualBoundingBoxRight) + PAD * 2)
  canvas.height = Math.max(1, Math.ceil(b.actualBoundingBoxAscent + b.actualBoundingBoxDescent) + PAD * 2)
  ctx.fillStyle = 'white'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = 'black'; ctx.font = `${size}px "${family}"`
  ctx.fillText(text, PAD + b.actualBoundingBoxLeft, PAD + b.actualBoundingBoxAscent)
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height)
  for (let p = 0; p < image.data.length; p += 4) image.data[p + 1] = image.data[p + 2] = image.data[p]
  return image
}

// Rows as the repository's exporter packs them: L2-normalized, 4 bits a dimension with one scale per row, scales to 6
// significant digits. Four bits rank as well as eight (bench/style.md, Quantization) at half the size.
export function packVectors(rows) {
  const scales = rows.map(r => Math.max(Math.max(...r.map(Math.abs)) / 7, 1e-12)), values = new Int8Array(rows.length * 128)
  rows.forEach((r, i) => r.forEach((v, d) => { values[i * 128 + d] = Math.max(-7, Math.min(7, Math.round(v / scales[i]))) }))
  return { encoding: 'int4-base64', shape: [rows.length, 128], data: toBase64(pack(values, 4)), scales: scales.map(s => Number(s.toPrecision(6))) }
}
function toBase64(bytes) {
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(binary)
}

// Both catalogs' faces and rows in one; a face of the second replaces a face of the first with the same id. Rows are
// copied as packed, so both catalogs must pack them alike.
export function joinCatalogs(first, second) {
  if (first.vectors.encoding !== second.vectors.encoding) throw new Error('Catalogs pack their vectors differently')
  const replaced = new Set(second.faces.map(f => f.id)), faces = [], rows = [], scales = [], owners = [], size = rowBytes(second.vectors)
  for (const [n, catalog] of [first, second].entries()) {
    const bytes = Uint8Array.from(atob(catalog.vectors.data), c => c.charCodeAt(0)), index = new Map()
    catalog.faces.forEach((face, i) => { if (n || !replaced.has(face.id)) { index.set(i, faces.length); faces.push(face) } })
    catalog.vectors.owners.forEach((owner, r) => {
      if (!index.has(owner)) return
      rows.push(bytes.subarray(r * size, (r + 1) * size)); scales.push(catalog.vectors.scales[r]); owners.push(index.get(owner))
    })
  }
  const packed = new Uint8Array(rows.length * size)
  rows.forEach((row, i) => packed.set(row, i * size))
  return { ...second, faces, vectors: { ...second.vectors, shape: [rows.length, 128], data: toBase64(packed), scales, owners } }
}

// A version 3 catalog of `faces`: { id, family, styleName, weight, style, source, ... }, where `source` is what a FontFace
// loads (font bytes, `url(...)`, or `local("PostScript name")` for an installed face). Other fields become the catalog
// face; `familyId` defaults to `local:<family>` and `scripts` to Latin. `project` returns one projection per window;
// `preparation` and `binding` come from the encoder; `source` names the catalog. Faces with no Latin letters, or that fail to load, are returned as skipped.
export async function buildCatalog(faces, { project, preparation, binding, source = 'my-fonts', progress = () => {}, signal }) {
  const ctx = document.createElement('canvas').getContext('2d'), entries = [], rows = [], owners = [], skipped = [], seen = new Set()
  // One vector per case: each line at each size, its windows embedded as a crop's are, all averaged.
  const embedCase = lines => Promise.all(lines.map(windows => project(windows).then(embedWindows))).then(embedded => unit(embedded.reduce((sum, e) => sum.map((v, d) => v + e[d]), new Float32Array(128))))
  const finish = async (face, cases) => {
    const vectors = await Promise.all(cases), owner = entries.length
    if (!vectors.length) { skipped.push(face); return }
    for (const vector of vectors) { rows.push(vector); owners.push(owner) }
    const { source, ...meta } = face
    entries.push({ familyId: `local:${face.family}`, scripts: ['Latn'], ...meta })
  }
  // A face is drawn and prepared while the model still runs the one before it; faces finish in order.
  let previous = Promise.resolve()
  for (const [done, face] of faces.entries()) {
    if (signal?.aborted) break
    progress(done, faces.length)
    if (seen.has(face.id)) continue
    seen.add(face.id)
    const font = new FontFace('gpu-font-reference', face.source)
    try { await font.load() } catch { skipped.push(face); continue }
    document.fonts.add(font)
    let cases
    try {
      cases = Object.values(plan(ctx, font.family))
        .map(lines => SIZES.flatMap(size => lines.map(text => prepareLine(drawLine(ctx, font.family, text, size), { ...preparation, deskew: true, sampler: 'windows' }))).filter(p => p.status === 'ok').map(p => p.windows))
        .filter(lines => lines.length).map(embedCase)
    } finally { document.fonts.delete(font) }
    const current = previous.then(() => finish(face, cases))
    await previous; previous = current
  }
  await previous
  progress(faces.length, faces.length)
  const catalog = entries.length ? { version: 3, kind: 'font-catalog', ...binding, dimensions: 128, source, referenceMethod: `faces-cases-browser-${SIZES.join('-')}`,
    faces: entries, vectors: { ...packVectors(rows.map(r => Array.from(r))), owners } } : null
  return { catalog, skipped }
}
