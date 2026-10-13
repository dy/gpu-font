// One JSON batch per process: RGB pictures read as the page reads a crop (src/clean.mjs readings: each line as it stands and
// cleaned, a steep one turned), each prepared by the browser's own normalization and deskew. The answer is one entry per
// reading, { kind, line, windows }, for train.style_photos. scripts/corpus-prepare.mjs, the frozen benchmark's grayscale
// preparer, stays as pinned.
import { readFileSync } from 'node:fs'
import { prepareLine } from '../src/line.mjs'
import { base64 } from '../src/catalog.mjs'
import { readings } from '../src/clean.mjs'

export function prepareReadings(images) {
  if (!Array.isArray(images) || images.length > 4096) throw new Error('Invalid image batch')
  return images.map(image => {
    const { width, height, pixels } = image
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 16777216 || typeof pixels !== 'string' || !base64(pixels)) throw new Error('Invalid RGB image')
    const rgb = Buffer.from(pixels, 'base64')
    if (rgb.length !== width * height * 3) throw new Error('Truncated or trailing RGB pixels')
    const data = new Uint8Array(width * height * 4)
    for (let i = 0; i < width * height; i++) { data.set(rgb.subarray(3 * i, 3 * i + 3), i * 4); data[i * 4 + 3] = 255 }
    const windows = crop => prepareLine(crop, { deskew: true, sampler: 'windows' }).windows.map(w => ({ width: w.width, height: w.height, pixels: Buffer.from(Uint8Array.from(w.pixels, p => Math.round(p * 255))).toString('base64') }))
    const lines = new Map()
    return readings({ width, height, data }).map(r => { if (!lines.has(r.rect)) lines.set(r.rect, lines.size); return { kind: r.kind, line: lines.get(r.rect), windows: windows(r.image) } })
  })
}

if (process.argv[1]?.endsWith('/photo-prepare.mjs')) process.stdout.write(JSON.stringify(prepareReadings(JSON.parse(readFileSync(0, 'utf8')))))
