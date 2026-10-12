// One JSON batch per process; reuse the browser's exact normalization and deskew. An image is grayscale bytes, or RGB
// with `channels: 3`; with `readings: true` it is read as the page reads a crop (src/clean.mjs readings: each line as it
// stands and cleaned, a steep one turned) and the answer is one entry per reading, { kind, line, windows }.
import { readFileSync } from 'node:fs'
import { prepareLine } from '../src/line.mjs'
import { base64 } from '../src/catalog.mjs'
import { readings } from '../src/clean.mjs'

export function prepareBatch(images) {
  if (!Array.isArray(images) || images.length > 4096) throw new Error('Invalid image batch')
  return images.map(image => {
    const { width, height, pixels, channels = 1 } = image
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 16777216 || typeof pixels !== 'string' || !base64(pixels) || ![1, 3].includes(channels)) throw new Error('Invalid image')
    const bytes = Buffer.from(pixels, 'base64')
    if (bytes.length !== width * height * channels) throw new Error('Truncated or trailing pixels')
    const data = new Uint8Array(width * height * 4)
    for (let i = 0; i < width * height; i++) { if (channels === 1) data.fill(bytes[i], i * 4, i * 4 + 3); else data.set(bytes.subarray(3 * i, 3 * i + 3), i * 4); data[i * 4 + 3] = 255 }
    const windows = crop => prepareLine(crop, { deskew: true, sampler: 'windows' }).windows.map(w => ({ width: w.width, height: w.height, pixels: Buffer.from(Uint8Array.from(w.pixels, p => Math.round(p * 255))).toString('base64') }))
    if (!image.readings) return windows({ width, height, data })
    const lines = new Map()
    return readings({ width, height, data }).map(r => { if (!lines.has(r.rect)) lines.set(r.rect, lines.size); return { kind: r.kind, line: lines.get(r.rect), windows: windows(r.image) } })
  })
}

if (process.argv[1]?.endsWith('/corpus-prepare.mjs')) process.stdout.write(JSON.stringify(prepareBatch(JSON.parse(readFileSync(0, 'utf8')))))
