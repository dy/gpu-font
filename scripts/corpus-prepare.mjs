// One JSON batch per process; reuse the browser's exact normalization and deskew.
import { readFileSync } from 'node:fs'
import { prepareLine } from '../src/line.mjs'
import { base64 } from '../src/catalog.mjs'

export function prepareBatch(images) {
  if (!Array.isArray(images) || images.length > 4096) throw new Error('Invalid image batch')
  return images.map(image => {
    const { width, height, pixels } = image
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 16777216 || typeof pixels !== 'string' || !base64(pixels)) throw new Error('Invalid grayscale image')
    const gray = Buffer.from(pixels, 'base64')
    if (gray.length !== width * height) throw new Error('Truncated or trailing grayscale pixels')
    const data = new Uint8Array(gray.length * 4)
    for (let i = 0; i < gray.length; i++) { data.fill(gray[i], i * 4, i * 4 + 3); data[i * 4 + 3] = 255 }
    const result = prepareLine({ width, height, data }, { deskew: true, sampler: 'windows' })
    return result.windows.map(w => ({ width: w.width, height: w.height, pixels: Buffer.from(Uint8Array.from(w.pixels, p => Math.round(p * 255))).toString('base64') }))
  })
}

if (process.argv[1]?.endsWith('/corpus-prepare.mjs')) process.stdout.write(JSON.stringify(prepareBatch(JSON.parse(readFileSync(0, 'utf8')))))
