import { readFile, writeFile, open, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { prepareLine } from '../src/line.mjs'
const hash = b => createHash('sha256').update(b).digest('hex')
const source = JSON.parse(await readFile('.data/hundred/source.json'))
const sourceHash = createHash('sha256')
for await (const bytes of createReadStream('.data/hundred/source.gray')) sourceHash.update(bytes)
if (sourceHash.digest('hex') !== source.rawSha256) throw new Error('Changed source pixels')
const input = await open('.data/hundred/source.gray')
await mkdir('.data/next', { recursive: true })
const output = await open('.data/next/deskew.u8', 'w'), windows = [], samples = [], digest = createHash('sha256')
let offset = 0
try {
  for (const sample of source.samples) {
    if (!['train', 'unknown-validation'].includes(sample.role)) continue
    const gray = Buffer.alloc(sample.width * sample.height)
    if ((await input.read(gray, 0, gray.length, sample.offset)).bytesRead !== gray.length) throw new Error('Truncated source')
    const data = new Uint8Array(gray.length * 4)
    for (let i = 0; i < gray.length; i++) { data.fill(gray[i], i * 4, i * 4 + 3); data[i * 4 + 3] = 255 }
    const result = prepareLine({ ...sample, data }, { deskew: true, sampler: 'windows' })
    if (result.status !== 'ok') throw new Error('Lost training sample')
    const source = samples.length; samples.push(sample)
    for (const window of result.windows) {
      const bytes = Uint8Array.from(window.pixels, p => Math.round(p * 255))
      await output.writeFile(bytes); digest.update(bytes)
      windows.push({ source, offset, width: window.width, height: window.height }); offset += bytes.length
    }
    if (samples.length % 10000 === 0) console.log(`Deskew training: ${samples.length}`)
  }
} finally { await input.close(); await output.close() }
await writeFile('.data/next/deskew.json', JSON.stringify({ samples, windows, tensorSha256: digest.digest('hex'), sourceSha256: source.rawSha256, lineSha256: hash(await readFile('src/line.mjs')), inputSha256: hash(await readFile('src/input.mjs')), normalizerSha256: hash(await readFile('src/prepare.mjs')) }))
console.log(`Prepared ${samples.length} samples, ${windows.length} windows`)
