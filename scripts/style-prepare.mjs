// Persistent browser preparation for on-the-fly training: the exact prepareLine, binary frames over stdio.
// In: u32 width, u32 height, width*height grayscale bytes. Out: u32 count, then per window u32 width, u32 height, bytes.
import { prepareLine } from '../src/line.mjs'

const u32 = n => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b }
let pending = Buffer.alloc(0)
process.stdin.on('data', chunk => {
  pending = pending.length ? Buffer.concat([pending, chunk]) : chunk
  const out = []
  while (pending.length >= 8) {
    const width = pending.readUInt32LE(0), height = pending.readUInt32LE(4), size = width * height
    if (!width || !height || size > 16_777_216) throw new Error('Invalid frame')
    if (pending.length < 8 + size) break
    const gray = pending.subarray(8, 8 + size), data = new Uint8Array(size * 4)
    for (let i = 0; i < size; i++) { data.fill(gray[i], i * 4, i * 4 + 3); data[i * 4 + 3] = 255 }
    pending = pending.subarray(8 + size)
    // A frame the preparation rejects yields no windows; the stream stays alive for the next frame.
    let windows = []
    try { windows = prepareLine({ width, height, data }, { deskew: true, sampler: 'windows' }).windows }
    catch (error) { process.stderr.write(`style-prepare: ${width}x${height} rejected: ${error.message}\n`) }
    out.push(u32(windows.length))
    for (const w of windows) out.push(u32(w.width), u32(w.height), Buffer.from(Uint8Array.from(w.pixels, p => Math.round(p * 255))))
  }
  if (out.length) process.stdout.write(Buffer.concat(out))
})
