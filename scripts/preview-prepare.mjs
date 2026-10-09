// Derive encoder inputs from the archive's declared regions, never its font files.
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { validateArchive, decodePng, cropView, sha256 } from './preview-lib.mjs'
import { prepareLine } from '../src/line.mjs'

export function prepareRegion(image, region) {
  const { width, height, channels, pixels } = cropView(image, region)
  const data = new Uint8Array(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    const at = i * channels, rgb = channels < 3 ? [pixels[at], pixels[at], pixels[at]] : pixels.subarray(at, at + 3)
    data.set(rgb, i * 4)
    data[i * 4 + 3] = channels === 2 || channels === 4 ? pixels[at + channels - 1] : 255
  }
  const result = prepareLine({ width, height, data }, { deskew: true, sampler: 'windows' })
  if (result.status !== 'ok') throw new Error(`Unusable reference region: ${result.status}`)
  return result.windows.map(w => ({ width: w.width, height: w.height, rect: w.rect,
    pixels: Buffer.from(Uint8Array.from(w.pixels, p => Math.round(p * 255))) }))
}

/** Every record's windows, in order: `samples` the records that gave any, `windows` owned by their index in it, `chunks` the
 * pixels. A record whose region holds no usable line (a face without digits drawn as an empty line, a capture that failed)
 * gives no reference and is listed in `skipped` with the reason, so the rest of the archive still compiles. */
export async function prepareRecords(records, load) {
  const chunks = [], windows = [], samples = [], skipped = []; let offset = 0
  for (const record of records) {
    const image = await load(record)
    let prepared
    try { prepared = prepareRegion(image, record.region) } catch (error) {
      if (!/^Unusable reference region: /.test(error.message)) throw error
      skipped.push({ id: record.id, status: error.message.slice('Unusable reference region: '.length) }); continue
    }
    const source = samples.push(record) - 1
    for (const window of prepared) {
      windows.push({ source, width: window.width, height: window.height, rect: window.rect, offset })
      chunks.push(window.pixels); offset += window.pixels.length
    }
  }
  return { samples, windows, chunks, skipped }
}

async function main(archive, encoderPath, output) {
  if (!archive || !encoderPath || !output) throw new Error('Expected archive, encoder JSON and output prefix')
  const relative = path.relative(path.resolve(archive), path.resolve(output))
  if (!relative || (relative !== '..' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative))) throw new Error('Outputs must stay outside the source archive')
  const encoderBytes = await readFile(encoderPath), encoder = JSON.parse(encoderBytes)
  if (encoder.kind !== 'font-encoder' || encoder.preparation?.method !== 'deskew-windows') throw new Error('Expected the shared deskew encoder')
  for (const [file, key] of [['src/input.mjs', 'sha256'], ['src/prepare.mjs', 'normalizerSha256'], ['src/line.mjs', 'lineSha256']]) {
    if (sha256(await readFile(file)) !== encoder.preparation[key]) throw new Error(`Incompatible preparation: ${file}`)
  }
  const audit = await validateArchive(archive)
  if (audit.errors.length) throw new Error(audit.errors.join('\n'))
  const { samples, windows, chunks, skipped } = await prepareRecords(audit.records, async record => decodePng(await readFile(path.join(archive, record.image.path))))
  const raw = Buffer.concat(chunks)
  await mkdir(path.dirname(output), { recursive: true })
  await writeFile(output + '.u8', raw)
  await writeFile(output + '.json', JSON.stringify({ manifestSha256: audit.manifestSha256, encoderSha256: sha256(encoderBytes),
    preparation: encoder.preparation, pixelSha256: sha256(raw), samples, windows, skipped }) + '\n')
  const counts = Object.entries(Object.groupBy(skipped, s => s.status)).map(([status, list]) => `${list.length} ${status}`).join(', ')
  console.log(`Prepared ${samples.length} references / ${windows.length} windows from archived pixels${skipped.length ? `; skipped ${skipped.length} (${counts})` : ''}`)
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main(...process.argv.slice(2))
