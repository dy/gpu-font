import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { createMatcher } from '../src/match.mjs'
import { readNetwork, inferCPU } from '../src/network.mjs'
import { prepareLine } from '../src/line.mjs'
import { readCatalog, embedWindows, matchCatalog, foldTwins, sha256, preparationHash, readHeads, verdict } from '../src/catalog.mjs'

const root = new URL('../', import.meta.url)
// Node has no WebGPU, so this exercises the CPU fallback; fetch reads the shipped files.
globalThis.fetch = async path => { const bytes = await readFile(new URL(path, root)); return { arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), json: async () => JSON.parse(bytes) } }
const image = (ink = true) => {
  const width = 200, height = 48, data = new Uint8ClampedArray(width * height * 4).fill(255)
  if (ink) for (let y = 10; y < 38; y++) for (let x = 10; x < 190; x++) if (x % 14 < 4) data.fill(0, (y * width + x) * 4, (y * width + x) * 4 + 3)
  return { width, height, data }
}

test('createMatcher ranks exactly as the modules it wraps, best first, and returns nothing for a blank crop', async () => {
  const matcher = await createMatcher('models/encoder/encoder.json', 'models/encoder/google-fonts.json')
  const matches = await matcher.match(image())
  const bytes = await readFile(new URL('models/encoder/encoder.json', root)), artifact = JSON.parse(bytes), model = readNetwork(artifact), heads = readHeads(artifact)
  const catalog = readCatalog(JSON.parse(await readFile(new URL('models/encoder/google-fonts.json', root))), { encoderSha256: await sha256(bytes), preparationSha256: await preparationHash(artifact.preparation) })
  const { windows } = prepareLine(image(), { ...model.preparation, deskew: true, sampler: 'windows' })
  const embedding = embedWindows(windows.map(w => inferCPU(model, w))), judged = heads ? verdict(embedding, heads) : null
  assert.deepEqual(matches, foldTwins(matchCatalog(embedding, catalog, judged), catalog, { script: judged?.script?.[0]?.label }))
  assert.ok(matches.length > 5 && matches.every((m, i) => !i || m.score <= matches[i - 1].score) && typeof matches[0].face.family === 'string')
  assert.deepEqual(await matcher.match(image(false)), [])
  matcher.destroy()
})

test('createMatcher refuses a catalog built for another model', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async path => path.endsWith('catalog.json') ? { json: async () => ({ ...JSON.parse(await readFile(new URL('models/encoder/google-fonts.json', root))), encoderSha256: '0'.repeat(64) }) } : original(path)
  try { await assert.rejects(createMatcher('models/encoder/encoder.json', 'other/catalog.json'), /different encoder/) } finally { globalThis.fetch = original }
})
