import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { createMatcher, catalogs } from '../src/match.mjs'
import { readNetwork, inferCPU } from '../src/network.mjs'
import { prepareLine } from '../src/line.mjs'
import { readCatalog, embedWindows, matchCatalog, foldTwins, sha256, preparationHash, readHeads, verdict } from '../src/catalog.mjs'

const root = new URL('../', import.meta.url)
const image = (ink = true) => {
  const width = 200, height = 48, data = new Uint8ClampedArray(width * height * 4).fill(255)
  if (ink) for (let y = 10; y < 38; y++) for (let x = 10; x < 190; x++) if (x % 14 < 4) data.fill(0, (y * width + x) * 4, (y * width + x) * 4 + 3)
  return { width, height, data }
}

// Node has no WebGPU, so this exercises the CPU fallback; the default model and catalog are read from disk.
test('createMatcher ranks exactly as the modules it wraps, best first, and returns nothing for a blank crop', async () => {
  const matcher = await createMatcher()
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

test('catalogs names every shipped catalog, and each loads with the default model', async () => {
  const index = JSON.parse(await readFile(new URL('models/encoder/catalogs/index.json', root)))
  assert.deepEqual(Object.keys(catalogs), ['google-fonts', ...index.map(c => c.id)])
  for (const url of Object.values(catalogs)) (await createMatcher(url)).destroy()
})

test('createMatcher refuses a catalog built for another model', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'gpu-font-')), file = join(dir, 'catalog.json')
  await writeFile(file, JSON.stringify({ ...JSON.parse(await readFile(new URL('models/encoder/google-fonts.json', root))), encoderSha256: '0'.repeat(64) }))
  try { await assert.rejects(createMatcher(pathToFileURL(file).href), /different encoder/) } finally { await rm(dir, { recursive: true }) }
})

test('createMatcher names the catalog or model that failed to load', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => ({ ok: false, status: 404 })
  try {
    await assert.rejects(createMatcher('https://example.com/catalog.json'), /Cannot load https:\/\/example\.com\/catalog\.json: HTTP 404/)
    await assert.rejects(createMatcher(catalogs.fontshare, 'https://example.com/model.json'), /Cannot load https:\/\/example\.com\/model\.json: HTTP 404/)
  } finally { globalThis.fetch = original }
})
