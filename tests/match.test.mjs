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
  assert.equal(matcher.threshold, model.threshold, 'The matcher exposes the model\'s rejection threshold, null when it ships none')
  assert.deepEqual(await matcher.embed(image()), embedding, 'embed gives the crop\'s embedding, the one match ranked by')
  assert.equal(await matcher.embed(image(false)), null); assert.equal(typeof matcher.encoder.project, 'function')
  matcher.destroy()
})

test('the matcher reads a textured crop as it stands and cleaned, each family at its better, and embeds it as its cleaned reading', async () => {
  const matcher = await createMatcher(), { clean } = await import('../src/clean.mjs'), { bestOf } = await import('../src/catalog.mjs')
  const bytes = await readFile(new URL('models/encoder/encoder.json', root)), artifact = JSON.parse(bytes), model = readNetwork(artifact), heads = readHeads(artifact)
  const catalog = readCatalog(JSON.parse(await readFile(new URL('models/encoder/google-fonts.json', root))), { encoderSha256: await sha256(bytes), preparationSha256: await preparationHash(artifact.preparation) })
  let seed = 3; const noise = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31
  const textured = image(); for (let i = 0; i < textured.data.length; i += 4) if (textured.data[i] === 255) { const v = 60 + 190 * noise(); textured.data.set([v, v * .8, 170], i) }  // the letters on a coarse grainy wall
  const embedOf = crop => embedWindows(prepareLine(crop, { ...model.preparation, deskew: true, sampler: 'windows' }).windows.map(w => inferCPU(model, w)))
  assert.notEqual(clean(textured), textured, 'the wall is not two-tone')
  const asIs = embedOf(textured), cleaned = embedOf(clean(textured)), judged = heads ? verdict(cleaned, heads) : null
  assert.deepEqual(await matcher.embed(textured), cleaned)  // the principal reading: the line cleaned
  assert.deepEqual(await matcher.match(textured), foldTwins(bestOf([asIs, cleaned].map(e => matchCatalog(e, catalog, judged))), catalog, { script: judged?.script?.[0]?.label }))
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
  try { await assert.rejects(createMatcher(pathToFileURL(file).href), /different model/) } finally { await rm(dir, { recursive: true }) }
})

test('createMatcher names the catalog or model that failed to load', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => ({ ok: false, status: 404 })
  try {
    await assert.rejects(createMatcher('https://example.com/catalog.json'), /Cannot load https:\/\/example\.com\/catalog\.json: HTTP 404/)
    await assert.rejects(createMatcher(catalogs.other, 'https://example.com/model.json'), /Cannot load https:\/\/example\.com\/model\.json: HTTP 404/)
  } finally { globalThis.fetch = original }
})

test('several crops\' matches merge with each family at its best score, ranked as one crop\'s are', async () => {
  const { bestOf } = await import('../src/catalog.mjs')
  const a = [{ family: 'x', score: .9 }, { family: 'y', score: .5 }], b = [{ family: 'y', score: .95 }, { family: 'z', score: .9 }]
  assert.deepEqual(bestOf([a, b]).map(m => [m.family, m.score]), [['y', .95], ['x', .9], ['z', .9]])  // a tie in score keeps family order
  assert.deepEqual(bestOf([a]), a); assert.deepEqual(bestOf([]), [])
})

test('the matcher reads a crop of two lines line by line: a family found on either line', async () => {
  const matcher = await createMatcher(), { lines, cut } = await import('../src/clean.mjs')
  const width = 200, height = 120, data = new Uint8ClampedArray(width * height * 4).fill(255)
  for (const [top, bottom] of [[10, 40], [70, 100]]) for (let y = top; y < bottom; y++) for (let x = 10; x < 190; x++) if (x % 14 < 4) data.fill(0, (y * width + x) * 4, (y * width + x) * 4 + 3)
  const two = { width, height, data }, parts = lines(two)
  assert.equal(parts.length, 2)
  const each = await Promise.all(parts.map(r => matcher.match(cut(two, r)))), both = await matcher.match(two)
  for (const m of both.slice(0, 5)) assert.ok(each.some(list => list.some(n => n.family === m.family && Math.abs(n.score - m.score) < 1e-6)), `${m.family} at its best line`)
  assert.deepEqual(await matcher.embed(two), await matcher.embed(cut(two, parts[0])))
  matcher.destroy()
})

test('the preparation modules are the bytes the shipped encoder was trained and bound with', async () => {
  // A change to any of them changes what every catalog row means: the page would still load, and match wrongly.
  const { createHash } = await import('node:crypto'), artifact = JSON.parse(await readFile(new URL('models/encoder/encoder.json', root)))
  for (const [file, key] of [['src/input.mjs', 'sha256'], ['src/prepare.mjs', 'normalizerSha256'], ['src/line.mjs', 'lineSha256']])
    assert.equal(createHash('sha256').update(await readFile(new URL(file, root))).digest('hex'), artifact.preparation[key], `${file} differs from the encoder's preparation`)
})
