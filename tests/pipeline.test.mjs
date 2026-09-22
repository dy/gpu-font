import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const baseline = fileURLToPath(new URL('../scripts/baseline.mjs', import.meta.url))
const preparation = fileURLToPath(new URL('../scripts/prepare.mjs', import.meta.url))

async function fixture(t) {
  const cwd = await mkdtemp(join(tmpdir(), 'gpu-font-test-'))
  t.after(() => rm(cwd, { recursive: true, force: true }))
  for (const folder of ['bench', 'src', '.data']) await mkdir(join(cwd, folder))
  const write = (path, bytes) => writeFile(join(cwd, path), typeof bytes === 'string' || Buffer.isBuffer(bytes) ? bytes : JSON.stringify(bytes))
  const fonts = [{ id: 'a', file: 'a.ttf', split: 'train' }]
  const config = { fonts, input: { width: 4, height: 4, padding: 1 } }
  const sources = { 'bench/pilot.json': JSON.stringify(config), 'bench/fonts.json': JSON.stringify({ fonts }), 'src/prepare.mjs': 'normalizer-v1' }
  for (const [path, bytes] of Object.entries(sources)) await write(path, bytes)
  // Smallest descriptor image: one centered ink pixel, repeated for a distinct-text query.
  const tensor = Buffer.alloc(18 * 4)
  for (let i = 0; i < 18; i++) tensor.writeFloatLE(i % 9 === 4 ? 0 : 1, i * 4)
  await write('.data/prepared.f32', tensor)
  const samples = ['prototype', 'query'].map(role => ({ id: role, role, family: 'a', split: 'train', renderer: 'fixture', text: role }))
  const manifest = { width: 3, height: 3, samples,
    configSha256: hash(sources['bench/pilot.json']), fontsSha256: hash(sources['bench/fonts.json']), preparationSha256: hash(sources['src/prepare.mjs']), tensorSha256: hash(tensor) }
  await write('.data/prepared.json', manifest)
  const run = script => spawnSync(process.execPath, [script], { cwd, encoding: 'utf8' })
  return { cwd, write, sources, config, tensor, manifest, run }
}

test('baseline validates freshness, exact tensor boundaries, and catalog-dependent chance', async t => {
  const f = await fixture(t)
  const valid = () => {
    const result = f.run(baseline)
    assert.equal(result.status, 0, result.stderr)
  }
  valid()
  const report = JSON.parse(await readFile(join(f.cwd, 'bench/baseline.json')))
  assert.deepEqual(report.chance, { top1: 1, top5: 1 })
  assert.equal(report.reports.grayscale.groups['fixture/train'].top1, 1)
  // A → stale B → A for each independently mutable source.
  for (const [path, source] of Object.entries(f.sources)) {
    await f.write(path, source + ' ')
    const result = f.run(baseline)
    assert.notEqual(result.status, 0)
    assert.match(result.stderr, /Stale preparation/)
    await f.write(path, source)
    valid()
  }
  // A truncated final float and a trailing byte both reject even with a matching hash.
  for (const tensor of [f.tensor.subarray(0, -1), Buffer.concat([f.tensor, Buffer.from([0])])]) {
    await f.write('.data/prepared.f32', tensor)
    await f.write('.data/prepared.json', { ...f.manifest, tensorSha256: hash(tensor) })
    assert.match(f.run(baseline).stderr, /tensor length mismatch/)
  }
  await f.write('.data/prepared.json', f.manifest)
  assert.match(f.run(baseline).stderr, /tensor checksum failed/)
  await f.write('.data/prepared.f32', f.tensor)
  for (const samples of [[], f.manifest.samples.slice(0, 1), f.manifest.samples.slice(1)]) {
    await f.write('.data/prepared.json', { ...f.manifest, samples })
    assert.match(f.run(baseline).stderr, /Expected prototype and query/)
  }
  await f.write('.data/prepared.json', f.manifest)
  valid()
})

test('preparation rejects a changed font selection before reading raw fixtures', async t => {
  const f = await fixture(t)
  for (const fonts of [[...f.config.fonts, { id: 'b', file: 'b.ttf', split: 'unseen-dev' }], [{ ...f.config.fonts[0], split: 'unseen-dev' }], [{ ...f.config.fonts[0], file: 'other.ttf' }]]) {
    await f.write('bench/pilot.json', { ...f.config, fonts })
    const result = f.run(preparation)
    assert.notEqual(result.status, 0)
    assert.match(result.stderr, /Font selection changed/)
  }
  await f.write('bench/pilot.json', f.sources['bench/pilot.json'])
  const rgba = Buffer.alloc(3 * 3 * 4, 255)
  rgba.fill(0, 4 * 4, 4 * 4 + 3)
  await f.write('.data/ink.rgba', rgba)
  const samples = f.manifest.samples.map(s => ({ ...s, width: 3, height: 3, raw: '.data/ink.rgba' }))
  for (const renderer of ['pillow', 'browser']) {
    await mkdir(join(f.cwd, '.data', renderer))
    await f.write(`.data/${renderer}/samples.json`, { ...f.manifest, samples: renderer === 'pillow' ? samples : [] })
  }
  const valid = f.run(preparation)
  assert.equal(valid.status, 0, valid.stderr)
  const tensor = await readFile(join(f.cwd, '.data/prepared.f32'))
  assert.equal(tensor.length, 2 * 4 * 4 * 4)
  let ink = 0
  for (let p = 0; p < 16; p++) ink += 1 - tensor.readFloatLE(p * 4)
  assert.ok(Math.abs(ink - (2 / 3) ** 2) < 1e-6)
  for (const renderer of ['pillow', 'browser']) await f.write(`.data/${renderer}/samples.json`, { ...f.manifest, samples: [] })
  assert.match(f.run(preparation).stderr, /No samples to prepare/)
})
