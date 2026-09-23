import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile, readFile, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { crc32, deflateSync } from 'node:zlib'
import {
  sha256, encodePng, decodePng, pngSize, inkBounds, padRegion, cropView, backgroundLuma,
  isSafeRelativePath, validateRecord, parseManifest, serializeManifest,
  mergeRecords, reconcileCaptures, validateArchive, readJsonl, writeJsonl,
  rankSpecimenCandidates, flagCapsOnlyFaces,
} from '../scripts/preview-lib.mjs'

const RECIPES = ['latin-lower-v1', 'latin-upper-v1', 'digits-v1', 'words-a-v1', 'words-b-v1']

test('specimen ranking preserves Unicode identities, canonical accents and meaningful punctuation', () => {
  const texts = ['你好世界', '漢字見本', 'Привет мир', 'Καλημέρα κόσμε', 'cafe\u0301', 'Regular', ' EXTRA   BOLD ', 'A+B', 'A-B']
  const result = rankSpecimenCandidates(texts.map(text => ({ text, fontSize: 48 })), { excludeTexts: ['漢字見本', 'CAFÉ', 'A+B'], labels: ['regular', 'extra bold'] })
  assert.deepEqual(result.preferred.map(c => c.text).sort(), ['你好世界', 'Привет мир', 'Καλημέρα κόσμε', 'A-B'].sort())
  assert.deepEqual(result.fallback.map(c => c.text), [' EXTRA   BOLD ', 'Regular'])
})

test('specimen ranking handles empty and malformed input and produces finite scores', () => {
  assert.deepEqual(rankSpecimenCandidates([]), { preferred: [], fallback: [] })
  assert.deepEqual(rankSpecimenCandidates([null, undefined, {}, { text: null }, { text: 42 }, { text: ' \n ' }]), { preferred: [], fallback: [] })
  const one = rankSpecimenCandidates([{ text: 'A' }])
  assert.deepEqual(one, { preferred: [{ text: 'A', score: 4 }], fallback: [] })
  for (const fontSize of [NaN, Infinity, -Infinity, -1, '48px', null]) assert.equal(rankSpecimenCandidates([{ text: 'A', fontSize }]).preferred[0].score, 4)
  for (const input of [null, undefined, {}, 'text']) assert.throws(() => rankSpecimenCandidates(input), /candidates/)
  assert.deepEqual(rankSpecimenCandidates([{ text: 'A' }], { labels: [null, undefined], excludeTexts: [null] }), one)
  for (const options of [{ labels: null }, { excludeTexts: 'text' }, { labels: [42] }]) assert.throws(() => rankSpecimenCandidates([], options), /text lists/)
})

test('specimen ranking prefers phrases, keeps stable ties and leaves A → A → B → A inputs unchanged', () => {
  const a = Object.freeze([Object.freeze({ text: 'a'.repeat(100), fontSize: 99999 }), Object.freeze({ text: 'two words', fontSize: 1 }), Object.freeze({ text: 'one two three', fontSize: 1 }), Object.freeze({ text: 'one two three', fontSize: 1, id: 'tie' })])
  const expected = rankSpecimenCandidates(a)
  assert.deepEqual(expected.preferred.map(c => c.text), ['one two three', 'one two three', 'two words', 'a'.repeat(100)])
  assert.equal(expected.preferred[0].id, undefined); assert.equal(expected.preferred[1].id, 'tie')
  assert.deepEqual(rankSpecimenCandidates(a), expected)
  assert.deepEqual(rankSpecimenCandidates([{ text: 'Regular' }], { labels: ['regular'] }), { preferred: [], fallback: [{ text: 'Regular', score: 28 }] })
  assert.deepEqual(rankSpecimenCandidates(a), expected)
  assert.ok(a.every(candidate => !('score' in candidate)))
})

/** A plain specimen: dark bar on a light ground, so ink bounds are predictable. */
function specimen({ width = 240, height = 80, seed = 0, background = 255 } = {}) {
  const pixels = Buffer.alloc(width * height * 4, background)
  for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255
  for (let y = 24; y < 56; y++) for (let x = 20 + seed; x < width - 20; x++) {
    const at = (y * width + x) * 4
    pixels[at] = pixels[at + 1] = pixels[at + 2] = 17
  }
  return encodePng({ width, height, channels: 4, pixels })
}

function record(overrides = {}) {
  const image = overrides.imageBytes ?? specimen()
  const { width, height } = pngSize(image)
  const base = {
    schemaVersion: 1,
    id: 'source-face-1-latin-lower-v1',
    sourceGroup: 'source-face-1-latin-lower-v1',
    role: 'reference',
    familyId: 'source:family-1',
    faceId: 'source:face-1',
    family: 'Example Serif',
    foundry: 'Example Foundry',
    styleName: 'Book',
    weight: 400,
    slant: 'upright',
    axes: {},
    sourceUrl: 'https://example.invalid/fonts/example-serif',
    capturedAt: '2026-09-22T12:00:00Z',
    recipeId: 'latin-lower-v1',
    text: 'abcdefghijklmnopqrstuvwxyz',
    scripts: ['Latn'],
    acquisition: 'browser-screenshot',
    render: { cssFontSize: 48, deviceScaleFactor: 2, zoom: 1 },
    image: { path: 'images/face-1-latin-lower-v1.png', width, height, sha256: sha256(image) },
    region: { x: 12, y: 20, width: 200, height: 40 },
    evidence: 'evidence/face-1.png',
    flags: [],
  }
  const { imageBytes, ...rest } = overrides
  return { record: { ...base, ...rest }, image }
}

async function archive(entries) {
  const dir = await mkdtemp(path.join(tmpdir(), 'preview-archive-'))
  await mkdir(path.join(dir, 'images'), { recursive: true })
  await mkdir(path.join(dir, 'evidence'), { recursive: true })
  const written = []
  for (const entry of entries) {
    const { record: row, image } = record(entry)
    await writeFile(path.join(dir, row.image.path), entry.skipImage ? Buffer.alloc(0) : image)
    await writeFile(path.join(dir, row.evidence), specimen({ width: 60, height: 40 }))
    written.push(row)
  }
  await writeFile(path.join(dir, 'manifest.jsonl'), serializeManifest(written))
  return { dir, records: written }
}

test('PNG round trip preserves size, pixels and reported dimensions', async () => {
  const bytes = specimen({ width: 120, height: 60 })
  const decoded = decodePng(bytes)
  assert.deepEqual(pngSize(bytes), { width: 120, height: 60, bitDepth: 8, colorType: 6, interlace: 0 })
  assert.equal(decoded.width, 120)
  assert.equal(decoded.pixels.length, 120 * 60 * 4)
  assert.equal(sha256(bytes), sha256(encodePng({ width: 120, height: 60, channels: 4, pixels: decoded.pixels })))
})

function scanlinePng(raw) {
  const original = encodePng({ width: 1, height: 1, channels: 4, pixels: Buffer.from([12, 34, 56, 78]) })
  const data = deflateSync(raw), body = Buffer.concat([Buffer.from('IDAT'), data]), chunk = Buffer.alloc(body.length + 8)
  chunk.writeUInt32BE(data.length); body.copy(chunk, 4); chunk.writeUInt32BE(crc32(body), chunk.length - 4)
  return Buffer.concat([original.subarray(0, 33), chunk, original.subarray(-12)])
}

test('PNG decoder checks the smallest image, exact final boundary, checksums and full scanlines', () => {
  const a = scanlinePng(Buffer.from([0, 12, 34, 56, 78])), b = scanlinePng(Buffer.from([0, 80, 60, 40, 20]))
  for (const [bytes, pixels] of [[a, [12, 34, 56, 78]], [a, [12, 34, 56, 78]], [b, [80, 60, 40, 20]], [a, [12, 34, 56, 78]]]) {
    const decoded = decodePng(bytes)
    assert.deepEqual([decoded.width, decoded.height, decoded.channels], [1, 1, 4])
    assert.deepEqual([...decoded.pixels], pixels)
  }
  for (const invalid of [null, Buffer.alloc(0), a.subarray(0, 8), a.subarray(0, 33), a.subarray(0, a.length - 12), a.subarray(0, a.length - 1), Buffer.concat([a, Buffer.from([0])])]) assert.throws(() => decodePng(invalid))
  const corrupt = Buffer.from(a); corrupt[42] ^= 1
  assert.throws(() => decodePng(corrupt), /checksum/)
  for (const length of [0, 4, 6]) assert.throws(() => decodePng(scanlinePng(Buffer.alloc(length))))
  assert.throws(() => decodePng(scanlinePng(Buffer.from([5, 1, 2, 3, 4]))), /filter/)
})

test('ink bounds find the glyphs, honour a search box and survive a tinted background', () => {
  const image = decodePng(specimen({ width: 200, height: 80 }))
  assert.deepEqual(inkBounds(image), { x: 20, y: 24, width: 160, height: 32 })
  assert.deepEqual(inkBounds(image, { within: { x: 40, y: 0, width: 60, height: 80 } }), { x: 40, y: 24, width: 60, height: 32 })
  const tinted = decodePng(specimen({ width: 200, height: 80, background: 250 }))
  assert.equal(Math.round(backgroundLuma(tinted).luma), 250)
  assert.deepEqual(inkBounds(tinted), { x: 20, y: 24, width: 160, height: 32 })
  assert.equal(inkBounds(decodePng(encodePng({ width: 8, height: 8, channels: 4, pixels: Buffer.alloc(8 * 8 * 4, 255) }))), null)
})

test('padded regions stay inside the image', () => {
  const image = { width: 100, height: 50 }
  assert.deepEqual(padRegion({ x: 10, y: 10, width: 20, height: 20 }, 8, image), { x: 2, y: 2, width: 36, height: 36 })
  assert.deepEqual(padRegion({ x: 0, y: 0, width: 100, height: 50 }, 8, image), { x: 0, y: 0, width: 100, height: 50 })
  const crop = cropView(decodePng(specimen({ width: 60, height: 40 })), { x: 10, y: 10, width: 20, height: 10 })
  assert.equal(crop.pixels.length, 20 * 10 * 4)
})

test('relative paths are refused when they escape the archive', () => {
  for (const good of ['images/a.png', 'evidence/b/c.png']) assert.ok(isSafeRelativePath(good))
  for (const bad of ['/etc/passwd', '../outside.png', 'images/../../x.png', '', null, 'images\\a.png'])
    assert.equal(isSafeRelativePath(bad), false, `${bad} must be refused`)
})

test('record schema rejects malformed metadata and accepts unknown optional keys', () => {
  assert.deepEqual(validateRecord(record().record), [])
  const cases = [
    [{ schemaVersion: 2 }, /schemaVersion/],
    [{ id: '' }, /id must be/],
    [{ role: 'query' }, /role/],
    [{ weight: 0 }, /weight/],
    [{ weight: 'bold' }, /weight/],
    [{ weight: Infinity }, /weight/],
    [{ axes: { wght: NaN } }, /axis/],
    [{ slant: 'slanted' }, /slant/],
    [{ axes: [] }, /axes/],
    [{ scripts: ['latin'] }, /scripts/],
    [{ acquisition: 'photograph' }, /acquisition/],
    [{ capturedAt: '22 September 2026' }, /capturedAt/],
    [{ text: 42 }, /text/],
    [{ evidence: '../evidence.png' }, /evidence/],
    [{ flags: 'none' }, /flags/],
  ]
  for (const [patch, pattern] of cases) {
    const problems = validateRecord(record(patch).record)
    assert.ok(problems.some(problem => pattern.test(problem)), `${JSON.stringify(patch)} → ${problems.join('; ')}`)
  }
  assert.deepEqual(validateRecord(record({ verification: { fontSha256: 'abc' }, source: { key: 'x' } }).record), [])
  assert.equal(validateRecord(null)[0], 'record is not a JSON object')
})

test('regions must be positive, integral and inside the image', () => {
  const cases = [
    [{ x: -1, y: 0, width: 10, height: 10 }, /region\.x/],
    [{ x: 0, y: 0, width: 0, height: 10 }, /region\.width/],
    [{ x: 0, y: 0, width: 10, height: 0 }, /region\.height/],
    [{ x: 0, y: 0, width: 10.5, height: 10 }, /region\.width/],
    [{ x: 200, y: 0, width: 100, height: 10 }, /width/],
    [{ x: 0, y: 70, width: 10, height: 40 }, /height/],
  ]
  for (const [region, pattern] of cases) {
    const problems = validateRecord(record({ region }).record)
    assert.ok(problems.some(problem => pattern.test(problem)), `${JSON.stringify(region)} → ${problems.join('; ')}`)
  }
})

test('a manifest with a broken line reports the line and keeps the rest', () => {
  const good = JSON.stringify(record().record)
  const { records, problems } = parseManifest(`${good}\n{"schemaVersion":1,\n\n${good}\n`)
  assert.equal(records.length, 2)
  assert.match(problems[0], /line 2: invalid JSON/)
})

test('archive validation catches missing, corrupt, mis-hashed and duplicated captures', async () => {
  const { dir } = await archive([{}])
  const ok = await validateArchive(dir)
  assert.deepEqual(ok.errors, [])
  assert.equal(ok.records.length, 1)

  const missing = await archive([{}])
  await rm(path.join(missing.dir, missing.records[0].image.path))
  assert.match((await validateArchive(missing.dir)).errors.join('\n'), /missing image file/)

  const corrupt = await archive([{}])
  await writeFile(path.join(corrupt.dir, corrupt.records[0].image.path), Buffer.from('not a png'))
  const corruptResult = await validateArchive(corrupt.dir)
  assert.match(corruptResult.errors.join('\n'), /sha256 mismatch/)
  assert.match(corruptResult.errors.join('\n'), /unreadable image/)

  const mismatched = await archive([{}])
  const patched = { ...mismatched.records[0], image: { ...mismatched.records[0].image, sha256: 'f'.repeat(64) } }
  await writeFile(path.join(mismatched.dir, 'manifest.jsonl'), serializeManifest([patched]))
  assert.match((await validateArchive(mismatched.dir)).errors.join('\n'), /sha256 mismatch/)

  const resized = await archive([{}])
  const wrongSize = { ...resized.records[0], image: { ...resized.records[0].image, width: 999 } }
  await writeFile(path.join(resized.dir, 'manifest.jsonl'), serializeManifest([wrongSize]))
  const resizedResult = await validateArchive(resized.dir)
  assert.match(resizedResult.errors.join('\n'), /does not match the record/)

  const duplicated = await archive([{}])
  await writeFile(path.join(duplicated.dir, 'manifest.jsonl'), serializeManifest([duplicated.records[0], duplicated.records[0]]))
  assert.match((await validateArchive(duplicated.dir)).errors.join('\n'), /duplicate record id/)

  const outOfBounds = await archive([{ region: { x: 0, y: 0, width: 10, height: 10 } }])
  const grown = { ...outOfBounds.records[0], region: { x: 0, y: 0, width: 5000, height: 10 } }
  await writeFile(path.join(outOfBounds.dir, 'manifest.jsonl'), serializeManifest([grown]))
  assert.match((await validateArchive(outOfBounds.dir)).errors.join('\n'), /region exceeds the image width/)
})

test('archive validation rejects empty/null records and PNGs with intact headers but broken pixels', async t => {
  const { dir, records } = await archive([{}])
  t.after(() => rm(dir, { recursive: true, force: true }))
  for (const manifest of ['', 'null\n']) {
    await writeFile(path.join(dir, 'manifest.jsonl'), manifest)
    assert.ok((await validateArchive(dir)).errors.length)
  }
  const bytes = scanlinePng(Buffer.from([0, 12, 34, 56]))
  const broken = { ...records[0], image: { ...records[0].image, width: 1, height: 1, sha256: sha256(bytes) }, region: { x: 0, y: 0, width: 1, height: 1 } }
  await writeFile(path.join(dir, broken.image.path), bytes)
  await writeFile(path.join(dir, 'manifest.jsonl'), serializeManifest([broken]))
  assert.match((await validateArchive(dir)).errors.join('\n'), /unreadable image.*scanlines/)
})

test('several regions may share one source image, but identical pixels across faces are an error', async () => {
  const shared = await archive([{}])
  const [first] = shared.records
  const second = {
    ...first, id: `${first.id}-b`, recipeId: 'words-a-v1', text: 'Hamburgefontsiv office minimum',
    region: { x: 12, y: 20, width: 100, height: 40 },
  }
  await writeFile(path.join(shared.dir, 'manifest.jsonl'), serializeManifest([first, second]))
  const sharedResult = await validateArchive(shared.dir)
  assert.deepEqual(sharedResult.errors, [])
  assert.equal(sharedResult.records.length, 2)

  const impostor = { ...second, faceId: 'source:face-2', styleName: 'Bold' }
  await writeFile(path.join(shared.dir, 'manifest.jsonl'), serializeManifest([first, impostor]))
  assert.match((await validateArchive(shared.dir)).errors.join('\n'), /identical pixels .* different faces/)
})

test('merging records replaces by id and never duplicates', () => {
  const first = record({ id: 'a' }).record, second = record({ id: 'b' }).record
  const merged = mergeRecords([first], [second, { ...first, styleName: 'Regular' }])
  assert.equal(merged.records.length, 2)
  assert.equal(merged.added, 1)
  assert.equal(merged.replaced, 1)
  assert.equal(merged.records.find(row => row.id === 'a').styleName, 'Regular')
})

/** A stand-in collector that exercises the real reconcile/merge/write path. */
async function collect(dir, faces) {
  const existing = await readJsonl(path.join(dir, 'manifest.jsonl'))
  const requested = faces.flatMap(face => RECIPES.map(recipe => ({ id: `${face}-${recipe}`, face, recipe })))
  const verdicts = new Map()
  for (const row of existing) {
    try {
      const bytes = await readFile(path.join(dir, row.image.path))
      verdicts.set(row.id, { ok: sha256(bytes) === row.image.sha256 })
    } catch { verdicts.set(row.id, { ok: false, reason: 'missing original' }) }
  }
  const { reuse, capture } = reconcileCaptures(requested, existing, row => verdicts.get(row.id) ?? { ok: false, reason: 'unknown' })
  const fresh = []
  for (const request of capture) {
    const bytes = specimen({ seed: [...request.face].reduce((total, character) => total + character.charCodeAt(0), 0) % 17 })
    const relative = `images/${request.id}.png`
    await writeFile(path.join(dir, relative), bytes)
    const { width, height } = pngSize(bytes)
    fresh.push(record({
      id: request.id, sourceGroup: request.id, faceId: `source:${request.face}`, recipeId: request.recipe,
      image: { path: relative, width, height, sha256: sha256(bytes) },
    }).record)
  }
  const merged = mergeRecords(existing, fresh)
  merged.records.sort((a, b) => a.id.localeCompare(b.id))
  await writeJsonl(path.join(dir, 'manifest.jsonl'), merged.records)
  return { reused: reuse.length, captured: capture.length, total: merged.records.length }
}

test('a repeated run reuses originals; a new face only adds records', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'preview-resume-'))
  await mkdir(path.join(dir, 'images'), { recursive: true })
  await mkdir(path.join(dir, 'evidence'), { recursive: true })
  await writeFile(path.join(dir, 'evidence/face-1.png'), specimen({ width: 60, height: 40 }))

  const first = await collect(dir, ['face-a'])
  assert.deepEqual([first.captured, first.reused, first.total], [5, 0, 5])
  const firstImage = path.join(dir, 'images/face-a-latin-lower-v1.png')
  const before = { bytes: await readFile(firstImage), stamp: (await stat(firstImage)).mtimeMs }

  const second = await collect(dir, ['face-a'])
  assert.deepEqual([second.captured, second.reused, second.total], [0, 5, 5])
  assert.deepEqual(await readFile(firstImage), before.bytes)
  assert.equal((await stat(firstImage)).mtimeMs, before.stamp, 'an intact original must not be rewritten')

  const third = await collect(dir, ['face-a', 'face-b'])
  assert.deepEqual([third.captured, third.reused, third.total], [5, 5, 10])
  assert.deepEqual(await readFile(firstImage), before.bytes)
  assert.equal(new Set((await readJsonl(path.join(dir, 'manifest.jsonl'))).map(row => row.id)).size, 10)

  const damaged = path.join(dir, 'images/face-b-digits-v1.png')
  await writeFile(damaged, Buffer.from('corrupted'))
  const fourth = await collect(dir, ['face-a', 'face-b'])
  assert.deepEqual([fourth.captured, fourth.reused, fourth.total], [1, 9, 10], 'only the damaged capture is redone')
  assert.deepEqual((await validateArchive(dir)).errors, [])
})

test('specimen ranking prefers phrases and sets style labels aside', () => {
  const candidates = [
    { text: 'Regular', fontSize: 120 },
    { text: 'Somebody loves you', fontSize: 160 },
    { text: 'Hello World', fontSize: 138 },
    { text: 'Sligoil', fontSize: 198 },
    { text: 'abcdefghijklmnopqrstuvwxyz', fontSize: 48 },
    { text: '   ', fontSize: 90 },
  ]
  const ranked = rankSpecimenCandidates(candidates, {
    excludeTexts: ['abcdefghijklmnopqrstuvwxyz'],
    labels: ['Regular', 'Bold', 'Sligoil'],
  })
  assert.deepEqual(ranked.preferred.map(candidate => candidate.text), ['Somebody loves you', 'Hello World'])
  assert.deepEqual(ranked.fallback.map(candidate => candidate.text), ['Sligoil', 'Regular'])
  assert.ok(ranked.preferred.every(candidate => candidate.score > 0))
  assert.deepEqual(rankSpecimenCandidates([], {}), { preferred: [], fallback: [] })
})

test('a face whose lowercase matches its capitals stops claiming small letters', () => {
  const line = (recipeId, faceId, ink, fontSize, extra = {}) => record({
    imageBytes: specimen({ width: 1600, height: 200 }),
    id: `${faceId}-${recipeId}`, faceId, recipeId,
    text: { 'latin-lower-v1': 'abcdefghijklmnopqrstuvwxyz', 'latin-upper-v1': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'words-a-v1': 'Hamburgefontsiv office minimum', 'digits-v1': '0123456789' }[recipeId],
    region: { x: 4, y: 4, width: ink.width + 16, height: ink.height + 16 },
    render: { cssFontSize: fontSize, deviceScaleFactor: 2, zoom: 1 },
    verification: { inkBounds: { x: 12, y: 12, ...ink }, ...extra },
  }).record

  // Measured only, captured at different tester sizes: the lowercase text is withheld.
  const measured = [line('latin-lower-v1', 'source:caps', { width: 503, height: 38 }, 24),
    line('latin-upper-v1', 'source:caps', { width: 723, height: 53 }, 34)]
  // Confirmed by inspection, drawn as small capitals: record the capitals actually drawn.
  const smallCaps = [line('latin-lower-v1', 'source:trajan', { width: 1300, height: 64 }, 48),
    line('latin-upper-v1', 'source:trajan', { width: 1700, height: 90 }, 48),
    line('words-a-v1', 'source:trajan', { width: 1500, height: 64 }, 48), line('digits-v1', 'source:trajan', { width: 600, height: 80 }, 48)]
  const text = [line('latin-lower-v1', 'source:text', { width: 1221, height: 93 }, 48),
    line('latin-upper-v1', 'source:text', { width: 876, height: 48 }, 24)]
  const mono = [line('latin-lower-v1', 'source:mono', { width: 1479, height: 93 }, 48, { fontSha256: 'a'.repeat(64) }),
    line('latin-upper-v1', 'source:mono', { width: 1482, height: 89 }, 48, { fontSha256: 'a'.repeat(64) })]
  const all = [...measured, ...smallCaps, ...text, ...mono]
  flagCapsOnlyFaces(all, { capsOnly: new Set(['source:trajan']) })

  assert.equal(measured[0].text, null, 'a measurement alone does not read the glyphs')
  assert.ok(measured[0].flags.includes('visible-text-not-verified'))
  assert.equal(measured[0].verification.requestedText, 'abcdefghijklmnopqrstuvwxyz')
  assert.deepEqual(validateRecord(measured[0]), [])

  assert.equal(smallCaps[0].text, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
  assert.equal(smallCaps[2].text, 'HAMBURGEFONTSIV OFFICE MINIMUM')
  assert.ok(smallCaps[0].flags.includes('lowercase-drawn-as-small-capitals'))
  assert.ok(smallCaps[2].flags.includes('caps-only-confirmed-by-inspection'))
  assert.equal(smallCaps[1].text, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
  assert.deepEqual(smallCaps[1].flags, [], 'the capitals line needs no transformation')
  assert.equal(smallCaps[3].text, '0123456789')

  assert.equal(text[0].text, 'abcdefghijklmnopqrstuvwxyz')
  assert.deepEqual(text[0].flags, [])
  assert.equal(mono[0].text, 'abcdefghijklmnopqrstuvwxyz', 'a binary-verified face is judged by its cmap, not by ink')

  // Recomputed from scratch: a second pass changes nothing, and dropping a declaration undoes it.
  const snapshot = JSON.stringify(all)
  flagCapsOnlyFaces(all, { capsOnly: new Set(['source:trajan']) })
  assert.equal(JSON.stringify(all), snapshot)
  flagCapsOnlyFaces(all)
  assert.equal(smallCaps[2].text, 'Hamburgefontsiv office minimum')
  assert.deepEqual(smallCaps[2].flags, [])
})
