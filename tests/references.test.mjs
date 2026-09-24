import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { deflateSync } from 'node:zlib'
import { LATIN, plan, packVectors, joinCatalogs } from '../src/references.mjs'
import { readFontFile, styleOf, FONT_FILE } from '../src/font-file.mjs'
import { readCatalog, unit, unpack } from '../src/catalog.mjs'

// Fonts are built here from their tables, so no font file enters the repository.
const utf16 = text => Buffer.from(text, 'utf16le').swap16()
// Windows English names (platform 3, UTF-16) by default; platform 1 writes Mac Roman.
function nameTable(names, platform = 3) {
  const strings = Object.entries(names).map(([id, text]) => [Number(id), platform === 1 ? Buffer.from(text, 'latin1') : utf16(text)]), header = Buffer.alloc(6 + 12 * strings.length)
  header.writeUInt16BE(strings.length, 2); header.writeUInt16BE(header.length, 4)
  let offset = 0
  strings.forEach(([id, bytes], i) => {
    const r = 6 + i * 12
    header.writeUInt16BE(platform, r); header.writeUInt16BE(platform === 1 ? 0 : 1, r + 2); header.writeUInt16BE(platform === 1 ? 0 : 0x409, r + 4); header.writeUInt16BE(id, r + 6)
    header.writeUInt16BE(bytes.length, r + 8); header.writeUInt16BE(offset, r + 10); offset += bytes.length
  })
  return Buffer.concat([header, ...strings.map(([, bytes]) => bytes)])
}
function os2(weight, italic) { const b = Buffer.alloc(78); b.writeUInt16BE(weight, 4); b.writeUInt16BE(italic ? 1 : 0x40, 62); return b }
const face = (family, style, weight, italic, postscript) => ({ name: nameTable({ 1: family, 2: style, 6: postscript }), 'OS/2': os2(weight, italic) })
const pad = b => Buffer.concat([b, Buffer.alloc((4 - b.length % 4) % 4)])
// Table directories at `at`, table data after `dataAt`; offsets are from the file's start, as in collections.
function directory(tables, dataAt) {
  const entries = Object.entries(tables), dir = Buffer.alloc(12 + 16 * entries.length), data = []
  dir.writeUInt32BE(0x00010000, 0); dir.writeUInt16BE(entries.length, 4)
  let offset = dataAt
  entries.forEach(([tag, bytes], i) => {
    const r = 12 + i * 16
    dir.write(tag.padEnd(4), r, 'latin1'); dir.writeUInt32BE(offset, r + 8); dir.writeUInt32BE(bytes.length, r + 12)
    data.push(pad(bytes)); offset += pad(bytes).length
  })
  return { dir, data: Buffer.concat(data) }
}
const ttf = tables => { const { dir, data } = directory(tables, 12 + 16 * Object.keys(tables).length); return Buffer.concat([dir, data]) }
function ttc(faces) {
  const head = Buffer.alloc(12 + 4 * faces.length), dirs = [], datas = []
  head.write('ttcf', 0, 'latin1'); head.writeUInt32BE(0x00010000, 4); head.writeUInt32BE(faces.length, 8)
  let at = head.length + faces.reduce((sum, t) => sum + 12 + 16 * Object.keys(t).length, 0), dirAt = head.length
  for (const [i, tables] of faces.entries()) {
    const { dir, data } = directory(tables, at)
    head.writeUInt32BE(dirAt, 12 + i * 4); dirAt += dir.length; at += data.length; dirs.push(dir); datas.push(data)
  }
  return Buffer.concat([head, ...dirs, ...datas])
}
function woff(tables, compress = true) {
  const entries = Object.entries(tables), header = Buffer.alloc(44 + 20 * entries.length), data = []
  header.write('wOFF', 0, 'latin1'); header.writeUInt32BE(0x00010000, 4); header.writeUInt16BE(entries.length, 12)
  let offset = header.length
  entries.forEach(([tag, bytes], i) => {
    const r = 44 + i * 20, packed = compress ? deflateSync(bytes) : bytes
    header.write(tag.padEnd(4), r, 'latin1'); header.writeUInt32BE(offset, r + 4); header.writeUInt32BE(packed.length, r + 8); header.writeUInt32BE(bytes.length, r + 12)
    data.push(pad(packed)); offset += pad(packed).length
  })
  return Buffer.concat([header, ...data])
}
const file = (name, bytes) => new File([bytes], name)
const described = faces => faces.map(({ source, ...rest }) => rest)

test('font files: family, style and weight from name and OS/2, typographic names first, per face of a collection', async () => {
  assert.deepEqual(described(await readFontFile(file('a.ttf', ttf(face('Lora', 'Bold Italic', 700, true, 'Lora-BoldItalic'))))),
    [{ family: 'Lora', styleName: 'Bold Italic', postscriptName: 'Lora-BoldItalic', weight: 700, style: 'italic' }])
  const typographic = { name: nameTable({ 1: 'Inter Light', 2: 'Regular', 16: 'Inter', 17: 'Light', 6: 'Inter-Light' }), 'OS/2': os2(300, false) }
  assert.deepEqual(described(await readFontFile(file('b.otf', ttf(typographic)))), [{ family: 'Inter', styleName: 'Light', postscriptName: 'Inter-Light', weight: 300, style: 'normal' }])
  // Each collection member becomes a standalone font whose own tables read back.
  const members = await readFontFile(file('c.ttc', ttc([face('Serif', 'Regular', 400, false, 'Serif-Regular'), face('Serif', 'Bold', 700, false, 'Serif-Bold')])))
  assert.deepEqual(described(members).map(f => [f.postscriptName, f.weight]), [['Serif-Regular', 400], ['Serif-Bold', 700]])
  for (const member of members) assert.deepEqual(described(await readFontFile(file('d.ttf', new Uint8Array(member.source)))), described([member]))
  assert.deepEqual(described(await readFontFile(file('e.woff', woff(face('Sans', 'Medium', 500, false, 'Sans-Medium'))))), [{ family: 'Sans', styleName: 'Medium', postscriptName: 'Sans-Medium', weight: 500, style: 'normal' }])
  // WOFF2 tables are Brotli-compressed: the file name names the face.
  assert.deepEqual(described(await readFontFile(file('Work_Sans-SemiBoldItalic.woff2', Buffer.from('wOF2\0\0\0\0\0\0\0\0')))), [{ family: 'Work Sans', styleName: 'Semi Bold Italic', postscriptName: undefined, weight: 600, style: 'italic' }])
  await assert.rejects(readFontFile(file('f.ttf', Buffer.from('not a font at all'))), /not a font/)
  assert.ok(['a.TTF', 'b.otc', 'c.woff2'].every(n => FONT_FILE.test(n)) && !FONT_FILE.test('d.pfb'))
})

test('style names map to CSS weights, longer names first', () => {
  assert.deepEqual(['Thin', 'ExtraLight', 'Light', 'Regular', 'Book', 'Medium', 'SemiBold', 'Demi Bold', 'Bold', 'Extra Bold', 'Black', 'Heavy Oblique'].map(n => styleOf(n)),
    [100, 200, 300, 400, 400, 500, 600, 600, 700, 800, 900, 900].map((weight, i) => ({ weight, style: i === 11 ? 'italic' : 'normal' })))
})

test('reference lines are the training catalog lines', async () => {
  const source = await readFile(new URL('../train/style_catalog.py', import.meta.url), 'utf8')
  assert.deepEqual(LATIN, JSON.parse(source.match(/^LATIN = (\{.*\})$/m)[1].replaceAll("'", '"')))
})

const binding = { encoderSha256: 'e'.repeat(64), preparationSha256: 'p'.repeat(64) }
const catalogOf = (ids, rows, owners) => ({ version: 3, kind: 'font-catalog', ...binding, dimensions: 128,
  faces: ids.map(id => ({ id, familyId: `local:${id}`, family: id })), vectors: { ...packVectors(rows), owners } })
const random = seed => Array.from({ length: 128 }, (_, d) => Math.sin(seed * 97 + d * 13.1))

test('packed rows are 4 bits at one scale per row, each within half a step, read back as decoded', () => {
  const rows = [1, 2, 3].map(s => Array.from(unit(random(s)))), catalog = catalogOf(['a', 'b'], rows, [0, 0, 1]), read = readCatalog(catalog, binding)
  const bytes = Buffer.from(catalog.vectors.data, 'base64'), values = unpack(bytes, 4, 3 * 128)
  assert.deepEqual([catalog.vectors.encoding, bytes.length], ['int4-base64', 3 * 64])
  rows.forEach((row, r) => {
    const scale = catalog.vectors.scales[r], own = values.subarray(r * 128, r * 128 + 128)
    // The largest component maps to ±7, as the repository's exporter packs it; scales keep 6 significant digits.
    assert.equal(Math.max(...Array.from(own, Math.abs)), 7)
    row.forEach((v, d) => assert.ok(Math.abs(own[d] * scale - v) <= scale / 2 * (1 + 1e-5), `row ${r}, dimension ${d}`))
    const decoded = unit(Array.from(own, q => Math.fround(q * scale)))
    read.vectors.subarray(r * 128, r * 128 + 128).forEach((v, d) => assert.ok(Math.abs(v - decoded[d]) < 1e-6))
  })
  // A 4-bit stream a byte short or long is refused, as an 8-bit one is; so is an encoding nobody writes.
  for (const data of [bytes.subarray(0, -1), Buffer.concat([bytes, Buffer.alloc(1)])]) assert.throws(() => readCatalog({ ...catalog, vectors: { ...catalog.vectors, data: data.toString('base64') } }, binding), /Truncated|Invalid vector bytes/)
  assert.throws(() => readCatalog({ ...catalog, vectors: { ...catalog.vectors, encoding: 'int5-base64' } }, binding), /shape/)
  assert.throws(() => joinCatalogs(...[catalog, catalog].map(c => ({ ...c, vectors: { ...c.vectors, encoding: 'int5-base64' } }))), /Unknown vector encoding/)
})

test('joining catalogs keeps every face once, the newer copy of a repeated one, with its own rows', () => {
  const rows = [1, 2, 3, 4, 5].map(s => Array.from(unit(random(s))))
  const first = catalogOf(['a', 'b'], rows.slice(0, 3), [0, 1, 1]), second = catalogOf(['b', 'c'], rows.slice(3), [0, 1])
  const joined = joinCatalogs(first, second), read = readCatalog(joined, binding)
  assert.deepEqual(joined.faces.map(f => f.id), ['a', 'b', 'c'])
  assert.deepEqual(read.owners, [0, 1, 2])
  // a keeps its row; b's two old rows are replaced by its one new row. Rows are copied as packed, bit for bit.
  const row = (catalog, r) => Array.from(catalog.vectors.subarray(r * 128, r * 128 + 128)), a = readCatalog(first, binding), b = readCatalog(second, binding)
  assert.deepEqual([row(read, 0), row(read, 1), row(read, 2)], [row(a, 0), row(b, 0), row(b, 1)])
  // Rows packed at another width cannot be copied into these.
  assert.throws(() => joinCatalogs(first, { ...second, vectors: { ...second.vectors, encoding: 'int8-base64' } }), /differently/)
})

test('font file edges: one-face collection, stored WOFF tables, no OS/2, Mac names only, truncated data', async () => {
  const [only] = await readFontFile(file('a.ttc', ttc([face('Solo', 'Regular', 400, false, 'Solo-Regular')])))
  assert.deepEqual(described([only]), [{ family: 'Solo', styleName: 'Regular', postscriptName: 'Solo-Regular', weight: 400, style: 'normal' }])
  assert.deepEqual(described(await readFontFile(file('b.woff', woff(face('Raw', 'Bold', 700, false, 'Raw-Bold'), false)))), [{ family: 'Raw', styleName: 'Bold', postscriptName: 'Raw-Bold', weight: 700, style: 'normal' }])
  // Without OS/2 the style name gives weight and slope.
  assert.deepEqual(described(await readFontFile(file('c.ttf', ttf({ name: nameTable({ 1: 'Bare', 2: 'Black Italic' }) })))), [{ family: 'Bare', styleName: 'Black Italic', postscriptName: undefined, weight: 900, style: 'italic' }])
  // Mac Roman names (platform 1) are read when no Unicode name exists.
  assert.deepEqual(described(await readFontFile(file('Old-Roman.ttf', ttf({ name: nameTable({ 1: 'Chicago', 2: 'Bold' }, 1) })))), [{ family: 'Chicago', styleName: 'Bold', postscriptName: undefined, weight: 700, style: 'normal' }])
  // Tables that run past the file's end reject rather than yield a face.
  const cut = ttf(face('Cut', 'Regular', 400, false, 'Cut-Regular'))
  await assert.rejects(readFontFile(file('d.ttf', cut.subarray(0, 40))))
  await assert.rejects(readFontFile(file('e.ttf', Buffer.alloc(4))), /not a font/)
})

test('reference plan: fixed lines for a full case, own letters when some are missing, nothing under eight', () => {
  // A letter the face lacks measures differently under the two fallbacks.
  const ctx = missing => ({ font: '', measureText(letter) { return { width: missing.includes(letter) && this.font.endsWith('monospace') ? 1 : 2 } } })
  assert.deepEqual(plan(ctx(''), 'F'), { 'Latn-lower': LATIN.lower, 'Latn-upper': LATIN.upper })
  assert.deepEqual(plan(ctx('qxz'), 'F'), { 'Latn-lower': ['abcdefghijkl', 'mnoprstuvwy'], 'Latn-upper': LATIN.upper })
  assert.deepEqual(plan(ctx('ABCDEFGHIJKLMNOPQRSTU'), 'F'), { 'Latn-lower': LATIN.lower }, 'Five capitals are too few')
  assert.deepEqual(plan(ctx('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'F'), {}, 'No Latin letters, no references')
})

test('joining edges: a catalog with itself, and one replacing every face', () => {
  const rows = [1, 2, 3].map(s => Array.from(unit(random(s)))), a = catalogOf(['a', 'b'], rows.slice(0, 2), [0, 1]), b = catalogOf(['a', 'b'], rows.slice(2).concat([rows[0]]), [1, 0])
  assert.deepEqual(joinCatalogs(a, a).vectors, a.vectors)
  assert.deepEqual(joinCatalogs(a, b), b)
})
