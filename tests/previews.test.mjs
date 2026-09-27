import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile, readdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { previewPath, dafontPreview, previewAddress } from '../previews.mjs'

const read = async path => JSON.parse(await readFile(path, 'utf8'))
// A lossless WebP's size, from its VP8L header: a 0x2f signature, then 14 bits of width - 1 and 14 of height - 1.
function webpSize(bytes) {
  if (bytes.toString('latin1', 0, 4) !== 'RIFF' || bytes.toString('latin1', 8, 16) !== 'WEBPVP8L' || bytes[20] !== 0x2f) return null
  const bits = bytes.readUInt32LE(21)
  return { width: (bits & 0x3fff) + 1, height: ((bits >> 14) & 0x3fff) + 1 }
}

test('a face id files its preview under the SHA-1 of its UTF-8 bytes, as scripts/previews.py writes it', async () => {
  for (const id of ['dafont:face:-aula-402', 'uncut-/HanaMeatball-Medium', '花園肉丸', '']) {
    const hex = createHash('sha1').update(Buffer.from(id, 'utf8')).digest('hex')
    assert.equal(await previewPath(id), `previews/${hex.slice(0, 2)}/${hex.slice(2, 16)}.webp`, id)
  }
})

test('a DaFont face shows DaFont\'s own preview: its slug with _ for -, then its previewFile, 0 when the catalog gives none', async () => {
  // Addresses as DaFont lists them (its category pages, .data/previews/dafont/listing.jsonl).
  for (const [face, address] of [
    [{ id: 'dafont:face:matcha-world', previewFile: 1 }, 'https://www.dafont.com/img/preview/m/a/matcha_world1.png'],
    [{ id: 'dafont:face:-aula-402' }, 'https://www.dafont.com/img/preview/_/a/_aula_4020.png'],
    [{ id: 'dafont:face:super-mario-256' }, 'https://www.dafont.com/img/preview/s/u/super_mario_2560.png'],  // a slug ending in digits
    [{ id: 'dafont:face:encamp', previewFile: 13 }, 'https://www.dafont.com/img/preview/e/n/encamp13.png'],
  ]) assert.equal(dafontPreview(face), address)
  assert.equal(await previewAddress({ id: 'dafont:face:encamp', previewFile: 13 }), 'https://www.dafont.com/img/preview/e/n/encamp13.png')
  assert.equal(await previewAddress({ id: 'debian-tork/Tork Bd' }), await previewPath('debian-tork/Tork Bd'), 'Any other face has its stored picture')
})

test('every shipped DaFont face\'s address is the one DaFont lists for it', { skip: !existsSync('.data/previews/dafont/listing.jsonl') && 'needs the local DaFont listing' }, async () => {
  const listed = new Map((await readFile('.data/previews/dafont/listing.jsonl', 'utf8')).trim().split('\n').map(line => JSON.parse(line)).map(row => [row.slug, row.preview]))
  const wrong = (await read('models/encoder/catalogs/dafont.json')).faces.filter(face => dafontPreview(face) !== listed.get(face.id.slice('dafont:face:'.length)))
  assert.deepEqual(wrong.slice(0, 5).map(face => face.id), [], `${wrong.length} DaFont faces point elsewhere`)
})

test('every face of a shipped catalog but Google Fonts\' and DaFont\'s has a 64-pixel lossless WebP preview or is listed without one; no other picture ships', async () => {
  const site = await read('site.json'), report = await read('bench/previews.json'), missing = new Set(report.missing.map(m => m.id)), shipped = new Set(), expected = new Set()
  for (const option of site.catalogs.filter(c => !['google-fonts', 'dafont'].includes(c.id)))
    for (const face of (await read(option.file)).faces) { shipped.add(face.id); if (!missing.has(face.id)) expected.add(await previewPath(face.id)) }
  assert.deepEqual([...missing].filter(id => !shipped.has(id)), [], 'Only shipped faces are listed without a preview')
  assert.ok(missing.size < shipped.size / 100, `${missing.size} of ${shipped.size} faces lack a preview`)
  const files = new Set((await readdir('previews', { recursive: true })).filter(f => f.endsWith('.webp')).map(f => `previews/${f}`))
  assert.deepEqual([...files].filter(f => !expected.has(f)), [], 'No picture of a face no longer shipped')
  assert.deepEqual([...expected].filter(f => !files.has(f)), [], 'Every shipped face has its picture')
  const wrong = []
  for (const file of files) { const size = webpSize(await readFile(file)); if (!size || size.height !== 64 || size.width > 1280) wrong.push(`${file} ${JSON.stringify(size)}`) }
  assert.deepEqual(wrong, [], 'Each picture is a lossless WebP 64 pixels high, at most 1,280 wide')
})
