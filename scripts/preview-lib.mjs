// Shared helpers for the raster-preview pilot: PNG inspection, ink regions,
// manifest reconciliation and offline archive validation. No third-party
// dependencies, so the archive can be checked on a machine without network.
import { createHash } from 'node:crypto'
import { inflateSync, deflateSync } from 'node:zlib'
import { readFile, writeFile, readdir, stat } from 'node:fs/promises'
import path from 'node:path'

export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex')

const MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
const CHANNELS = { 0: 1, 2: 3, 4: 2, 6: 4 }

export function pngChunks(buffer) {
  if (!Buffer.isBuffer(buffer) || !buffer.subarray(0, 8).equals(MAGIC)) throw new Error('Not a PNG file')
  const chunks = []
  for (let at = 8; at < buffer.length;) {
    if (at + 12 > buffer.length) throw new Error('Truncated PNG chunk')
    const length = buffer.readUInt32BE(at), type = buffer.toString('ascii', at + 4, at + 8)
    if (at + length + 12 > buffer.length) throw new Error('Truncated PNG chunk data')
    if (crc32(buffer.subarray(at + 4, at + 8 + length)) !== buffer.readUInt32BE(at + 8 + length)) throw new Error('Invalid PNG chunk checksum')
    if ((!chunks.length && type !== 'IHDR') || (type === 'IHDR' && (chunks.length || length !== 13))) throw new Error('Invalid PNG header')
    chunks.push({ type, data: buffer.subarray(at + 8, at + 8 + length) })
    at += length + 12
    if (type === 'IEND') {
      if (length || at !== buffer.length) throw new Error('Invalid PNG final boundary')
      if (!chunks.some(chunk => chunk.type === 'IDAT')) throw new Error('PNG has no image data')
      return chunks
    }
  }
  throw new Error('PNG has no final chunk')
}

/** Read dimensions after checking the container; decodePng also verifies scanlines. */
export function pngSize(buffer) {
  const header = pngChunks(buffer).find(chunk => chunk.type === 'IHDR')
  if (!header) throw new Error('PNG has no IHDR chunk')
  const width = header.data.readUInt32BE(0), height = header.data.readUInt32BE(4)
  if (!width || !height || width * height > 16_777_216 || header.data[10] || header.data[11]) throw new Error('Unsupported PNG dimensions or coding')
  return {
    width, height,
    bitDepth: header.data[8], colorType: header.data[9], interlace: header.data[12],
  }
}

export function decodePng(buffer) {
  const { width, height, bitDepth, colorType, interlace } = pngSize(buffer)
  const channels = CHANNELS[colorType]
  if (bitDepth !== 8 || !channels || interlace !== 0)
    throw new Error(`Unsupported PNG: depth ${bitDepth}, colour type ${colorType}, interlace ${interlace}`)
  const idat = Buffer.concat(pngChunks(buffer).filter(chunk => chunk.type === 'IDAT').map(chunk => chunk.data))
  const stride = width * channels, expected = height * (stride + 1)
  const raw = inflateSync(idat, { maxOutputLength: expected })
  if (raw.length !== expected) throw new Error('Truncated PNG scanlines')
  const pixels = Buffer.alloc(height * stride)
  for (let row = 0; row < height; row++) {
    const filter = raw[row * (stride + 1)], line = raw.subarray(row * (stride + 1) + 1, (row + 1) * (stride + 1))
    const out = pixels.subarray(row * stride, (row + 1) * stride), prior = row ? pixels.subarray((row - 1) * stride, row * stride) : null
    for (let i = 0; i < stride; i++) {
      const left = i >= channels ? out[i - channels] : 0, up = prior ? prior[i] : 0
      const upLeft = prior && i >= channels ? prior[i - channels] : 0
      if (filter === 0) out[i] = line[i]
      else if (filter === 1) out[i] = (line[i] + left) & 0xff
      else if (filter === 2) out[i] = (line[i] + up) & 0xff
      else if (filter === 3) out[i] = (line[i] + ((left + up) >> 1)) & 0xff
      else if (filter === 4) {
        const estimate = left + up - upLeft
        const dl = Math.abs(estimate - left), du = Math.abs(estimate - up), dul = Math.abs(estimate - upLeft)
        out[i] = (line[i] + (dl <= du && dl <= dul ? left : du <= dul ? up : upLeft)) & 0xff
      } else throw new Error(`Unknown PNG filter ${filter} on row ${row}`)
    }
  }
  return { width, height, channels, pixels }
}

export function encodePng({ width, height, channels = 4, pixels }) {
  const stride = width * channels, raw = Buffer.alloc(height * (stride + 1))
  for (let row = 0; row < height; row++)
    pixels.copy ? pixels.copy(raw, row * (stride + 1) + 1, row * stride, (row + 1) * stride)
      : Buffer.from(pixels).copy(raw, row * (stride + 1) + 1, row * stride, (row + 1) * stride)
  const chunk = (type, data) => {
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]), out = Buffer.alloc(body.length + 8)
    out.writeUInt32BE(data.length, 0); body.copy(out, 4)
    out.writeUInt32BE(crc32(body) >>> 0, body.length + 4)
    return out
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8; ihdr[9] = { 1: 0, 2: 4, 3: 2, 4: 6 }[channels]; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0
  return Buffer.concat([MAGIC, chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))])
}

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})
function crc32(buffer) {
  let crc = 0xffffffff
  for (const byte of buffer) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8)
  return (crc ^ 0xffffffff) >>> 0
}

const lumaAt = ({ channels, pixels }, at) => {
  const alpha = channels === 4 || channels === 2 ? pixels[at + channels - 1] : 255
  const [r, g, b] = channels >= 3 ? [pixels[at], pixels[at + 1], pixels[at + 2]] : [pixels[at], pixels[at], pixels[at]]
  const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
  return 255 - alpha * (255 - luma) / 255 // composited on white
}

/** A rectangle of an image, copied out so its own corners define its background. */
export function cropView(image, region) {
  if (!image || ![1, 2, 3, 4].includes(image.channels) || !Number.isInteger(image.width) || !Number.isInteger(image.height) || image.width < 1 || image.height < 1 || image.pixels?.length !== image.width * image.height * image.channels) throw new Error('Invalid source image')
  if (!region || ['x', 'y', 'width', 'height'].some(k => !Number.isInteger(region[k])) || region.x < 0 || region.y < 0 || region.width < 1 || region.height < 1 || region.x + region.width > image.width || region.y + region.height > image.height) throw new Error('Invalid image region')
  const { x, y, width, height } = region
  const pixels = Buffer.alloc(width * height * image.channels), source = Buffer.from(image.pixels)
  for (let row = 0; row < height; row++)
    source.copy(pixels, row * width * image.channels,
      ((y + row) * image.width + x) * image.channels, ((y + row) * image.width + x + width) * image.channels)
  return { width, height, channels: image.channels, pixels }
}

/** Background luma, read from the four corners of the image. */
export function backgroundLuma(image) {
  const { width, height, channels } = image
  const corners = [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]]
    .map(([x, y]) => lumaAt(image, (y * width + x) * channels))
    .sort((a, b) => a - b)
  return { luma: (corners[1] + corners[2]) / 2, spread: corners[3] - corners[0] }
}

/**
 * Bounding box of every pixel that differs from the background, in image pixels.
 * `delta` is small on purpose: faint antialiasing is part of the glyph.
 */
export function inkBounds(image, { delta = 6, within = null } = {}) {
  const { width, height, channels } = image
  const area = within ?? { x: 0, y: 0, width, height }
  const background = backgroundLuma(within ? cropView(image, area) : image).luma
  let minX = area.x + area.width, minY = area.y + area.height, maxX = -1, maxY = -1
  for (let y = area.y; y < area.y + area.height; y++) for (let x = area.x; x < area.x + area.width; x++) {
    if (Math.abs(lumaAt(image, (y * width + x) * channels) - background) <= delta) continue
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  return maxX < 0 ? null : { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 }
}

/** Grow a box by `margin` on all sides, clamped to the image. */
export function padRegion(box, margin, { width, height }) {
  const x = Math.max(0, box.x - margin), y = Math.max(0, box.y - margin)
  return {
    x, y,
    width: Math.min(width, box.x + box.width + margin) - x,
    height: Math.min(height, box.y + box.height + margin) - y,
  }
}

/**
 * Rank specimen lines found on a source page. A readable phrase beats a single
 * word, and a line that only repeats a style label or the family name is kept
 * apart as a last resort rather than passed off as a specimen phrase.
 */
export function rankSpecimenCandidates(candidates, { excludeTexts = [], labels = [] } = {}) {
  if (!Array.isArray(candidates)) throw new TypeError('Expected specimen candidates')
  if ([excludeTexts, labels].some(list => !Array.isArray(list) || list.some(text => text != null && typeof text !== 'string'))) throw new TypeError('Expected specimen text lists')
  const key = value => value.normalize('NFC').toLowerCase().replace(/\s+/gu, ' ').trim()
  const excluded = new Set(excludeTexts.filter(text => text != null).map(key))
  const labelled = new Set(labels.filter(text => text != null).map(key))
  const size = candidate => Number.isFinite(candidate.fontSize) ? Math.min(200, Math.max(0, candidate.fontSize)) : 0
  const score = candidate => {
    const words = candidate.text.trim().split(/\s+/).length
    return (words >= 3 ? 2000 : words === 2 ? 1200 : 0) +
      Math.min(candidate.text.trim().length, 60) * 4 + size(candidate)
  }
  const ordered = candidates
    .filter(candidate => typeof candidate?.text === 'string' && candidate.text.trim() && !excluded.has(key(candidate.text)))
    .map(candidate => ({ ...candidate, score: score(candidate) }))
    .sort((a, b) => b.score - a.score || size(b) - size(a))
  const isLabel = candidate => labelled.has(key(candidate.text))
  return { preferred: ordered.filter(candidate => !isLabel(candidate)), fallback: ordered.filter(isLabel) }
}

/**
 * Without the binary we cannot read a cmap, so compare each face's own lowercase
 * and uppercase lines: a design that maps lowercase onto capitals draws them the
 * same, and its lowercase record must not claim small letters. Sizes are
 * normalised first, because a source may render the two lines at different sizes.
 */
export function flagCapsOnlyFaces(records) {
  const byFace = new Map()
  for (const record of records) {
    const group = byFace.get(record.faceId) ?? {}
    group[record.recipeId] = record
    byFace.set(record.faceId, group)
  }
  const add = (record, flag) => { if (!record.flags.includes(flag)) record.flags.push(flag) }
  for (const group of byFace.values()) {
    const lower = group['latin-lower-v1'], upper = group['latin-upper-v1']
    if (!lower || !upper) continue
    // Faces whose binary was read report their own case mapping; only previews
    // without a font file need this measurement.
    if (lower.verification?.fontSha256 || upper.verification?.fontSha256) continue
    // Measure the ink itself where it was recorded, in ems of the rendered size:
    // the stored region also carries a fixed margin that scales with nothing.
    const ink = record => record.verification?.inkBounds ?? record.region
    const scale = record => (record.render?.cssFontSize || 1) * (record.render?.deviceScaleFactor || 1)
    const widthRatio = (ink(lower).width / scale(lower)) / (ink(upper).width / scale(upper))
    const heightRatio = (ink(lower).height / scale(lower)) / (ink(upper).height / scale(upper))
    if (widthRatio < 0.97 || widthRatio > 1.03 || heightRatio < 0.94 || heightRatio > 1.06) continue
    for (const record of [lower, upper]) add(record, 'lowercase-and-uppercase-render-alike')
    add(lower, 'visible-text-not-verified')
    if (lower.text !== null) {
      lower.verification = { ...lower.verification, requestedText: lower.text }
      lower.text = null // the small letters may never have been drawn
    }
  }
  return records
}

export const isSafeRelativePath = value => typeof value === 'string' && value.length > 0 &&
  !path.isAbsolute(value) && !value.startsWith('/') && !/(^|\/)\.\.(\/|$)/.test(value) && !value.includes('\\')

const SLANTS = new Set(['upright', 'italic', 'oblique', null])
const ACQUISITIONS = new Set(['browser-screenshot', 'source-image', 'provided-image'])
const isInt = value => Number.isInteger(value)
const isText = value => typeof value === 'string' && value.length > 0

/** Structural check of a single manifest record; unknown optional keys are kept. */
export function validateRecord(record) {
  const problems = []
  const fail = message => problems.push(message)
  if (record === null || typeof record !== 'object' || Array.isArray(record)) return ['record is not a JSON object']
  if (record.schemaVersion !== 1) fail(`schemaVersion must be 1, saw ${JSON.stringify(record.schemaVersion)}`)
  for (const key of ['id', 'sourceGroup', 'familyId', 'faceId', 'family', 'sourceUrl', 'recipeId'])
    if (!isText(record[key])) fail(`${key} must be a non-empty string`)
  if (record.role !== 'reference') fail(`role must be "reference", saw ${JSON.stringify(record.role)}`)
  if (!(record.styleName === null || isText(record.styleName))) fail('styleName must be a string or null')
  if (!(record.weight === null || (Number.isFinite(record.weight) && record.weight > 0))) fail('weight must be a positive number or null')
  if (!SLANTS.has(record.slant ?? null)) fail(`slant must be upright, italic, oblique or null, saw ${JSON.stringify(record.slant)}`)
  if (record.axes === null || typeof record.axes !== 'object' || Array.isArray(record.axes)) fail('axes must be an object')
  else for (const [tag, value] of Object.entries(record.axes))
    if (!Number.isFinite(value)) fail(`axis ${tag} must be finite numeric`)
  if (!(record.foundry === null || isText(record.foundry))) fail('foundry must be a string or null')
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/.test(record.capturedAt ?? '')) fail('capturedAt must be an ISO 8601 UTC timestamp')
  if (!(record.text === null || typeof record.text === 'string')) fail('text must be a string or null')
  if (!Array.isArray(record.scripts) || record.scripts.some(tag => !/^[A-Z][a-z]{3}$/.test(tag))) fail('scripts must be an array of ISO 15924 tags')
  if (!ACQUISITIONS.has(record.acquisition)) fail(`acquisition must be one of ${[...ACQUISITIONS].join(', ')}`)
  if (record.render === null || typeof record.render !== 'object' || Array.isArray(record.render)) fail('render must be an object')
  if (!Array.isArray(record.flags)) fail('flags must be an array')
  if (!isSafeRelativePath(record.evidence ?? '')) fail('evidence must be a relative path inside the archive')

  const image = record.image
  if (image === null || typeof image !== 'object' || Array.isArray(image)) fail('image must be an object')
  else {
    if (!isSafeRelativePath(image.path)) fail('image.path must be a relative path inside the archive')
    if (!isInt(image.width) || image.width <= 0) fail('image.width must be a positive integer')
    if (!isInt(image.height) || image.height <= 0) fail('image.height must be a positive integer')
    if (!/^[0-9a-f]{64}$/.test(image.sha256 ?? '')) fail('image.sha256 must be a lowercase hex SHA-256')
  }

  const region = record.region
  if (region === null || typeof region !== 'object' || Array.isArray(region)) fail('region must be an object')
  else {
    for (const key of ['x', 'y']) if (!isInt(region[key]) || region[key] < 0) fail(`region.${key} must be an integer of at least zero`)
    for (const key of ['width', 'height']) if (!isInt(region[key]) || region[key] <= 0) fail(`region.${key} must be a positive integer`)
    if (image && isInt(image.width) && isInt(region.x) && isInt(region.width) && region.x + region.width > image.width)
      fail('region exceeds the image width')
    if (image && isInt(image.height) && isInt(region.y) && isInt(region.height) && region.y + region.height > image.height)
      fail('region exceeds the image height')
  }
  return problems
}

export function parseManifest(text) {
  const records = [], problems = []
  text.split('\n').forEach((line, index) => {
    if (!line.trim()) return
    let record
    try { record = JSON.parse(line) } catch (error) {
      problems.push(`line ${index + 1}: invalid JSON (${error.message})`)
      return
    }
    for (const problem of validateRecord(record)) problems.push(`line ${index + 1}: ${problem}`)
    records.push(record)
  })
  return { records, problems }
}

export const serializeManifest = records => records.map(record => JSON.stringify(record)).join('\n') + (records.length ? '\n' : '')

/** Later records with an existing id replace earlier ones; order stays stable. */
export function mergeRecords(existing, incoming) {
  const byId = new Map(existing.map(record => [record.id, record]))
  let added = 0, replaced = 0
  for (const record of incoming) {
    if (byId.has(record.id)) replaced++
    else added++
    byId.set(record.id, record)
  }
  return { records: [...byId.values()], added, replaced }
}

/**
 * Decide what a run still has to capture. `verify` reports whether a previous
 * record's files are intact; intact captures are reused untouched, so a repeated
 * run neither duplicates records nor rewrites originals.
 */
export function reconcileCaptures(requested, records, verify) {
  const byId = new Map(records.map(record => [record.id, record]))
  const reuse = [], capture = [], repairs = []
  for (const request of requested) {
    const record = byId.get(request.id)
    if (!record) { capture.push(request); continue }
    const verdict = verify(record, request)
    if (verdict.ok) reuse.push(record)
    else { capture.push(request); repairs.push({ id: request.id, reason: verdict.reason }) }
  }
  return { reuse, capture, repairs }
}

/** Full offline audit: schema, identity, files, hashes, dimensions, regions. */
export async function validateArchive(dir) {
  const manifestPath = path.join(dir, 'manifest.jsonl')
  const text = await readFile(manifestPath, 'utf8')
  const { records, problems } = parseManifest(text)
  const errors = [...problems], warnings = []
  if (!records.length) errors.push('Empty manifest')
  const seen = new Map(), hashes = new Map(), groups = new Map()

  for (const record of records) {
    if (record === null || typeof record !== 'object' || Array.isArray(record)) continue
    const label = record?.id ?? '(no id)'
    if (seen.has(record.id)) errors.push(`${label}: duplicate record id`)
    else seen.set(record.id, record)

    for (const [kind, relative] of [['image', record?.image?.path], ['evidence', record?.evidence]]) {
      if (!isSafeRelativePath(relative ?? '')) continue
      try { await stat(path.join(dir, relative)) } catch { errors.push(`${label}: missing ${kind} file ${relative}`) }
    }
    if (!isSafeRelativePath(record?.image?.path ?? '')) continue

    let bytes
    try { bytes = await readFile(path.join(dir, record.image.path)) } catch { continue }
    const digest = sha256(bytes)
    if (digest !== record.image.sha256) errors.push(`${label}: image sha256 mismatch (file ${digest})`)
    try {
      const { width, height } = decodePng(bytes)
      if (width !== record.image.width || height !== record.image.height)
        errors.push(`${label}: decoded size ${width}x${height} does not match the record`)
      if (record.region && Number.isInteger(record.region.x) &&
        (record.region.x + record.region.width > width || record.region.y + record.region.height > height))
        errors.push(`${label}: region falls outside the decoded image`)
    } catch (error) { errors.push(`${label}: unreadable image (${error.message})`) }

    const sharing = hashes.get(digest) ?? []
    sharing.push(record)
    hashes.set(digest, sharing)
    const group = groups.get(record.sourceGroup) ?? new Set()
    group.add(digest)
    groups.set(record.sourceGroup, group)
  }

  for (const [digest, sharing] of hashes) {
    const faces = new Set(sharing.map(record => record.faceId))
    if (faces.size > 1) errors.push(`identical pixels ${digest.slice(0, 12)} shared by different faces: ${[...faces].join(', ')}`)
    const paths = new Set(sharing.map(record => record.image.path))
    if (paths.size > 1) warnings.push(`identical pixels ${digest.slice(0, 12)} stored at ${paths.size} paths`)
  }
  for (const [group, digests] of groups)
    if (digests.size > 1) warnings.push(`sourceGroup ${group} spans ${digests.size} distinct images`)

  const retained = (await readJsonl(path.join(dir, 'attempts.jsonl')).catch(() => [])).flatMap(attempt => attempt.retained ?? [])
  const referenced = new Set([...records.flatMap(record => [record?.image?.path, record?.evidence].filter(Boolean)), ...retained])
  const stray = []
  for (const folder of ['images', 'evidence']) {
    let entries = []
    try { entries = await readdir(path.join(dir, folder)) } catch { continue }
    for (const entry of entries) if (!referenced.has(`${folder}/${entry}`)) stray.push(`${folder}/${entry}`)
  }
  return { records, errors, warnings, stray, manifestSha256: sha256(Buffer.from(text, 'utf8')) }
}

export async function readJsonl(file) {
  let text
  try { text = await readFile(file, 'utf8') } catch { return [] }
  return text.split('\n').filter(line => line.trim()).map(line => JSON.parse(line))
}

export const writeJsonl = (file, rows) => writeFile(file, rows.map(row => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : ''))
