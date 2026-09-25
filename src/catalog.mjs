// Catalog vectors are tied to exact encoder bytes and a canonical preparation hash.
export const sha256 = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value
export const preparationHash = preparation => sha256(new TextEncoder().encode(JSON.stringify(canonical(preparation))))
// Padded standard base64; one character class scans megabytes linearly where grouped quantifiers crawl.
export const base64 = text => text.length % 4 === 0 && /^[A-Za-z0-9+/]*={0,2}$/.test(text)

// Signed integers `bits` apiece (2 to 8) in a little-endian bit stream: value i fills bits i * bits onward. At 8 bits
// these are plain int8 bytes. Both directions stay in plain loops: they run over millions of weights.
export function pack(values, bits) {
  const bytes = new Uint8Array(Math.ceil(values.length * bits / 8)), mask = (1 << bits) - 1
  for (let i = 0, bit = 0; i < values.length; i++, bit += bits) {
    const word = (values[i] & mask) << (bit & 7)
    bytes[bit >> 3] |= word
    if (word > 255) bytes[(bit >> 3) + 1] |= word >> 8
  }
  return bytes
}
export function unpack(bytes, bits, count) {
  const values = new Int8Array(count), mask = (1 << bits) - 1, sign = 1 << (bits - 1)
  for (let i = 0, bit = 0; i < count; i++, bit += bits) {
    const value = ((bytes[bit >> 3] | bytes[(bit >> 3) + 1] << 8) >> (bit & 7)) & mask
    values[i] = value >= sign ? value - (1 << bits) : value
  }
  return values
}
// Catalog vectors: `dimensions` numbers a row (the encoder's output, 128 or 64), one scale per row, 8 or 4 bits a number.
const VECTOR_BITS = { 'int8-base64': 8, 'int4-base64': 4 }
// An embedding has a multiple of 16 numbers, 16 to 1024: what a style link or a catalog row can be read as.
export const validDimensions = d => Number.isInteger(d) && d >= 16 && d <= 1024 && d % 16 === 0
export function rowBytes(vectors, dimensions) {
  if (!VECTOR_BITS[vectors?.encoding]) throw new Error('Unknown vector encoding')
  if (!validDimensions(dimensions)) throw new Error('Invalid embedding dimensions')
  return dimensions * VECTOR_BITS[vectors.encoding] / 8
}

export function unit(values) {
  if (!values?.length || !Array.from(values).every(Number.isFinite)) throw new Error('Invalid embedding')
  const norm = Math.hypot(...values)
  if (norm <= 1e-12) throw new Error('Zero embedding')
  return Float32Array.from(values, v => v / norm)
}

export function embedWindows(projections) {
  if (!Array.isArray(projections) || !projections.length || projections.length > 3) throw new Error('Expected one to three projections')
  if (!validDimensions(projections[0]?.length)) throw new Error('Invalid embedding dimensions')
  const sum = new Float32Array(projections[0].length)
  for (const p of projections) {
    if (p?.length !== sum.length) throw new Error('Invalid embedding dimensions')
    const vector = unit(p)
    for (let i = 0; i < sum.length; i++) sum[i] += vector[i]
  }
  return unit(sum)
}

export function readCatalog(data, binding) {
  if (!data || ![1, 2, 3].includes(data.version) || data.kind !== 'font-catalog' || !validDimensions(data.dimensions) || (data.version === 3 ? data.referencesPerFace !== undefined : ![1, 4].includes(data.referencesPerFace))) throw new Error('Invalid font catalog')
  if (!binding?.encoderSha256 || !binding?.preparationSha256 || data.encoderSha256 !== binding.encoderSha256 || data.preparationSha256 !== binding.preparationSha256) throw new Error('Catalog was built with a different model: rebuild it with this one')
  const faces = data.faces, v = data.vectors, dimensions = data.dimensions
  if (!Array.isArray(faces) || !faces.length || faces.length > 100000 || faces.some(f => !f || ['id', 'familyId', 'family'].some(k => typeof f[k] !== 'string' || !f[k])) || new Set(faces.map(f => f.id)).size !== faces.length) throw new Error('Expected unique catalog faces')
  const names = new Map()
  for (const f of faces) {
    if (names.has(f.familyId) && (data.version === 1 || names.get(f.familyId) !== f.family)) throw new Error('Conflicting catalog families')
    names.set(f.familyId, f.family)
  }
  const rows = data.version === 3 ? v?.shape?.[0] : faces.length * data.referencesPerFace
  if (!Number.isInteger(rows) || rows < faces.length || rows > 640000) throw new Error('Invalid reference count')
  if (!VECTOR_BITS[v?.encoding] || JSON.stringify(v.shape) !== JSON.stringify([rows, dimensions])) throw new Error('Invalid vector shape')
  const size = rows * rowBytes(v, dimensions)
  if (typeof v.data !== 'string' || v.data.length > Math.ceil(size / 3) * 4 || !base64(v.data)) throw new Error('Invalid vector bytes')
  const binary = atob(v.data), bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  if (bytes.length !== size) throw new Error('Truncated or trailing vectors')
  if (!Array.isArray(v.scales) || v.scales.length !== rows || v.scales.some(s => !Number.isFinite(s) || s <= 0)) throw new Error('Invalid vector scales')
  if (!Array.isArray(v.owners) || v.owners.length !== rows || v.owners.some(i => !Number.isInteger(i) || i < 0 || i >= faces.length)) throw new Error('Invalid vector owners')
  const counts = new Uint32Array(faces.length)
  v.owners.forEach(i => counts[i]++)
  if (counts.some(n => data.version === 3 ? n < 1 : n !== data.referencesPerFace)) throw new Error('Missing catalog owner')
  // Dequantize and L2-normalize each row in place; plain loops keep per-face catalogs (tens of thousands of rows) fast.
  const packed = unpack(bytes, VECTOR_BITS[v.encoding], rows * dimensions), vectors = new Float32Array(rows * dimensions)
  for (let r = 0, o = 0; r < rows; r++, o += dimensions) {
    let norm = 0
    for (let d = 0; d < dimensions; d++) { const value = Math.fround(packed[o + d] * v.scales[r]); vectors[o + d] = value; norm += value * value }
    if (!Number.isFinite(norm)) throw new Error('Invalid embedding')  // float32 overflow, as stored
    if (!(norm > 1e-24)) throw new Error('Zero embedding')
    norm = Math.sqrt(norm)
    for (let d = 0; d < dimensions; d++) vectors[o + d] /= norm
  }
  return { faces: faces.map(f => ({ ...f })), vectors, owners: [...v.owners], families: names.size, dimensions }
}

// Read catalogs searched as one: their faces and rows in order, each row's owner shifted past the faces before it.
export function unionCatalogs(catalogs) {
  const dimensions = catalogs[0]?.dimensions
  if (catalogs.some(c => c.dimensions !== dimensions)) throw new Error('Catalogs of different embedding dimensions')
  const faces = catalogs.flatMap(c => c.faces), vectors = new Float32Array(catalogs.reduce((n, c) => n + c.vectors.length, 0)), owners = []
  let row = 0, base = 0
  for (const c of catalogs) { vectors.set(c.vectors, row); row += c.vectors.length; for (const owner of c.owners) owners.push(owner + base); base += c.faces.length }
  return { faces, vectors, owners, families: new Set(faces.map(f => f.familyId)).size, dimensions }
}

// A read catalog cut to the faces `keep` accepts, with their rows: one source of a grouped catalog, searched alone.
export function subsetCatalog(catalog, keep) {
  const index = new Map(), faces = [], rows = []
  catalog.faces.forEach((face, i) => { if (keep(face)) { index.set(i, faces.length); faces.push(face) } })
  catalog.owners.forEach((owner, row) => { if (index.has(owner)) rows.push(row) })
  const d = catalog.dimensions, vectors = new Float32Array(rows.length * d)
  rows.forEach((row, i) => vectors.set(catalog.vectors.subarray(row * d, row * d + d), i * d))
  return { faces, vectors, owners: rows.map(row => index.get(catalog.owners[row])), families: new Set(faces.map(f => f.familyId)).size, dimensions: d }
}

// A style as URL-safe text: the first 8 hex digits of the model that read it, a dot, then its values as signed bytes in
// base64url (180 characters for 128 numbers). A cosine search reads only direction, so the vector is scaled to fill int8.
export function writeStyle(embedding, modelSha256) {
  if (!validDimensions(embedding?.length)) throw new Error('Invalid embedding dimensions')
  const v = unit(embedding), top = Math.max(...v.map(Math.abs)), bytes = Int8Array.from(v, x => Math.round(x / top * 127))
  return `${modelSha256.slice(0, 8)}.${btoa(String.fromCharCode(...new Uint8Array(bytes.buffer))).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '')}`
}
// { model, embedding }: the model's 8 hex digits, for the caller to compare with its own, and the unit vector.
export function readStyle(text) {
  const [, model, data] = /^([0-9a-f]{8})\.([\w-]{22,1366})$/.exec(text ?? '') ?? []
  if (!data) throw new Error('Invalid style')
  const bytes = Int8Array.from(atob(data.replaceAll('-', '+').replaceAll('_', '/') + '='.repeat((4 - data.length % 4) % 4)), c => c.charCodeAt(0))
  if (!validDimensions(bytes.length)) throw new Error('Invalid style')
  return { model, embedding: unit(bytes) }
}

// Optional typed verdicts shipped with the encoder: weight, italic, script, Google category and fine style class.
export function readHeads(artifact) {
  const h = artifact?.heads, dimensions = artifact?.dimensions
  if (h === undefined) return null
  if (!validDimensions(dimensions)) throw new Error('Invalid embedding dimensions')
  const layer = (value, rows, name) => {
    if (!Array.isArray(value?.weights) || value.weights.length !== rows || value.weights.some(r => !Array.isArray(r) || r.length !== dimensions || !r.every(Number.isFinite)) || !Array.isArray(value.bias) || value.bias.length !== rows || !value.bias.every(Number.isFinite)) throw new Error(`Invalid ${name} head`)
    return { weights: value.weights.map(r => Float32Array.from(r)), bias: Float32Array.from(value.bias) }
  }
  const labelled = (value, name) => {
    if (!Array.isArray(value?.labels) || !value.labels.length || value.labels.some(l => typeof l !== 'string' || !l) || new Set(value.labels).size !== value.labels.length) throw new Error(`Invalid ${name} labels`)
    return { ...layer(value, value.labels.length, name), labels: [...value.labels] }
  }
  if (!Number.isFinite(h?.weight?.scale) || !Number.isFinite(h.weight.offset)) throw new Error('Invalid weight head')
  return { weight: { ...layer(h.weight, 1, 'weight'), scale: h.weight.scale, offset: h.weight.offset }, italic: layer(h.italic, 1, 'italic'),
    script: labelled(h.script, 'script'), category: labelled(h.category, 'category'), fine: labelled(h.fine, 'fine') }
}

const affine = (layer, x) => layer.weights.map((w, i) => w.reduce((sum, v, d) => sum + v * x[d], layer.bias[i]))
const sigmoid = v => 1 / (1 + Math.exp(-v))
const softmax = values => { const top = Math.max(...values), e = values.map(v => Math.exp(v - top)), total = e.reduce((a, b) => a + b); return e.map(v => v / total) }
const ranked = (labels, p) => labels.map((label, i) => ({ label, p: p[i] })).sort((a, b) => b.p - a.p || (a.label < b.label ? -1 : 1))

export function verdict(embedding, heads) {
  const x = unit(embedding)
  return { weight: affine(heads.weight, x)[0] * heads.weight.scale + heads.weight.offset, italic: sigmoid(affine(heads.italic, x)[0]),
    script: ranked(heads.script.labels, softmax(affine(heads.script, x))), category: ranked(heads.category.labels, softmax(affine(heads.category, x))),
    fine: ranked(heads.fine.labels, affine(heads.fine, x).map(sigmoid)) }
}

// One row per family: a family whose letters in the query's script are indistinguishable from a higher-ranked one's (the
// catalog's per-script twins) folds into it when one name holds the other's first word (IBM Plex Sans KR under IBM Plex Sans,
// Ek Mukta with Mukta), never by chains. Parastoo, which borrows Lora's Latin letters, is a font of its own and keeps its row.
// A group is named by the member whose name sits inside the most others and scored by its best member, as a family is by its
// best face. Without a script nothing folds. `limit` bounds the groups; later families still join kept ones.
const twinIndex = new WeakMap()
export function foldTwins(matches, catalog, { script, limit = Infinity } = {}) {
  if (!twinIndex.has(catalog)) twinIndex.set(catalog, new Map())
  const index = twinIndex.get(catalog)
  if (!index.has(script)) index.set(script, new Map(catalog.faces.filter(f => Array.isArray(f.twins?.[script])).map(f => [f.familyId, new Set(f.twins[script])])))
  const twins = index.get(script), words = m => m.face.family.split(' ')
  const named = (a, b) => words(b).includes(words(a)[0]) || words(a).includes(words(b)[0])
  const related = (a, b) => (twins.get(a.family)?.has(b.family) || twins.get(b.family)?.has(a.family)) && named(a, b), groups = []
  for (const match of matches) {
    const group = groups.find(g => related(g[0], match))
    if (group) group.push(match)
    else if (groups.length < limit) groups.push([match])
  }
  return groups.map(group => {
    const padded = m => ` ${m.face.family} `, within = m => group.filter(o => padded(o).includes(padded(m))).length
    const shown = group.reduce((best, m) => within(m) > within(best) ? m : best)
    return { ...shown, score: group[0].score, siblings: group.filter(m => m !== shown).map(m => m.face.family) }
  })
}

// Script-aware search: skip fonts that cannot draw the detected script; when the catalog has none, match style across scripts.
export function matchCatalog(embedding, catalog, judged = null) {
  const script = judged?.script?.[0]?.label, matches = script ? rankCatalog(embedding, catalog, { script }) : []
  return matches.length ? matches : rankCatalog(embedding, catalog)
}

// `script` skips faces that cannot draw it; faces without declared coverage stay eligible.
export function rankCatalog(embedding, catalog, { script } = {}) {
  const d = catalog.dimensions
  if (embedding?.length !== d) throw new Error('Invalid embedding dimensions')
  const query = unit(embedding), best = new Map()
  for (let row = 0; row < catalog.owners.length; row++) {
    const face = catalog.faces[catalog.owners[row]]
    if (script && Array.isArray(face.scripts) && !face.scripts.includes(script)) continue
    let score = 0
    for (let i = 0; i < d; i++) score += query[i] * catalog.vectors[row * d + i]
    score = Math.max(-1, Math.min(1, score))
    const previous = best.get(face.familyId)
    if (!previous || score > previous.score || (score === previous.score && face.id < previous.face.id)) best.set(face.familyId, { family: face.familyId, score, face })
  }
  return [...best.values()].sort((a, b) => b.score - a.score || (a.family < b.family ? -1 : a.family > b.family ? 1 : 0))
}
