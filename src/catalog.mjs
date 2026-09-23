// Catalog vectors are tied to exact encoder bytes and a canonical preparation hash.
export const sha256 = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value
export const preparationHash = preparation => sha256(new TextEncoder().encode(JSON.stringify(canonical(preparation))))

export function unit(values) {
  if (!values?.length || !Array.from(values).every(Number.isFinite)) throw new Error('Invalid embedding')
  const norm = Math.hypot(...values)
  if (norm <= 1e-12) throw new Error('Zero embedding')
  return Float32Array.from(values, v => v / norm)
}

export function embedWindows(projections) {
  if (!Array.isArray(projections) || !projections.length || projections.length > 3) throw new Error('Expected one to three projections')
  const sum = new Float32Array(128)
  for (const p of projections) {
    if (p?.length !== sum.length) throw new Error('Invalid embedding dimensions')
    const vector = unit(p)
    for (let i = 0; i < sum.length; i++) sum[i] += vector[i]
  }
  return unit(sum)
}

export function readCatalog(data, binding) {
  if (!data || ![1, 2, 3].includes(data.version) || data.kind !== 'font-catalog' || data.dimensions !== 128 || (data.version === 3 ? data.referencesPerFace !== undefined : ![1, 4].includes(data.referencesPerFace))) throw new Error('Invalid font catalog')
  if (!binding?.encoderSha256 || !binding?.preparationSha256 || data.encoderSha256 !== binding.encoderSha256 || data.preparationSha256 !== binding.preparationSha256) throw new Error('Catalog uses a different encoder or preparation')
  const faces = data.faces, v = data.vectors
  if (!Array.isArray(faces) || !faces.length || faces.length > 10000 || faces.some(f => !f || ['id', 'familyId', 'family'].some(k => typeof f[k] !== 'string' || !f[k])) || new Set(faces.map(f => f.id)).size !== faces.length) throw new Error('Expected unique catalog faces')
  const names = new Map()
  for (const f of faces) {
    if (names.has(f.familyId) && (data.version === 1 || names.get(f.familyId) !== f.family)) throw new Error('Conflicting catalog families')
    names.set(f.familyId, f.family)
  }
  const rows = data.version === 3 ? v?.shape?.[0] : faces.length * data.referencesPerFace
  if (!Number.isInteger(rows) || rows < faces.length || rows > 640000) throw new Error('Invalid reference count')
  if (v?.encoding !== 'int8-base64' || JSON.stringify(v.shape) !== JSON.stringify([rows, 128])) throw new Error('Invalid vector shape')
  if (typeof v.data !== 'string' || v.data.length > Math.ceil(rows * 128 / 3) * 4 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(v.data)) throw new Error('Invalid vector bytes')
  const binary = atob(v.data), bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  if (bytes.length !== rows * 128) throw new Error('Truncated or trailing vectors')
  if (!Array.isArray(v.scales) || v.scales.length !== rows || v.scales.some(s => !Number.isFinite(s) || s <= 0)) throw new Error('Invalid vector scales')
  if (!Array.isArray(v.owners) || v.owners.length !== rows || v.owners.some(i => !Number.isInteger(i) || i < 0 || i >= faces.length)) throw new Error('Invalid vector owners')
  const counts = new Uint32Array(faces.length)
  v.owners.forEach(i => counts[i]++)
  if (counts.some(n => data.version === 3 ? n < 1 : n !== data.referencesPerFace)) throw new Error('Missing catalog owner')
  // Dequantize and L2-normalize each row in place; plain loops keep per-face catalogs (tens of thousands of rows) fast.
  const packed = new Int8Array(bytes.buffer), vectors = new Float32Array(bytes.length)
  for (let r = 0, o = 0; r < rows; r++, o += 128) {
    let norm = 0
    for (let d = 0; d < 128; d++) { const value = Math.fround(packed[o + d] * v.scales[r]); vectors[o + d] = value; norm += value * value }
    if (!Number.isFinite(norm)) throw new Error('Invalid embedding')  // float32 overflow, as stored
    if (!(norm > 1e-24)) throw new Error('Zero embedding')
    norm = Math.sqrt(norm)
    for (let d = 0; d < 128; d++) vectors[o + d] /= norm
  }
  return { faces: faces.map(f => ({ ...f })), vectors, owners: [...v.owners], families: names.size }
}

// Optional typed verdicts shipped with the encoder: weight, italic, script, Google category and fine style class.
export function readHeads(artifact) {
  const h = artifact?.heads
  if (h === undefined) return null
  const layer = (value, rows, name) => {
    if (!Array.isArray(value?.weights) || value.weights.length !== rows || value.weights.some(r => !Array.isArray(r) || r.length !== 128 || !r.every(Number.isFinite)) || !Array.isArray(value.bias) || value.bias.length !== rows || !value.bias.every(Number.isFinite)) throw new Error(`Invalid ${name} head`)
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

// One row per design: a family whose letters in the query's script are indistinguishable from a higher-ranked one's (the
// catalog's per-script twins) folds into it, never by chains. A group is named by the member whose name begins the most
// others (IBM Plex Sans, not its Thai-derived twin Anuphan) and scored by its best member, as a family is by its best face.
// Without a script nothing folds. `limit` bounds the groups; later families still join kept ones.
const twinIndex = new WeakMap()
export function foldTwins(matches, catalog, { script, limit = Infinity } = {}) {
  if (!twinIndex.has(catalog)) twinIndex.set(catalog, new Map())
  const index = twinIndex.get(catalog)
  if (!index.has(script)) index.set(script, new Map(catalog.faces.filter(f => Array.isArray(f.twins?.[script])).map(f => [f.familyId, new Set(f.twins[script])])))
  const twins = index.get(script), related = (a, b) => twins.get(a)?.has(b) || twins.get(b)?.has(a), groups = []
  for (const match of matches) {
    const group = groups.find(g => related(g[0].family, match.family))
    if (group) group.push(match)
    else if (groups.length < limit) groups.push([match])
  }
  return groups.map(group => {
    const name = m => m.face.family, begins = m => group.filter(o => name(o).startsWith(name(m) + ' ')).length
    const shown = group.reduce((best, m) => begins(m) > begins(best) ? m : best)
    return { ...shown, score: group[0].score, siblings: group.filter(m => m !== shown).map(name) }
  })
}

// Script-aware search: skip fonts that cannot draw the detected script; when the catalog has none, match style across scripts.
export function matchCatalog(embedding, catalog, judged = null) {
  const script = judged?.script?.[0]?.label, matches = script ? rankCatalog(embedding, catalog, { script }) : []
  return matches.length ? matches : rankCatalog(embedding, catalog)
}

// `script` skips faces that cannot draw it; faces without declared coverage stay eligible.
export function rankCatalog(embedding, catalog, { script } = {}) {
  if (embedding?.length !== 128) throw new Error('Invalid embedding dimensions')
  const query = unit(embedding), best = new Map()
  for (let row = 0; row < catalog.owners.length; row++) {
    const face = catalog.faces[catalog.owners[row]]
    if (script && Array.isArray(face.scripts) && !face.scripts.includes(script)) continue
    let score = 0
    for (let d = 0; d < 128; d++) score += query[d] * catalog.vectors[row * 128 + d]
    score = Math.max(-1, Math.min(1, score))
    const previous = best.get(face.familyId)
    if (!previous || score > previous.score || (score === previous.score && face.id < previous.face.id)) best.set(face.familyId, { family: face.familyId, score, face })
  }
  return [...best.values()].sort((a, b) => b.score - a.score || (a.family < b.family ? -1 : a.family > b.family ? 1 : 0))
}
