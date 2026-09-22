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
  const bytes = Uint8Array.from(atob(v.data), c => c.charCodeAt(0))
  if (bytes.length !== rows * 128) throw new Error('Truncated or trailing vectors')
  if (!Array.isArray(v.scales) || v.scales.length !== rows || v.scales.some(s => !Number.isFinite(s) || s <= 0)) throw new Error('Invalid vector scales')
  if (!Array.isArray(v.owners) || v.owners.length !== rows || v.owners.some(i => !Number.isInteger(i) || i < 0 || i >= faces.length)) throw new Error('Invalid vector owners')
  const counts = new Uint32Array(faces.length)
  v.owners.forEach(i => counts[i]++)
  if (counts.some(n => data.version === 3 ? n < 1 : n !== data.referencesPerFace)) throw new Error('Missing catalog owner')
  const packed = new Int8Array(bytes.buffer), vectors = new Float32Array(bytes.length)
  for (let r = 0; r < rows; r++) vectors.set(unit(Float32Array.from(packed.subarray(r * 128, (r + 1) * 128), n => n * v.scales[r])), r * 128)
  return { faces: faces.map(f => ({ ...f })), vectors, owners: [...v.owners], families: names.size }
}

export function rankCatalog(embedding, catalog) {
  if (embedding?.length !== 128) throw new Error('Invalid embedding dimensions')
  const query = unit(embedding), best = new Map()
  for (let row = 0; row < catalog.owners.length; row++) {
    const face = catalog.faces[catalog.owners[row]]
    let score = 0
    for (let d = 0; d < 128; d++) score += query[d] * catalog.vectors[row * 128 + d]
    score = Math.max(-1, Math.min(1, score))
    const previous = best.get(face.familyId)
    if (!previous || score > previous.score || (score === previous.score && face.id < previous.face.id)) best.set(face.familyId, { family: face.familyId, score, face })
  }
  return [...best.values()].sort((a, b) => b.score - a.score || (a.family < b.family ? -1 : a.family > b.family ? 1 : 0))
}
