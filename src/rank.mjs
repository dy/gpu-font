export function unit(vector) {
  const norm = Math.hypot(...vector)
  if (!norm || !Number.isFinite(norm)) throw new RangeError('Expected a finite nonzero vector')
  return Float32Array.from(vector, x => x / norm)
}

export function rank(query, catalog) {
  const q = unit(query)
  return catalog.map(({ family, vector }) => {
    if (vector.length !== q.length) throw new RangeError('Vector dimensions differ')
    const v = unit(vector)
    return { family, score: q.reduce((sum, x, i) => sum + x * v[i], 0) }
  }).sort((a, b) => b.score - a.score || a.family.localeCompare(b.family))
}
