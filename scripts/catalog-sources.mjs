import { readCatalog } from '../src/catalog.mjs'

// Combine reference batches by their documented source, preserving genuine faces.
export function sourceCatalogs(inputs, binding) {
  const groups = new Map(), names = { 'google-fonts': 'Google Fonts', fontshare: 'Fontshare', myfonts: 'MyFonts', velvetyne: 'Velvetyne' }
  for (const { catalog, records } of inputs) {
    readCatalog(catalog, binding)
    const byId = new Map(records.map(r => [r.id, r])), raw = Buffer.from(catalog.vectors.data, 'base64')
    for (const [owner, face] of catalog.faces.entries()) {
      const references = face.referenceIds?.map(id => byId.get(id))
      if (!references?.length || references.some(r => !r || r.faceId !== face.id || !/^[a-z][a-z0-9-]*$/.test(r.source?.key))) throw new Error('Missing reference source')
      const source = references[0].source
      if (references.some(r => r.source.key !== source.key)) throw new Error('Conflicting reference sources')
      if (!groups.has(source.key)) groups.set(source.key, { name: names[source.key] || source.name, faces: new Map(), manifests: new Set() })
      const group = groups.get(source.key), rows = catalog.vectors.owners.flatMap((o, i) => o === owner ? [i] : [])
      const entry = { face, data: Buffer.concat(rows.map(i => raw.subarray(i * 128, (i + 1) * 128))).toString('base64'), scales: rows.map(i => catalog.vectors.scales[i]) }
      if (group.faces.has(face.id) && JSON.stringify(group.faces.get(face.id)) !== JSON.stringify(entry)) throw new Error('Conflicting source face: ' + face.id)
      group.faces.set(face.id, entry); group.manifests.add(catalog.sourceManifestSha256)
    }
  }
  return [...groups].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([id, group]) => {
    const faces = [], chunks = [], scales = [], owners = []
    for (const [owner, [, entry]] of [...group.faces].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).entries()) {
      faces.push(entry.face); chunks.push(Buffer.from(entry.data, 'base64')); scales.push(...entry.scales); owners.push(...entry.scales.map(() => owner))
    }
    const data = { version: 3, kind: 'font-catalog', ...binding, dimensions: 128, source: id, sourceManifests: [...group.manifests].sort(), faces,
      vectors: { encoding: 'int8-base64', shape: [owners.length, 128], data: Buffer.concat(chunks).toString('base64'), scales, owners } }
    readCatalog(data, binding)
    return { id, name: group.name, data }
  })
}
