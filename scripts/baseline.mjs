import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { descriptor } from '../src/descriptor.mjs'
import { rank, unit } from '../src/rank.mjs'

const manifestBytes = await readFile('.data/prepared.json')
const manifest = JSON.parse(manifestBytes)
const binary = await readFile('.data/prepared.f32')
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
for (const [key, path] of [['configSha256', 'bench/pilot.json'], ['fontsSha256', 'bench/fonts.json'], ['preparationSha256', 'src/prepare.mjs']]) {
  if (hash(await readFile(path)) !== manifest[key]) throw new Error(`Stale preparation: ${path}`)
}
if (hash(binary) !== manifest.tensorSha256) throw new Error('Prepared tensor checksum failed')
if (!manifest.samples.some(s => s.role === 'prototype') || !manifest.samples.some(s => s.role === 'query')) throw new Error('Expected prototype and query samples')
const count = manifest.width * manifest.height
if (binary.length !== count * 4 * manifest.samples.length) throw new Error('Prepared tensor length mismatch')
const reports = {}
for (const mode of ['grayscale', 'binary']) {
  const vectors = manifest.samples.map((sample, i) => {
    if (sample.role === 'train') return null
    const pixels = Float32Array.from({ length: count }, (_, p) => {
      const value = binary.readFloatLE((i * count + p) * 4)
      return mode === 'binary' ? +(value >= .5) : value
    })
    return descriptor(pixels, manifest.width, manifest.height)
  })
  const means = new Map()
  manifest.samples.forEach((sample, i) => {
    if (sample.role !== 'prototype') return
    if (!means.has(sample.family)) means.set(sample.family, new Float32Array(vectors[i].length))
    const mean = means.get(sample.family)
    for (let j = 0; j < mean.length; j++) mean[j] += vectors[i][j]
  })
  const catalog = [...means].map(([family, vector]) => ({ family, vector: unit(vector) }))
  const groups = {}
  const errors = new Map()
  manifest.samples.forEach((sample, i) => {
    if (sample.role !== 'query') return
    const matches = rank(vectors[i], catalog)
    const key = `${sample.renderer}/${sample.split}`
    const group = groups[key] ??= { queries: 0, top1: 0, top5: 0, families: {} }
    const family = group.families[sample.family] ??= { queries: 0, top1: 0, top5: 0 }
    for (const stats of [group, family]) {
      stats.queries++
      stats.top1 += +(matches[0].family === sample.family)
      stats.top5 += +matches.slice(0, 5).some(m => m.family === sample.family)
    }
    if (matches[0].family !== sample.family) {
      const pair = `${sample.family} → ${matches[0].family}`
      errors.set(pair, (errors.get(pair) ?? 0) + 1)
    }
  })
  for (const group of Object.values(groups)) {
    for (const family of Object.values(group.families)) {
      family.top1 /= family.queries; family.top5 /= family.queries
    }
    // Macro-average families, independent of unequal query counts.
    const families = Object.values(group.families)
    group.top1 = families.reduce((s, f) => s + f.top1, 0) / families.length
    group.top5 = families.reduce((s, f) => s + f.top5, 0) / families.length
  }
  reports[mode] = { catalogFamilies: catalog.length, groups,
    commonErrors: [...errors].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([pair, queries]) => ({ pair, queries })) }
}
const catalogSize = reports.grayscale.catalogFamilies
const report = { description: `Synthetic development pilot; ${catalogSize}-way retrieval, disjoint prototype/query text. No independent real screenshot claim.`,
  manifestSha256: hash(manifestBytes), tensorSha256: manifest.tensorSha256, preparationMs: manifest.preparationMs,
  chance: { top1: 1 / catalogSize, top5: Math.min(5, catalogSize) / catalogSize }, reports }
await writeFile('bench/baseline.json', JSON.stringify(report, null, 2) + '\n')
for (const [mode, result] of Object.entries(reports)) for (const [group, stats] of Object.entries(result.groups)) {
  console.log(`${mode} ${group}: top1 ${(stats.top1 * 100).toFixed(1)}%, top5 ${(stats.top5 * 100).toFixed(1)}% (${stats.queries} queries)`)
}
