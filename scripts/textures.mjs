// A bank of surface textures for training views of photographed text: Poly Haven's CC0 textures, the 1k diffuse map of
// each, except every texture WhatFontIs-Bench composited its photographs on, so no benchmark background is ever learned.
// Kept in .data/textures with an index naming the source and licence; train/style_data.py draws crops of them.
//
//   node scripts/textures.mjs [--limit 400]
import { readFile, writeFile, mkdir, access } from 'node:fs/promises'

const OUT = '.data/textures', API = 'https://api.polyhaven.com/assets?t=textures', FILE = id => `https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/${id}/${id}_diff_1k.jpg`
const LABELS = '.data/whatfontis/WhatFontIs-Bench/v1/labels.jsonl'
const at = process.argv.indexOf('--limit'), limit = at < 0 ? 400 : Number(process.argv[at + 1])
const exists = path => access(path).then(() => true, () => false)

await mkdir(OUT, { recursive: true })
const used = new Set((await readFile(LABELS, 'utf8')).trim().split('\n').map(line => JSON.parse(line).texture).filter(Boolean))
const assets = await (await fetch(API)).json()
// Most downloaded first: the textures people photograph text on (walls, wood, metal, paper) are the popular ones.
const ids = Object.entries(assets).filter(([id]) => !used.has(id)).sort((a, b) => b[1].download_count - a[1].download_count).map(([id]) => id).slice(0, limit)
const index = { source: 'Poly Haven', licence: 'CC0 1.0', url: 'https://polyhaven.com/textures', map: 'diffuse, 1k JPEG', excluded: [...used].sort(), textures: [] }
let queue = [...ids], fetched = 0, missing = 0
await Promise.all(Array.from({ length: 4 }, async () => {
  for (let id; (id = queue.shift());) {
    const path = `${OUT}/${id}.jpg`
    if (!await exists(path)) {
      const response = await fetch(FILE(id))
      if (!response.ok) { missing++; continue }
      await writeFile(path, new Uint8Array(await response.arrayBuffer()))
    }
    index.textures.push({ id, name: assets[id].name, categories: assets[id].categories, file: `${id}.jpg` }); fetched++
    if (fetched % 50 === 0) console.log(`${fetched} textures`)
  }
}))
index.textures.sort((a, b) => a.id.localeCompare(b.id))
await writeFile(`${OUT}/index.json`, JSON.stringify(index, null, 1) + '\n')
console.log(`${fetched} textures in ${OUT}, ${missing} without a 1k diffuse map, ${used.size} of the benchmark's excluded`)
