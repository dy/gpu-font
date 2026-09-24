// From an image of text to ranked fonts in two calls: load a model and a catalog, then match RGBA crops.
import { readNetwork, inferCPU } from './network.mjs'
import { createNetworkGPU } from './network-gpu.mjs'
import { prepareLine } from './line.mjs'
import { readCatalog, embedWindows, matchCatalog, foldTwins, sha256, preparationHash, readHeads, verdict } from './catalog.mjs'

// The shipped model and catalogs, resolved next to this module wherever it is served from. Each URL is a literal, so
// bundlers copy the file.
const MODEL = new URL('../models/encoder/encoder.json', import.meta.url)
export const catalogs = {
  'google-fonts': new URL('../models/encoder/google-fonts.json', import.meta.url),
  debian: new URL('../models/encoder/catalogs/debian.json', import.meta.url),
  other: new URL('../models/encoder/catalogs/other.json', import.meta.url)
}

// A file's bytes. Node's fetch cannot open file: URLs, so Node reads those from disk.
async function load(url) {
  const fs = String(url).startsWith('file:') && globalThis.process?.getBuiltinModule?.('node:fs/promises')
  if (fs) return fs.readFile(new URL(url))
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Cannot load ${url}: HTTP ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
}

// The model on WebGPU, or on the CPU without it; `project` returns one projection per prepared window.
export async function createEncoder(modelUrl = MODEL) {
  const bytes = await load(modelUrl)
  const artifact = JSON.parse(new TextDecoder().decode(bytes)), model = readNetwork(artifact)
  // A catalog only fits the exact model and preparation that indexed it.
  const binding = { encoderSha256: await sha256(bytes), preparationSha256: await preparationHash(artifact.preparation) }
  let gpu = await createNetworkGPU(model).catch(() => null)
  return {
    model, binding, heads: readHeads(artifact), preparation: model.preparation,
    async project(windows) {
      if (gpu) try { return await Promise.all(windows.map(w => gpu.infer(w))) } catch { gpu.destroy(); gpu = null } // A lost device falls back to the CPU.
      return windows.map(w => inferCPU(model, w))
    },
    destroy() { gpu?.destroy(); gpu = null }
  }
}

// The catalog comes first: it is what callers choose, and the model it was built with rarely changes.
export async function createMatcher(catalogUrl = catalogs['google-fonts'], modelUrl = MODEL) {
  // The catalog downloads while the model loads; it is read once the model's hashes are known.
  const data = load(catalogUrl).then(bytes => JSON.parse(new TextDecoder().decode(bytes)))
  data.catch(() => {})
  const encoder = await createEncoder(modelUrl)
  let catalog
  try { catalog = readCatalog(await data, encoder.binding) } catch (error) { encoder.destroy(); throw error }
  return {
    // Fonts best first, identical designs folded into one entry with their `siblings`; [] when the crop holds no text.
    async match(imageData) {
      const { status, windows } = prepareLine(imageData, { ...encoder.preparation, deskew: true, sampler: 'windows' })
      if (status !== 'ok') return []
      const embedding = embedWindows(await encoder.project(windows)), judged = encoder.heads ? verdict(embedding, encoder.heads) : null
      return foldTwins(matchCatalog(embedding, catalog, judged), catalog, { script: judged?.script?.[0]?.label })
    },
    destroy: encoder.destroy
  }
}
