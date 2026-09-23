// From an image of text to ranked fonts in two calls: load a model and a catalog, then match RGBA crops.
import { readNetwork, inferCPU } from './network.mjs'
import { createNetworkGPU } from './network-gpu.mjs'
import { prepareLine } from './line.mjs'
import { readCatalog, embedWindows, matchCatalog, foldTwins, sha256, preparationHash, readHeads, verdict } from './catalog.mjs'

export async function createMatcher(modelUrl, catalogUrl) {
  const bytes = await (await fetch(modelUrl)).arrayBuffer()
  const artifact = JSON.parse(new TextDecoder().decode(bytes)), model = readNetwork(artifact), heads = readHeads(artifact)
  // A catalog only fits the exact model and preparation that indexed it.
  const binding = { encoderSha256: await sha256(bytes), preparationSha256: await preparationHash(artifact.preparation) }
  const catalog = readCatalog(await (await fetch(catalogUrl)).json(), binding)
  let gpu = await createNetworkGPU(model).catch(() => null)
  return {
    // Fonts best first, identical designs folded into one entry with their `siblings`; [] when the crop holds no text.
    async match(imageData) {
      const { status, windows } = prepareLine(imageData, { ...model.preparation, deskew: true, sampler: 'windows' })
      if (status !== 'ok') return []
      let projections
      if (gpu) try { projections = await Promise.all(windows.map(w => gpu.infer(w))) } catch { gpu.destroy(); gpu = null } // A lost device falls back to the CPU.
      projections ??= windows.map(w => inferCPU(model, w))
      const embedding = embedWindows(projections), judged = heads ? verdict(embedding, heads) : null
      return foldTwins(matchCatalog(embedding, catalog, judged), catalog, { script: judged?.script?.[0]?.label })
    },
    destroy() { gpu?.destroy(); gpu = null }
  }
}
