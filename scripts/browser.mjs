import { chromium } from 'playwright'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const configBytes = await readFile('bench/pilot.json')
const fontBytes = await readFile('bench/fonts.json')
const config = JSON.parse(configBytes), fonts = JSON.parse(fontBytes)
await mkdir('.data/browser', { recursive: true })
const browser = await chromium.launch()
try {
  const page = await browser.newPage()
  const samples = []
  for (const font of fonts.fonts) {
    const bytes = await readFile(`.data/fonts/${font.id}.ttf`)
    if (hash(bytes) !== font.instanceSha256) throw new Error(`Instance checksum failed: ${font.id}`)
    const crops = await page.evaluate(async ({ bytes, config }) => {
      const face = new FontFace('pilot', new Uint8Array(bytes).buffer)
      await face.load()
      document.fonts.clear()
      document.fonts.add(face)
      const crops = []
      for (const text of config.evaluation.texts.slice(1)) for (const size of config.evaluation.sizes) {
        for (const dpr of config.evaluation.dprs) for (const polarity of config.evaluation.polarities) {
          const canvas = document.createElement('canvas')
          let ctx = canvas.getContext('2d', { willReadFrequently: true })
          ctx.font = `${size * dpr}px pilot`
          const m = ctx.measureText(text), pad = 6 * dpr
          canvas.width = Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + pad * 2
          canvas.height = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + pad * 2
          ctx = canvas.getContext('2d', { willReadFrequently: true })
          ctx.fillStyle = polarity === 'dark' ? '#fff' : '#000'
          ctx.fillRect(0, 0, canvas.width, canvas.height)
          ctx.fillStyle = polarity === 'dark' ? '#000' : '#fff'
          ctx.font = `${size * dpr}px pilot`
          ctx.fillText(text, pad + m.actualBoundingBoxLeft, pad + m.actualBoundingBoxAscent)
          crops.push({ text, size, dpr, polarity, width: canvas.width, height: canvas.height,
            rgba: Array.from(ctx.getImageData(0, 0, canvas.width, canvas.height).data) })
        }
      }
      return crops
    }, { bytes: Array.from(bytes), config })
    for (const [index, crop] of crops.entries()) {
      const { rgba, ...meta } = crop
      const id = `${font.id}-${String(index).padStart(4, '0')}`
      const raw = `.data/browser/${id}.rgba`
      await writeFile(raw, new Uint8Array(rgba))
      samples.push({ id, family: font.id, split: font.split, role: 'query', renderer: 'chromium', raw, ...meta })
    }
  }
  await writeFile('.data/browser/samples.json', JSON.stringify({ renderer: `Chromium ${browser.version()}`, configSha256: hash(configBytes), fontsSha256: hash(fontBytes), samples }, null, 2) + '\n')
  console.log(`Rendered ${samples.length} Chromium Canvas queries; these are synthetic, not independent real screenshots`)
} finally {
  await browser.close()
}
