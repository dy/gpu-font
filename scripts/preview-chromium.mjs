// Faces FreeType cannot draw, drawn by Chromium as the catalogs index them: colour layers over empty outlines
// (Basalte Multicolor), outlines FreeType rejects or overflows on. scripts/previews.py hands it the faces it failed.
//
//   node scripts/preview-chromium.mjs faces.json     each { file, text, weight, italic, out }: the face's text, black on white, at out
import { readFile, writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'

const faces = JSON.parse(await readFile(process.argv[2], 'utf8'))
const browser = await chromium.launch()
try {
  const page = await browser.newPage()
  for (const [index, face] of faces.entries()) {
    const png = await page.evaluate(async ({ index, bytes, text, weight, italic }) => {
      const font = new FontFace(`face-${index}`, Uint8Array.from(atob(bytes), c => c.charCodeAt(0)), { weight: String(weight), style: italic ? 'italic' : 'normal' })
      await font.load(); document.fonts.add(font)
      const size = 96, css = `${italic ? 'italic ' : ''}${weight} ${size}px "${font.family}"`, probe = new OffscreenCanvas(1, 1).getContext('2d')
      probe.font = css
      const canvas = new OffscreenCanvas(Math.ceil(probe.measureText(text).width) + 4 * size, 5 * size), c = canvas.getContext('2d')
      c.fillStyle = '#fff'; c.fillRect(0, 0, canvas.width, canvas.height)
      c.font = css; c.fillStyle = '#000'; c.textBaseline = 'alphabetic'; c.fillText(text, 2 * size, 3 * size)
      document.fonts.delete(font)
      const blob = await canvas.convertToBlob({ type: 'image/png' })
      return new Promise(resolve => { const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]); reader.readAsDataURL(blob) })
    }, { index, bytes: (await readFile(face.file)).toString('base64'), text: face.text, weight: face.weight, italic: face.italic })
    await writeFile(face.out, Buffer.from(png, 'base64'))
  }
} finally { await browser.close() }
