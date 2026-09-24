// Renders og.png, the 1200 × 630 link preview: the headline beside the app's own crop and first matches, captured from
// the running page, so the preview shows what the model really answers. Run: node scripts/og.mjs
import { spawn } from 'node:child_process'
import { writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'

const port = 4199, base = `http://127.0.0.1:${port}`, shown = 2
const server = spawn(process.execPath, ['scripts/demo-serve.mjs'], { env: { ...process.env, PORT: String(port) }, stdio: ['ignore', 'pipe', 'inherit'] })
await new Promise((resolve, reject) => { server.stdout.once('data', resolve); server.once('exit', () => reject(new Error('Server did not start'))) })
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])] })
try {
  // The app at its first state: the Lora sample, cropped, and its ranked matches with their previews loaded. A phone-width
  // layout, scaled up in the card, keeps the match names legible in a small preview.
  const app = await browser.newPage({ viewport: { width: 360, height: 1400 }, deviceScaleFactor: 3 })
  await app.goto(base)
  await app.waitForFunction(n => document.body.dataset.ready === 'true' && document.querySelectorAll('#results .result').length >= n, shown)
  await app.waitForLoadState('networkidle') // each preview's Google Fonts face
  await app.evaluate(() => { document.getElementById('input-regions').hidden = true; return document.fonts.ready })
  const png = buffer => `data:image/png;base64,${buffer.toString('base64')}`
  const rows = await app.locator('#results .result').evaluateAll((items, n) => items.slice(0, n).map(i => i.getBoundingClientRect().toJSON()), shown)
  const [first, last] = [rows[0], rows.at(-1)]
  const matches = png(await app.screenshot({ clip: { x: first.x, y: first.y, width: first.width, height: last.bottom - 1 - first.y } }))
  const crop = png(await app.locator('#image-frame').screenshot())

  const card = await browser.newPage({ viewport: { width: 1200, height: 630 } })
  await card.setContent(`<!doctype html><html lang="en"><head>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter&family=Source+Serif+4&display=swap">
<link rel="stylesheet" href="${base}/style.css">
<style>
body { width: 1200px; height: 630px; margin: 0; overflow: hidden; }
.og { display: grid; grid-template: minmax(0, 1fr) / 500px 1fr; gap: 64px; box-sizing: border-box; height: 100%; padding: 64px 72px; }
.og-copy { display: flex; flex-direction: column; }
.og-mark { font: 400 34px/1 var(--font-display); letter-spacing: -.02em; color: var(--color-ink); }
.og h1 { margin-top: auto; font: 400 68px/1.02 var(--font-display); letter-spacing: -.03em; color: var(--color-ink); }
.og-lead { margin-top: 28px; font: 400 22px/1.4 var(--font-body); color: var(--color-muted); }
.og-shot { display: grid; align-content: center; justify-self: end; gap: 24px; width: 460px; }
.og-shot img { display: block; width: 100%; height: auto; }
.og-crop { padding: 20px; border: 1px solid var(--color-rule); border-radius: var(--radius-panel); background: var(--color-well); }
</style></head><body><div class="og">
<div class="og-copy"><p class="og-mark">gpu-font</p><h1>Find the font in&nbsp;an&nbsp;image</h1><p class="og-lead">In your browser, on WebGPU: the closest free fonts, with weight and italic.</p></div>
<div class="og-shot"><div class="og-crop"><img src="${crop}" alt=""></div><img src="${matches}" alt=""></div>
</div></body></html>`)
  await card.evaluate(() => Promise.all([document.fonts.ready, ...[...document.images].map(i => i.decode())]))
  await writeFile('og.png', await card.screenshot({ type: 'png' }))
  console.log('Wrote og.png (1200 × 630).')
} finally { await browser.close(); server.kill() }
