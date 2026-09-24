// The site is the repository root. Only the legacy classifier build copies it into dist/.
import { cp, mkdir } from 'node:fs/promises'

export const siteFiles = ['index.html', 'catalogs.html', 'style.css', 'tokens.css', 'app.mjs', 'crop.mjs', 'catalogs.mjs', 'my-fonts.mjs']

export async function copySite() {
  await mkdir('dist', { recursive: true })
  for (const name of siteFiles) await cp(name, `dist/${name}`)
  await mkdir('dist/bench', { recursive: true }); await cp('bench/foundries.json', 'dist/bench/foundries.json')
}
