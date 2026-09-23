// The site is the repository root; dist/ is its static build.
import { cp, mkdir } from 'node:fs/promises'

export const siteFiles = ['index.html', 'sources.html', 'style.css', 'tokens.css', 'app.mjs', 'crop.mjs', 'sources.mjs']

export async function copySite() {
  await mkdir('dist', { recursive: true })
  for (const name of siteFiles) await cp(name, `dist/${name}`)
  // The Sources page renders the terms ledger as committed.
  await mkdir('dist/assets', { recursive: true }); await cp('bench/foundries.json', 'dist/assets/foundries.json')
}
