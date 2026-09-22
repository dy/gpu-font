// Build contact-sheet.html: every accepted region, drawn over its own image, so
// a person can check face identity and crop boundaries without network access.
import { readFile, writeFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { readJsonl, sha256 } from './preview-lib.mjs'

const dir = process.argv[2] ?? '.data/previews/pilot-v1'
const records = await readJsonl(path.join(dir, 'manifest.jsonl'))
const attempts = await readJsonl(path.join(dir, 'attempts.jsonl'))
const manifestSha = sha256(await readFile(path.join(dir, 'manifest.jsonl')))

const escape = value => String(value ?? '').replace(/[&<>"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[character]))
const families = new Map()
for (const record of records) {
  const group = families.get(record.familyId) ?? { family: record.family, records: [] }
  group.records.push(record)
  families.set(record.familyId, group)
}

const card = record => {
  const { image, region } = record
  const percent = (value, total) => `${(value / total * 100).toFixed(3)}%`
  return `<figure>
  <a href="${escape(image.path)}" title="full resolution">
    <span class="frame">
      <img src="${escape(image.path)}" alt="${escape(record.family)} ${escape(record.styleName ?? '')} ${escape(record.recipeId)}" loading="lazy">
      <span class="region" style="left:${percent(region.x, image.width)};top:${percent(region.y, image.height)};width:${percent(region.width, image.width)};height:${percent(region.height, image.height)}"></span>
    </span>
  </a>
  <figcaption>
    <b>${escape(record.recipeId)}</b> · ${escape(record.styleName ?? 'style unknown')} ·
    weight ${escape(record.weight ?? 'null')} · ${escape(record.slant ?? 'null')} ·
    ${image.width}×${image.height} px · region ${region.width}×${region.height} @ ${region.x},${region.y}<br>
    text ${record.text === null ? '<i>null</i>' : `“${escape(record.text)}”`} · scripts ${escape(record.scripts.join(', ') || 'unknown')}<br>
    <a href="${escape(record.evidence)}">evidence</a> ·
    <a href="${escape(record.sourceUrl)}">source</a> ·
    <code>${escape(record.id)}</code>
    ${record.flags.length ? `<br><span class="flags">${record.flags.map(escape).join(' · ')}</span>` : ''}
  </figcaption>
</figure>`
}

const section = ([familyId, group]) => {
  const first = group.records[0]
  return `<section>
  <h2>${escape(group.family)}</h2>
  <p class="meta">${escape(first.foundry ?? 'foundry unknown')} · ${escape(first.source?.name ?? '')} ·
    ${escape(first.source?.category ?? '')} · <code>${escape(familyId)}</code> ·
    ${new Set(group.records.map(record => record.faceId)).size} face(s) · ${group.records.length} records</p>
  <div class="grid">${group.records.sort((a, b) => a.id.localeCompare(b.id)).map(card).join('\n')}</div>
</section>`
}

const html = `<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>Preview pilot contact sheet</title>
<style>
  :root { color-scheme: light; }
  body { margin: 0; padding: 24px; font: 13px/1.5 ui-sans-serif, system-ui, sans-serif; background: #f6f6f4; color: #16161a; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 28px 0 2px; }
  .meta, .summary { color: #55555f; }
  .summary { margin: 0 0 8px; }
  .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); margin-top: 10px; }
  figure { margin: 0; background: #fff; border: 1px solid #e0e0dc; border-radius: 6px; overflow: hidden; }
  .frame { position: relative; display: block; }
  img { display: block; width: 100%; height: auto; background: #fff; }
  .region { position: absolute; outline: 1px solid rgba(214, 40, 40, .85); box-shadow: 0 0 0 1px rgba(255,255,255,.6) inset; }
  figcaption { padding: 8px 10px; font-size: 11px; line-height: 1.45; border-top: 1px solid #efefeb; }
  .flags { color: #9a5b00; }
  code { font: 11px/1.4 ui-monospace, monospace; color: #444; }
  table { border-collapse: collapse; font-size: 12px; margin-top: 8px; }
  td, th { border: 1px solid #e0e0dc; padding: 4px 8px; text-align: left; vertical-align: top; }
</style>
<h1>Raster preview pilot — contact sheet</h1>
<p class="summary">${records.length} accepted records · ${families.size} families ·
${new Set(records.map(record => record.faceId)).size} faces · manifest sha256 <code>${manifestSha}</code> ·
built ${new Date().toISOString().replace(/\.\d+Z$/, 'Z')}<br>
Red outline marks the manifest region in image pixels. Images are third-party captures kept local; see report.md for source terms.</p>
${[...families].sort((a, b) => a[1].family.localeCompare(b[1].family)).map(section).join('\n')}
<section>
  <h2>Logged attempts (${attempts.length})</h2>
  <table><tr><th>outcome</th><th>what</th><th>reason</th><th>retained</th></tr>
  ${attempts.map(attempt => `<tr><td>${escape(attempt.outcome)}</td><td>${escape([attempt.family, attempt.faceKey, attempt.recipeId, attempt.id].filter(Boolean).join(' · '))}</td><td>${escape(attempt.reason)}</td><td>${escape((attempt.retained ?? []).join(' '))}</td></tr>`).join('\n')}
  </table>
</section>
</html>
`
await writeFile(path.join(dir, 'contact-sheet.html'), html)
console.log(`Wrote ${path.join(dir, 'contact-sheet.html')} for ${records.length} records`)
