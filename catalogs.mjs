// The Catalogs page: the catalogs a search covers (site.json), My fonts indexed in this browser, and every other source
// considered, rendered from bench/foundries.json as committed.
import { createEncoder } from './src/match.mjs'
import { buildCatalog, joinCatalogs } from './src/references.mjs'
import { readFontFile, styleOf, FONT_FILE } from './src/font-file.mjs'
import { readCatalog } from './src/catalog.mjs'
import { readMyFonts, saveMyFonts, removeMyFonts } from './my-fonts.mjs'
import sprae from 'https://cdn.jsdelivr.net/npm/sprae@13.9.4/dist/sprae.js'

const TERMS = { permitted: 'Allowed', 'none-found': 'No restriction found', 'not-verified': 'Not verified', restricted: 'Restricted', ban: 'Forbidden', mixed: 'Mixed' }
const WORK = { searched: 'Searched', 'not-searched': 'Not searched', indexed: 'Indexed', partial: 'Partly indexed', inventoried: 'Inventoried', planned: 'Planned', paused: 'Paused', 'permission-to-request': 'Permission to request', 'awaiting-permission': 'Awaiting permission', declined: 'Declined', excluded: 'Excluded' }
const $ = id => document.getElementById(id)
const count = n => n.toLocaleString('en-US')
const plural = (n, one, many) => `${count(n)} ${n === 1 ? one : many}`

const [data, site] = await Promise.all(['./bench/foundries.json', './site.json'].map(async path => (await fetch(path)).json()))
const tally = Object.keys(TERMS).map(status => [status, data.sources.filter(s => s.terms.status === status).length]).filter(([, n]) => n)
$('sources-summary').textContent = `${plural(site.families, 'family', 'families')} searched from ${plural(site.catalogs.length, 'catalog', 'catalogs')}; ${plural(data.sources.length, 'source', 'sources')} checked. Terms: ${tally.map(([status, n]) => `${TERMS[status].toLowerCase()} ${count(n)}`).join(', ')}.`
// One table: the catalogs a search covers as site.json ships them, then every other source grouped by why it isn't
// searched. Other and each group fold open into their sources; the column sort orders the sources inside each group.
const byId = new Map(data.sources.map(s => [s.id, s])), searched = new Set(site.catalogs.flatMap(option => option.sources?.map(s => s.id) ?? [option.id]))
// Each shipped catalog downloads as the JSON the search reads.
const size = bytes => bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1e3)} KB`
const agree = members => ({ status: members.every(s => s.terms.status === members[0].terms.status) ? members[0].terms.status : 'mixed' })
// Families counted by the work done, else the approximate size the source advertises.
const families = source => source.work.familiesIndexed ?? source.work.familiesInventoried ?? source.familiesAvailable ?? null
const shipped = site.catalogs.map(({ id, name, sources, families, faces, file, bytes }) => {
  const work = { status: 'searched', familiesIndexed: families, route: plural(faces, 'face', 'faces') }, json = { file, bytes }
  if (!sources) return byId.get(id) && { ...byId.get(id), work, json }
  const members = sources.map(({ id }) => byId.get(id)).filter(Boolean).map(s => ({ ...s, work: { ...s.work, status: 'searched' } }))
  return { id, name, url: null, members, licence: 'Per source', terms: agree(members), work, json }
}).filter(Boolean)
// Not searched, grouped by licence: open fonts whose terms allow collecting but aren't indexed yet; free fonts whose
// site terms forbid collecting or are unverified; and commercial fonts (paid, subscription, rental or tied to a platform).
// A group's families add up its members' family counts; ≈…+ where some are estimates or counted in other units.
// Icon sets are glyph collections, not fonts to find; they stay in the ledger, off the page.
const unsearched = data.sources.filter(s => !searched.has(s.id) && s.kind !== 'icons')
const commercial = s => /^(commercial|subscription|rental)/i.test(s.licence ?? '') || ['retailer', 'platform'].includes(s.kind)
const cleared = s => ['permitted', 'none-found'].includes(s.terms.status)
const REASONS = [['open', 'Open, not searched yet', s => !commercial(s) && cleared(s)], ['free', 'Free, not cleared', s => !commercial(s) && !cleared(s)], ['commercial', 'Commercial', commercial]]
const groups = [...shipped, ...REASONS.map(([id, name, belongs]) => {
  const members = unsearched.filter(belongs)
  const known = members.filter(m => (m.availableUnit ?? 'families') === 'families' && families(m) != null)
  return { id, name, url: null, members, licence: 'Per source', terms: agree(members), work: { status: 'not-searched' },
    familiesAvailable: known.reduce((n, m) => n + families(m), 0) || null, partial: known.length < members.length || known.some(m => m.work.familiesIndexed == null && m.work.familiesInventoried == null) }
}).filter(group => group.members.length)]
// A worked count reads exactly; a size the source advertises reads ≈N, with its unit under it when that is not families.
const counted = source => (source.work.familiesIndexed ?? source.work.familiesInventoried) != null
const familiesText = source => { const n = families(source)
  return n == null ? '–' : counted(source) ? count(n) : `≈${count(n)}${source.partial ? '+' : ''}` }
const familiesUnit = source => { const unit = source.availableUnit ?? 'families'
  return families(source) == null || counted(source) || unit === 'families' ? null : unit }
// Sorts by any column; families missing sort last either way. A click on the sorted column reverses it; a new column
// starts ascending, families descending.
const position = (list, value) => { const i = list.indexOf(value); return i < 0 ? list.length : i }
const keys = { name: s => s.name.toLowerCase(), terms: s => position(Object.keys(TERMS), s.terms.status), licence: s => (s.licence ?? '').toLowerCase(), status: s => position(Object.keys(WORK), s.work.status), families }
const sorted = (list, { key, dir }) => list.toSorted((a, b) => {
  const x = keys[key](a), y = keys[key](b)
  return (x == null) - (y == null) || (x < y ? -dir : x > y ? dir : 0) || a.name.localeCompare(b.name)
})
// Each cell leads with one line. A licence longer than its column (LEAD characters at the page's full width) leads
// with its first clause, up to a punctuation mark or a qualifier (with, per, for), and continues under it in small text;
// one with no clause to break at shows whole under its cut.
const LEAD = 30
const brief = text => {
  if (!text) return '–'
  let lead = text.length <= LEAD ? text : text.replace(/^(.+?)\s*(?:[,:;(]| (?:with|per|for) ).*$/s, '$1')
  if (lead.length > LEAD) lead = `${lead.slice(0, LEAD).replace(/\s+\S*$/, '')}…`
  return lead.replace(/^./, c => c.toUpperCase())
}
const rest = text => {
  if ((text?.length ?? 0) <= LEAD) return null
  const lead = brief(text)
  return lead.endsWith('…') ? text : text.slice(lead.length).replace(/^\s*[,:;]?\s*/, '')
}
const state = sprae(document.querySelector('main'), {
  TERMS, WORK, count, plural, familiesText, familiesUnit, size, groups, expanded: [], brief, rest,
  aside: work => work.reason ?? work.route, sort: { key: 'families', dir: -1 }, mine: [],
  // Folded groups by id; a plain list, since a store's missing keys would resolve in its parent scope.
  rows: (groups, expanded, sort) => groups.flatMap(group => [group, ...(group.members && expanded.includes(group.id) ? sorted(group.members, sort).map(source => ({ ...source, member: true })) : [])]),
  next: (sort, key) => ({ key, dir: sort.key === key ? -sort.dir : key === 'families' ? -1 : 1 }),
  order: (sort, key) => sort.key === key ? (sort.dir > 0 ? 'ascending' : 'descending') : null
})

// My fonts: indexed here with the site's model, kept in IndexedDB, offered in the search page's catalog menu.
const say = (text, error = false) => { $('my-fonts-status').textContent = text; $('my-fonts-status').toggleAttribute('data-error', error) }
const fits = catalog => catalog.encoderSha256 === site.modelSha256 && catalog.preparationSha256 === site.preparationSha256
let encoder = null, stopping = null
// Indexed fonts show as a table, one row per family; the status line only reports progress, notes and errors.
async function showStored(note = '') {
  const stored = await readMyFonts(), current = !!stored && fits(stored.catalog)
  $('my-fonts-stored').hidden = !current
  state.mine = current ? [...Map.groupBy(stored.catalog.faces, f => f.family)].map(([family, faces]) => ({ family, styles: faces.map(f => f.styleName).join(', '), faces: faces.length }))
    .sort((a, b) => a.family.localeCompare(b.family)) : []
  say(stored && !current ? 'Indexed for an earlier model; index them again to search them.' : note)
}
function busy(on) {
  for (const id of ['index-installed', 'index-folder', 'index-files', 'my-fonts-download', 'my-fonts-remove']) $(id).disabled = on
  $('index-stop').hidden = !on
}
// Each family's face nearest upright 400 comes first, then the second of each, so stopping early still covers every family.
function regularFirst(faces) {
  const distance = f => (f.style === 'italic' ? 1000 : 0) + Math.abs(f.weight - 400), order = new Map()
  for (const members of Map.groupBy(faces, f => f.family).values()) members.sort((a, b) => distance(a) - distance(b)).forEach((f, i) => order.set(f, i))
  return faces.toSorted((a, b) => order.get(a) - order.get(b))
}
// Adds `faces` to My fonts; a face indexed before is replaced.
async function index(faces, unread = 0) {
  if (!faces.length) { say(unread ? `No readable fonts: ${plural(unread, 'file', 'files')} skipped.` : 'No fonts found.', true); return }
  faces = regularFirst(faces)
  const controller = new AbortController(); stopping = controller; busy(true)
  try {
    encoder ??= createEncoder(site.model).then(e => { if (e.binding.encoderSha256 !== site.modelSha256) throw new Error('Model checksum failed.'); return e })
    const e = await encoder.catch(error => { encoder = null; throw error })
    const { catalog, skipped } = await buildCatalog(faces, { project: e.project, preparation: e.preparation, binding: e.binding, signal: controller.signal,
      progress: (done, total) => say(`Indexing ${count(done)} of ${plural(total, 'face', 'faces')}…`) })
    const missed = skipped.length + unread, note = missed ? `${plural(missed, 'font', 'fonts')} skipped: no Latin letters, or unreadable.` : ''
    if (!catalog) { await showStored(note || 'Nothing indexed.'); return }
    const stored = await readMyFonts(), joined = stored && fits(stored.catalog) ? joinCatalogs(stored.catalog, catalog) : catalog
    readCatalog(joined, e.binding)
    await saveMyFonts(joined)
    await showStored([controller.signal.aborted ? 'Stopped early.' : '', note].filter(Boolean).join(' '))
  } catch (error) { say(`Could not index: ${error.message}`, true) }
  finally { busy(false); stopping = null }
}
// Installed fonts as the system lists them; each face renders by its PostScript name, so no font file is read.
$('index-installed').hidden = !('queryLocalFonts' in window)
$('index-installed').addEventListener('click', async () => {
  let fonts
  try { fonts = await window.queryLocalFonts() } catch (error) { say(error.name === 'NotAllowedError' ? 'Permission to list fonts was denied.' : error.message, true); return }
  index(fonts.map(f => ({ id: `local:${f.postscriptName}`, family: f.family, styleName: f.style, ...styleOf(f.style), source: `local("${f.postscriptName}")`, local: f.postscriptName })))
})
async function indexFiles(files) {
  const picked = [...files].filter(file => FONT_FILE.test(file.name)), faces = []
  let unread = 0
  say(`Reading ${plural(picked.length, 'file', 'files')}…`)
  for (const file of picked) {
    try { for (const face of await readFontFile(file)) faces.push({ ...face, id: `file:${face.postscriptName ?? `${face.family} ${face.styleName}`}` }) } catch { unread++ }
  }
  index(faces, unread)
}
$('index-folder').addEventListener('click', () => $('folder-input').click())
$('index-files').addEventListener('click', () => $('files-input').click())
for (const id of ['folder-input', 'files-input']) $(id).addEventListener('change', event => { const files = [...event.target.files]; event.target.value = ''; if (files.length) indexFiles(files) })
$('index-stop').addEventListener('click', () => stopping?.abort())
// My fonts leave as the catalog JSON a search reads: open it from the catalog menu in any browser, or pass its URL to
// createMatcher(). Like any catalog, it fits only the model that built it.
$('my-fonts-download').addEventListener('click', async () => {
  const stored = await readMyFonts()
  if (!stored) return
  const url = URL.createObjectURL(new Blob([JSON.stringify(stored.catalog) + '\n'], { type: 'application/json' }))
  Object.assign(document.createElement('a'), { href: url, download: 'my-fonts.json' }).click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
})
$('my-fonts-remove').addEventListener('click', async () => { await removeMyFonts(); showStored('Removed.') })
showStored()
