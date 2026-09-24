// The canon: a ranked list of typefaces every designer should know, with where
// each one stands in our coverage. Built only from open facts: Wikidata (CC0)
// for identity, designer, year and Wikipedia reach; Wikimedia pageviews for
// attention; Google Fonts' own popularity ranks for the open web. Naming
// typefaces and stating facts about them needs no one's permission.
//
//   node scripts/canon.mjs [--size 1000] [--refresh] [--private]
//
// --private also counts the preview captures held for private experiments and writes
// to .data/private/canon.json; the committed bench/canon.json never mentions them.
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'

const CACHE = '.data/canon'
const PRIVATE = process.argv.includes('--private')
const OUT = PRIVATE ? '.data/private/canon.json' : 'bench/canon.json'
const UA = 'gpu-font/0.1 (typeface recognition research; https://github.com/dy/gpu-font)'
const args = process.argv.slice(2)
const size = Number(args[args.indexOf('--size') + 1] || 1000) || 1000
const refresh = args.includes('--refresh')
// Promotional entries get seeded across small wikis, so reach counts only these.
const MAJOR_WIKIS = ['enwiki', 'dewiki', 'frwiki', 'jawiki', 'eswiki', 'ruwiki', 'itwiki', 'zhwiki', 'ptwiki', 'plwiki', 'nlwiki', 'svwiki', 'ukwiki', 'kowiki', 'cswiki']
const TYPEFACE_CLASSES = ['Q17451', 'Q58481926'] // "typeface" and "typeface family": classics use the latter

const wait = ms => new Promise(resolve => setTimeout(resolve, ms))
const key = name => name.toLowerCase().replace(/[®™©]/g, '').normalize('NFKD').replace(/[^a-z0-9]/g, '')

async function cached(name, produce) {
  const file = `${CACHE}/${name}.json`
  if (!refresh) try { return JSON.parse(await readFile(file, 'utf8')) } catch {}
  const value = await produce()
  await writeFile(file, JSON.stringify(value))
  return value
}

async function getJson(url, init = {}) {
  for (let attempt = 1; ; attempt++) {
    const response = await fetch(url, { ...init, headers: { 'user-agent': UA, accept: 'application/json', ...init.headers } })
    if (response.ok) return response.json()
    if (response.status === 404) return null
    if (attempt >= 4 || ![429, 500, 502, 503, 504].includes(response.status)) throw new Error(`${response.status} for ${url.slice(0, 120)}`)
    await wait(2000 * 2 ** attempt)
  }
}

const sparql = query => getJson(`https://query.wikidata.org/sparql?query=${encodeURIComponent(query)}`, { headers: { accept: 'application/sparql-results+json' } })

async function typefaces() {
  const rows = []
  for (const cls of TYPEFACE_CLASSES) {
    const result = await sparql(`SELECT ?item ?itemLabel (SAMPLE(?year) AS ?year)
      (GROUP_CONCAT(DISTINCT ?designerLabel; separator="|") AS ?designers) (GROUP_CONCAT(DISTINCT ?makerLabel; separator="|") AS ?foundries) WHERE {
      ?item wdt:P31/wdt:P279* wd:${cls} .
      OPTIONAL { ?item wdt:P571 ?date . BIND(YEAR(?date) AS ?year) }
      OPTIONAL { ?item wdt:P287 ?designer . ?designer rdfs:label ?designerLabel . FILTER(LANG(?designerLabel) = "en") }
      OPTIONAL { ?item wdt:P176 ?maker . ?maker rdfs:label ?makerLabel . FILTER(LANG(?makerLabel) = "en") }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } GROUP BY ?item ?itemLabel`)
    for (const b of result.results.bindings) rows.push({
      id: b.item.value.split('/').pop(), name: b.itemLabel.value, year: b.year ? Number(b.year.value) : null,
      designers: b.designers?.value ? b.designers.value.split('|') : [], foundries: b.foundries?.value ? b.foundries.value.split('|') : [],
    })
    await wait(1500)
  }
  return [...new Map(rows.map(row => [row.id, row])).values()].filter(row => !/^Q\d+$/.test(row.name))
}

const STYLE_CLASSES = new Set(['handwriting style', 'script style', 'writing system', 'natural writing system', 'bicameral script', 'cursive', 'teaching script', 'calligraphy style'])

/** Typeface, genre (a class other entries belong to) or writing style, from Wikidata's own typing. */
async function kinds(ids) {
  const out = {}
  for (let i = 0; i < ids.length; i += 150) {
    const values = ids.slice(i, i + 150).map(id => `wd:${id}`).join(' ')
    const result = await sparql(`SELECT ?item (GROUP_CONCAT(DISTINCT ?cLabel; separator="|") AS ?classes) (COUNT(DISTINCT ?member) AS ?members) WHERE {
      VALUES ?item { ${values} }
      OPTIONAL { ?item wdt:P31 ?c . ?c rdfs:label ?cLabel . FILTER(LANG(?cLabel) = "en") }
      OPTIONAL { ?member wdt:P31|wdt:P279 ?item } } GROUP BY ?item`)
    for (const b of result.results.bindings)
      out[b.item.value.split('/').pop()] = { classes: b.classes?.value ? b.classes.value.split('|') : [], members: Number(b.members.value) }
    await wait(1200)
  }
  return out
}

function kindOf(face, info) {
  if (info?.classes.some(label => STYLE_CLASSES.has(label))) return 'style'
  if (/\b(typefaces?|type)$/i.test(face.name) || (info?.members ?? 0) >= 2) return 'genre'
  return 'typeface'
}

async function sitelinks(ids) {
  const out = {}
  for (let i = 0; i < ids.length; i += 50) {
    const batch = ids.slice(i, i + 50)
    const data = await getJson(`https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=sitelinks&ids=${batch.join('|')}`)
    for (const [id, entity] of Object.entries(data.entities ?? {})) {
      const links = entity.sitelinks ?? {}
      out[id] = { total: Object.keys(links).length, major: MAJOR_WIKIS.filter(wiki => links[wiki]).length, enTitle: links.enwiki?.title ?? null }
    }
    await wait(300)
  }
  return out
}

async function pageviews(titles) {
  const out = {}, queue = [...titles]
  const end = new Date(); end.setUTCDate(1)
  const start = new Date(end); start.setUTCFullYear(end.getUTCFullYear() - 1)
  const day = date => date.toISOString().slice(0, 10).replace(/-/g, '')
  const worker = async () => {
    for (let title; (title = queue.shift());) {
      const url = `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/${encodeURIComponent(title.replace(/ /g, '_'))}/monthly/${day(start)}/${day(end)}`
      const data = await getJson(url).catch(() => null)
      out[title] = data ? data.items.reduce((sum, item) => sum + item.views, 0) : 0
      await wait(150)
    }
  }
  await Promise.all(Array.from({ length: 4 }, worker))
  return out
}

async function googlePopularity() {
  const text = await (await fetch('https://fonts.google.com/metadata/fonts', { headers: { 'user-agent': UA } })).text()
  const data = JSON.parse(text.replace(/^\)\]\}'\n?/, ''))
  return (data.familyMetadataList ?? []).map(family => ({ name: family.family, popularity: family.popularity, designers: family.designers ?? [], category: family.category }))
}

/** Families captured from preview-only sources: the pilot archive and the paused catalogue runs. */
export async function heldFamilies() {
  const rows = []
  const lines = async file => { try { return (await readFile(file, 'utf8')).split('\n').filter(Boolean).map(line => JSON.parse(line)) } catch { return [] } }
  for (const record of await lines('.data/previews/pilot-v1/manifest.jsonl'))
    if (['myfonts', 'adobe-fonts'].includes(record.source?.key)) rows.push({ name: record.family, source: record.source.key })
  for (const source of ['myfonts', 'adobe-fonts'])
    for (const record of await lines(`.data/previews/catalog-v1/${source}/manifest.jsonl`)) rows.push({ name: record.family, source })
  return [...new Map(rows.map(row => [`${row.source}|${row.name}`, row])).values()]
}

const bare = name => name.replace(/\s+(fonts|font family|typeface family|family|typeface)$/i, '').trim()

// Documented open equivalents of a classic (bench/equivalents.json). Only relations
// that reproduce the letterforms count as coverage; metric twins and "inspired by" do not.
const VISUAL = ['same-design', 'clone', 'revival', 'basis']
const MATCHES = ['exact', ...VISUAL]

/** Where a classic stands, given the families we hold and the documented equivalents. */
export function matcher(families, equivalentRows) {
  const order = { indexed: 0, inventoried: 1, 'held-locally': 2 }
  const known = [...families].sort((a, b) => order[a.status] - order[b.status])
  const find = name => known.find(entry => key(entry.name) === key(name))
  const equivalents = {}
  for (const row of equivalentRows) (equivalents[key(row.classic)] ??= []).push(row)
  return name => {
    const target = bare(name), k = key(target)
    // Every documented equivalent, with where it stands here, so a reader sees metric twins too.
    const listed = (equivalents[k] ?? equivalents[key(name)] ?? []).map(({ family, relation, evidence }) => ({ family, relation, status: find(family)?.status ?? 'missing', evidence }))
    const extra = listed.length ? { equivalents: listed } : {}
    // The name itself or a documented twin, whichever we can serve best: an open clone
    // on file outranks the original captured from a preview-only source.
    const documented = [find(target) && { entry: find(target), match: 'exact' },
      ...listed.filter(row => VISUAL.includes(row.relation) && row.status !== 'missing').map(row => ({ entry: find(row.family), match: row.relation }))]
      .filter(Boolean).sort((a, b) => order[a.entry.status] - order[b.entry.status] || MATCHES.indexOf(a.match) - MATCHES.indexOf(b.match))[0]
    if (documented) return { status: documented.entry.status, source: documented.entry.source, via: documented.entry.name, match: documented.match, ...extra }
    // A family that carries the name as whole words is related, not the same design:
    // Helvetica Now, EB Garamond, Libre Baskerville, DejaVu Sans. Best status, then the shortest name.
    // A documented relation outranks the name: Libre Caslon is only inspired by Caslon.
    const words = new RegExp(`(^|\\s)${target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(\\s|$)`, 'i')
    const documentedNames = new Set(listed.map(row => key(row.family)))
    const related = target.length >= 4 && known.filter(entry => words.test(entry.name) && !documentedNames.has(key(entry.name)))
      .sort((a, b) => order[a.status] - order[b.status] || a.name.length - b.name.length)[0]
    return related ? { status: related.status, source: related.source, via: related.name, match: 'related', ...extra } : { status: 'missing', ...extra }
  }
}

async function coverage() {
  const read = async file => { try { return JSON.parse(await readFile(file, 'utf8')) } catch { return null } }
  const known = []
  for (const family of (await read('bench/corpus.json'))?.families ?? []) if (!family.excluded) known.push({ name: family.family, status: 'indexed', source: 'google-fonts' })
  for (const file of (await readdir('models/encoder/catalogs')).filter(name => name.endsWith('.json') && name !== 'index.json'))
    // A grouped catalogue (Other) tags each face with the source it came from.
    for (const face of (await read(`models/encoder/catalogs/${file}`))?.faces ?? []) known.push({ name: face.family, status: 'indexed', source: face.sourceId ?? file.slice(0, -5) })
  for (const family of (await read('bench/open-fonts.json'))?.families ?? []) if (!family.excluded) known.push({ name: family.family, status: 'inventoried', source: family.source })
  // Preview captures held for private experiments, read from the capture archives themselves.
  if (PRIVATE) for (const held of await heldFamilies()) known.push({ ...held, status: 'held-locally' })
  return matcher(known, (await read('bench/equivalents.json'))?.equivalents ?? [])
}

async function main() {
  await mkdir(CACHE, { recursive: true })
  await mkdir(OUT.slice(0, OUT.lastIndexOf('/')), { recursive: true })
  const faces = await cached('wikidata-typefaces', typefaces)
  console.log(`Wikidata: ${faces.length} typefaces`)
  const links = await cached('wikidata-sitelinks', () => sitelinks(faces.map(face => face.id)))
  const kindInfo = await cached('wikidata-kinds', () => kinds(faces.map(face => face.id)))
  const titles = [...new Set(faces.map(face => links[face.id]?.enTitle).filter(Boolean))]
  const views = await cached('enwiki-pageviews', () => pageviews(titles))
  console.log(`English Wikipedia articles: ${titles.length}`)
  const google = await cached('google-fonts-popularity', googlePopularity)
  console.log(`Google Fonts families with a popularity rank: ${google.length}`)

  // Wikipedia side: reach across major wikis and a year of attention, both on a log scale.
  const maxViews = Math.max(1, ...Object.values(views))
  const entries = new Map()
  for (const face of faces) {
    const link = links[face.id] ?? { total: 0, major: 0, enTitle: null }
    const viewCount = link.enTitle ? views[link.enTitle] ?? 0 : 0
    if (!link.major && !viewCount) continue
    // Reach counts in proportion to readership: seeded articles nobody reads cannot carry a face.
    const readership = Math.sqrt(Math.min(1, viewCount / 50000))
    const score = 0.6 * Math.log1p(viewCount) / Math.log1p(maxViews) + 0.4 * readership * link.major / MAJOR_WIKIS.length
    const k = key(face.name), previous = entries.get(k)
    if (previous && previous.score >= score) continue
    entries.set(k, {
      name: face.name, kind: kindOf(face, kindInfo[face.id]), wikidata: face.id, designers: face.designers, year: face.year, foundries: face.foundries,
      signals: { majorWikipedias: link.major, wikipediaSitelinks: link.total, enwikiPageviewsYear: viewCount }, score,
    })
  }
  // Open-web side: Google's own popularity ranks (lower is more popular).
  const ranked = google.filter(family => family.popularity).sort((a, b) => a.popularity - b.popularity)
  ranked.forEach((family, index) => {
    const score = 0.9 * (1 - Math.log1p(index) / Math.log1p(ranked.length))
    const k = key(family.name), entry = entries.get(k)
    if (entry) { entry.signals.googleFontsRank = index + 1; entry.score = Math.max(entry.score, score); return }
    entries.set(k, { name: family.name, kind: 'typeface', wikidata: null, designers: family.designers, year: null, foundries: ['Google Fonts'],
      signals: { googleFontsRank: index + 1 }, score })
  })

  const lookup = await coverage()
  const sorted = [...entries.values()].sort((a, b) => b.score - a.score)
  // Styles and genres are worth knowing but are not fonts to recognise; they are listed apart.
  const canon = sorted.filter(entry => entry.kind === 'typeface').slice(0, size).map((entry, index) => ({
    rank: index + 1, ...entry, score: Math.round(entry.score * 1000) / 1000, coverage: lookup(entry.name),
  }))
  const styles = sorted.filter(entry => entry.kind !== 'typeface' && entry.score >= canon.at(-1).score)
    .map(({ name, kind, wikidata, designers, year, signals }) => ({ name, kind, wikidata, designers, year, signals }))
  const tally = canon.reduce((counts, entry) => {
    const { status, match } = entry.coverage
    const label = !match || match === 'exact' ? status : `${status} (${match === 'related' ? 'related family' : match})`
    return { ...counts, [label]: (counts[label] ?? 0) + 1 }
  }, {})
  await writeFile(OUT, JSON.stringify({
    version: 1, generated: new Date().toISOString().slice(0, 10), size: canon.length,
    method: 'Score = the larger of a Wikipedia score (0.6 × log English pageviews over the last year + 0.4 × share of 15 major Wikipedias with an article) and a Google Fonts score (0.9 × log-rank of its popularity). Identity, designers, year and foundry from Wikidata (CC0).',
    sources: ['https://query.wikidata.org/', 'https://wikimedia.org/api/rest_v1/', 'https://fonts.google.com/metadata/fonts'],
    coverage: tally, typefaces: canon, stylesAndGenres: styles,
  }, null, 1) + '\n')
  console.log(`Wrote ${OUT}: ${canon.length} typefaces, ${styles.length} styles and genres listed apart`, tally)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
