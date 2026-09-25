// A separate pinned selection: never rewrite the first pilot's source config.
import { readFile, writeFile } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'

export const SELECTION = [
  ['Synonym', 'sans'], ['Author', 'sans'], ['Bespoke Sans', 'sans'], ['Amulya', 'sans'],
  ['Excon', 'sans'], ['Plein', 'sans'], ['Ranade', 'sans'], ['Chillax', 'sans'],
  ['Gambarino', 'serif'], ['Rowan', 'serif'], ['Boska', 'serif'], ['Bespoke Serif', 'serif'],
  ['Bonny', 'serif'], ['Neco', 'serif'], ['Recia', 'serif'],
  ['Hoover', 'slab'], ['Paquito', 'slab'], ['RX100', 'mono'],
  ['Technor', 'display'], ['Pilcrow Rounded', 'sans'], ['Boxing', 'display'], ['Striper', 'display'],
  ['Nippo', 'display'], ['Stardom', 'display'],
]

// Pin the observed coverage of this bounded batch. These optional specimens
// were attempted once; changing this table explicitly opts into another attempt.
const PROVIDED_UNAVAILABLE = {
  Ranade: 'No source-provided single line tied to the selected static face (4 unverified candidates).',
  'Pilcrow Rounded': 'No source-provided single line tied to the selected static face (2 unverified candidates).',
  Striper: 'No single-line specimen element found in the declared face.',
}

export function resolveFamilies(catalogue, excludedNames) {
  return SELECTION.map(([family, category]) => {
    if (excludedNames.has(family)) throw new Error(`Selection overlaps an existing inventory: ${family}`)
    const entry = catalogue.find(item => item.name === family)
    const style = entry?.styles.find(item => item.weight.name === 'Regular' && !item.is_italic && !item.is_variable)
    if (!style) throw new Error(`No verified static Regular style: ${family}`)
    const sourceUrl = `https://www.fontshare.com/fonts/${entry.slug}`
    return {
      key: `fontshare-${entry.slug}`, source: 'fontshare', family,
      familyId: `fontshare:family:${entry.id}`, foundry: entry.publisher?.name ?? null,
      designers: (entry.designers ?? []).map(item => item.name),
      category, sourceCategory: entry.category, role: 'additional', specimenUrl: sourceUrl,
      provided: !Object.hasOwn(PROVIDED_UNAVAILABLE, family),
      providedUnavailable: PROVIDED_UNAVAILABLE[family] ?? null,
      providedMinChars: 8, providedViewportWidth: 2400,
      tokens: [family.toLowerCase()],
      faces: [{
        key: `fontshare-${entry.slug}-${style.weight.number}-upright`,
        faceId: `fontshare:style:${style.id}`, styleName: style.weight.name,
        declaredWeight: style.weight.number, declaredSlant: 'upright',
        cssWeight: style.weight.number, cssStyle: 'normal',
        fontUrl: `https:${style.file}.ttf`, cssUrl: null, sourceUrl, kind: 'base',
        cssFamilyToken: family.toLowerCase(), evidenceLabels: [style.weight.name],
      }],
    }
  })
}

async function main() {
  // The collection recipes stay out of the repository; only the build needs them, so a checkout still loads the selection above.
  const { politeFetch, RECIPES, RENDER } = await import('./preview-sources.mjs')
  const previous = JSON.parse(await readFile('bench/preview-sources.json', 'utf8'))
  const corpus = JSON.parse(await readFile('bench/corpus.json', 'utf8'))
  const catalogue = process.env.PREVIEW_FONTSHARE_CATALOGUE
    ? JSON.parse(await readFile(process.env.PREVIEW_FONTSHARE_CATALOGUE, 'utf8'))
    : await (await politeFetch('https://api.fontshare.com/v2/fonts?limit=200')).json()
  const excluded = new Set([...previous.families, ...corpus.families].map(item => item.family))
  const families = resolveFamilies(catalogue.fonts, excluded)
  const config = {
    version: 1, generatedAt: new Date().toISOString(), recipes: RECIPES, render: RENDER,
    sources: { fontshare: previous.sources.fontshare },
    inventoryComparison: { corpusCommit: corpus.commit, corpusFamilies: corpus.families.length,
      exactNameOverlap: 0, unseenDesignClaim: false },
    families,
  }
  const out = process.env.PREVIEW_SOURCES ?? 'bench/preview-sources-v2.json'
  await writeFile(out, JSON.stringify(config, null, 2) + '\n')
  console.log(`Wrote ${out}: ${families.length} families / ${families.length} static Regular faces`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main()
