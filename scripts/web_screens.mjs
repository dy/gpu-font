// Screenshots of real web pages, each crop labelled by the font Chromium actually drew it with.
//
// People bring screenshots of websites. Here the label is not a guess: for each line of text the browser
// reports the font files that rendered its glyphs (CDP CSS.getPlatformFontsForNode), and a crop is kept only
// when a single font drew every glyph. Sites come from the Chrome UX Report's top origins (Google's
// real-usage ranking, published as CSV on GitHub); each homepage is loaded once, under Chromium's own user
// agent with this tool's name added, and only when its robots.txt allows this tool and ClaudeBot to fetch it.
// The crops are other people's designs: they stay in .data/screens for evaluation.
//
//   node scripts/web_screens.mjs [--top 5000] [--workers 4]
import { chromium } from 'playwright'
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { gunzipSync } from 'node:zlib'

const OUT = '.data/screens'
const LIST = 'https://raw.githubusercontent.com/zakird/crux-top-lists/e4997a8328df329b11f29d737287ab1c18ec669a/data/global/202608.csv.gz'
const NAME = 'gpu-font-research (+https://github.com/dy/gpu-font)'
const option = (name, fallback) => { const at = process.argv.indexOf(`--${name}`); return at < 0 ? fallback : Number(process.argv[at + 1]) }
const top = option('top', 5000), workers = option('workers', 4)
const PER_PAGE = 12 // crops a page at most, one per font and size
const DEADLINE = 120000 // ms a site may hold its worker: a page that never settles is closed, not waited on
const wait = ms => new Promise(resolve => setTimeout(resolve, ms))

/** Whether robots.txt lets both this tool and ClaudeBot fetch `path`: the groups naming either, else the * group. */
export function allowed(robots, path = '/') {
  const groups = [], lines = robots.split(/\r?\n/).map(line => line.replace(/#.*/, '').trim()).filter(Boolean)
  let current = null
  for (const line of lines) {
    const [field, ...rest] = line.split(':'), value = rest.join(':').trim(), key = field.trim().toLowerCase()
    if (key === 'user-agent') { if (!current || current.rules.length) groups.push(current = { agents: [], rules: [] }); current.agents.push(value.toLowerCase()) }
    else if ((key === 'allow' || key === 'disallow') && current) current.rules.push({ allow: key === 'allow', path: value })
  }
  const verdict = agent => {
    const named = groups.filter(group => group.agents.some(name => name !== '*' && agent.includes(name)))
    const rules = (named.length ? named : groups.filter(group => group.agents.includes('*'))).flatMap(group => group.rules)
    const match = rule => rule.path && new RegExp('^' + rule.path.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\\\$$/, '$')).test(path)
    const best = rules.filter(match).sort((a, b) => b.path.length - a.path.length || a.allow - b.allow)[0] // longest rule wins; Disallow on a tie
    return !best || best.allow
  }
  return verdict('gpu-font-research') && verdict('claudebot')
}

async function origins() {
  const response = await fetch(LIST)
  const rows = gunzipSync(Buffer.from(await response.arrayBuffer())).toString().trim().split('\n').slice(1).map(line => line.split(','))
  return rows.filter(([, rank]) => Number(rank) <= top).map(([origin]) => origin)
}

async function robotsFor(origin) {
  try {
    const response = await fetch(`${origin}/robots.txt`, { headers: { 'user-agent': NAME }, signal: AbortSignal.timeout(15000) })
    return response.ok ? await response.text() : '' // no robots.txt: nothing is refused
  } catch { return null } // the site did not answer: leave it
}

/** One-line text elements holding their own text, marked for the browser to name their fonts. */
const candidates = () => {
  const out = []
  for (const element of document.querySelectorAll('h1,h2,h3,h4,h5,h6,p,a,span,button,li,label,td,th,figcaption,blockquote,div')) {
    if (out.length >= 80) break
    if ([...element.children].some(child => child.textContent.trim())) continue // a child element could be set in another font
    const text = element.textContent.replace(/\s+/g, ' ').trim(), style = getComputedStyle(element)
    if (text.length < 4 || text.length > 60 || style.visibility !== 'visible' || Number(style.opacity) < 0.9 || parseFloat(style.fontSize) < 11) continue
    const range = document.createRange(); range.selectNodeContents(element)
    const rects = [...range.getClientRects()].filter(rect => rect.width > 1 && rect.height > 1)
    if (rects.length !== 1) continue
    element.setAttribute('data-screen', out.length)
    out.push({ index: out.length, text, fontFamily: style.fontFamily, fontSize: parseFloat(style.fontSize), fontWeight: style.fontWeight, fontStyle: style.fontStyle, color: style.color })
  }
  return out
}

async function capture(browser, options, origin) {
  // A context per site: closing it ends everything the site started (popups, iframes, service workers)
  const context = await browser.newContext(options), records = []
  const deadline = setTimeout(() => context.close().catch(() => {}), DEADLINE)
  try {
    const page = await context.newPage(), client = await context.newCDPSession(page)
    await page.goto(origin, { waitUntil: 'load', timeout: 30000 })
    await page.waitForTimeout(1500); await page.evaluate(() => document.fonts.ready)
    const found = await page.evaluate(candidates)
    await client.send('DOM.enable'); await client.send('CSS.enable')
    const { root } = await client.send('DOM.getDocument', { depth: 0 })
    const { nodeIds } = await client.send('DOM.querySelectorAll', { nodeId: root.nodeId, selector: '[data-screen]' })
    const seen = new Set(), host = new URL(page.url()).hostname
    for (const nodeId of nodeIds) {
      if (records.length >= PER_PAGE) break
      const { attributes } = await client.send('DOM.getAttributes', { nodeId })
      const item = found[Number(attributes[attributes.indexOf('data-screen') + 1])]
      const { fonts } = await client.send('CSS.getPlatformFontsForNode', { nodeId })
      if (fonts.length !== 1) continue // several fonts shared the glyphs: no single label
      const [font] = fonts, key = `${font.postScriptName}|${Math.round(item.fontSize)}`
      if (seen.has(key)) continue
      seen.add(key)
      const box = await page.evaluate(index => {
        const element = document.querySelector(`[data-screen="${index}"]`); element.scrollIntoView({ block: 'center' })
        const range = document.createRange(); range.selectNodeContents(element); const rect = range.getBoundingClientRect()
        return { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
      }, item.index)
      const pad = Math.max(4, box.height * 0.3), view = page.viewportSize()
      const clip = { x: Math.max(0, box.x - pad), y: Math.max(0, box.y - pad) }
      clip.width = Math.min(view.width, box.x + box.width + pad) - clip.x; clip.height = Math.min(view.height, box.y + box.height + pad) - clip.y
      if (clip.width < 8 || clip.height < 8) continue
      const image = `images/${host}-${records.length}.png`
      await page.screenshot({ path: `${OUT}/${image}`, clip, animations: 'disabled' })
      records.push({ origin, url: page.url(), capturedAt: new Date().toISOString(), text: item.text, image,
        font: { family: font.familyName, postScriptName: font.postScriptName, custom: font.isCustomFont, glyphs: font.glyphCount },
        css: { fontFamily: item.fontFamily, fontSize: item.fontSize, fontWeight: item.fontWeight, fontStyle: item.fontStyle, color: item.color } })
    }
  } finally { clearTimeout(deadline); await context.close().catch(() => {}) }
  return records
}

async function main() {
  await mkdir(`${OUT}/images`, { recursive: true })
  const done = new Set((await readFile(`${OUT}/done.txt`, 'utf8').catch(() => '')).split('\n').filter(Boolean))
  const queue = (await origins()).filter(origin => !done.has(origin))
  console.log(`${queue.length} origins to visit (top ${top})`)
  const browser = await chromium.launch(), agent = `${(await browser.newPage().then(async page => { const ua = await page.evaluate(() => navigator.userAgent); await page.close(); return ua }))} ${NAME}`
  const options = { userAgent: agent, viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 }
  let crops = 0
  await Promise.all(Array.from({ length: workers }, async () => {
    for (let origin; browser.isConnected() && (origin = queue.shift());) {
      const robots = await robotsFor(origin)
      let outcome
      if (robots === null) outcome = 'no answer'
      else if (!allowed(robots)) outcome = 'robots.txt refuses'
      else {
        try {
          const records = await capture(browser, options, origin)
          if (records.length) await appendFile(`${OUT}/records.jsonl`, records.map(record => JSON.stringify(record)).join('\n') + '\n')
          crops += records.length; outcome = `${records.length} crops`
        } catch (error) {
          if (!browser.isConnected()) break // stopped, not the site's failure: leave the origin for the next run
          outcome = `failed: ${error.message.split('\n')[0].slice(0, 80)}`
        }
      }
      await appendFile(`${OUT}/done.txt`, origin + '\n'); await appendFile(`${OUT}/visits.jsonl`, JSON.stringify({ origin, outcome }) + '\n')
      await wait(500)
    }
  }))
  await browser.close()
  console.log(`${crops} crops`)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
