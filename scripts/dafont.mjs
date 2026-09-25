// Real "what font is this?" requests: the DaFont forum requests Max Halford collected for his LLM benchmark
// (https://maxhalford.github.io/llm-font-recognition/), each with the answer the forum confirmed. Names alone say which
// answers a shipped catalog holds; the pictures of those, cut by hand to the asked-about line (bench/dafont-crops.json),
// measure gpu-font beside the chatbots he scored. Pictures stay in .data/: they belong to the people who posted them.
import { readFile, writeFile, mkdir, access } from 'node:fs/promises'
import { matchPhotos, rankOf, catalogNames, normalize } from './photos.mjs'

const COMMIT = '810d681299104de5e5affded5b2f7cb7ef7d8918', RAW = `https://raw.githubusercontent.com/MaxHalford/llm-font-recognition/${COMMIT}/src/data`
const ROOT = '.data/dafont', AGENT = 'gpu-font benchmark (+https://github.com/dy/gpu-font)'
const exists = path => access(path).then(() => true, () => false)
async function fetched(url, path) {
  if (!await exists(path)) {
    const response = await fetch(url, { headers: { 'User-Agent': AGENT } })
    if (!response.ok) throw new Error(`Cannot load ${url}: HTTP ${response.status}`)
    await writeFile(path, new Uint8Array(await response.arrayBuffer()))
    await new Promise(done => setTimeout(done, 1000))  // one request a second
  }
  return readFile(path)
}

await mkdir(`${ROOT}/images`, { recursive: true })
const tasks = JSON.parse(await fetched(`${RAW}/tasks.json`, `${ROOT}/tasks.json`))
const guesses = JSON.parse(await fetched(`${RAW}/guesses.json`, `${ROOT}/guesses.json`))
const names = await catalogNames(), answered = tasks.filter(t => t.identified_font)
const indexed = answered.filter(t => names.has(normalize(t.identified_font)))

// Which answers each catalog holds, and the most requested answers none of them does.
const count = list => [...list.reduce((m, v) => m.set(v, (m.get(v) || 0) + 1), new Map())].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
const coverage = { requests: tasks.length, answered: answered.length, indexed: indexed.length,
  byCatalog: Object.fromEntries(count(indexed.map(t => names.get(normalize(t.identified_font)).catalog))),
  missing: count(answered.filter(t => !names.has(normalize(t.identified_font))).map(t => t.identified_font)).filter(([, n]) => n > 1).map(([font, n]) => ({ font, requests: n })) }

// The indexed requests with a crop, matched; each chatbot scored on the same requests by the same rule. Only their pictures
// are fetched: a crop is made by hand, so an indexed answer without one is listed, not downloaded.
const { rule, crops, skipped = {} } = JSON.parse(await readFile('bench/dafont-crops.json').catch(() => '{"crops":{}}'))
const cropped = indexed.filter(t => crops[t.task_id]), uncut = indexed.filter(t => !crops[t.task_id] && !skipped[t.task_id])
if (uncut.length) console.log(`${uncut.length} indexed requests have no crop in bench/dafont-crops.json: ${uncut.length > 40 ? uncut.slice(0, 40).map(t => t.task_id).join(' ') + ' …' : uncut.map(t => t.task_id).join(' ')}`)
if (!cropped.length) process.exit()
for (const t of cropped) t.path = `${ROOT}/images/${t.task_id}${new URL(t.img_url).pathname.match(/\.\w+$/)[0]}`, await fetched(t.img_url, t.path)
const { browser, adapter, results } = await matchPhotos(cropped.map(t => ({ path: t.path, box: crops[t.task_id] })))
const ours = cropped.map((t, i) => ({ task: t.task_id, font: t.identified_font, ...(results[i].status !== 'ok' && { status: results[i].status }), rank: rankOf(results[i].rows, t.identified_font), first: results[i].rows[0]?.name ?? null,
  score: results[i].rows[0] ? Math.round(results[i].rows[0].score * 1e3) / 1e3 : null, rows: results[i].rows }))
const share = (list, test) => list.length ? Math.round(1e4 * list.filter(test).length / list.length) / 1e4 : null
const scores = list => ({ requests: list.length, top1: share(list, r => r.rank === 1), top5: share(list, r => r.rank && r.rank <= 5) })
// Each chatbot answered a different share of the requests, so gpu-font is scored beside it on exactly those it answered.
const answers = new Map(answered.map(t => [t.task_id, t])), mine = new Map(ours.map(r => [r.task, r]))
const chatbots = Object.fromEntries(count(guesses.map(g => g.model_name)).map(([model]) => {
  const ranked = guesses.filter(g => g.model_name === model && answers.has(g.task_id)).map(g => {
    const candidates = [1, 2, 3, 4, 5].map(n => g[`candidate_font_${n}`]).filter(Boolean).map(normalize)
    const at = candidates.indexOf(normalize(answers.get(g.task_id).identified_font)); return { task: g.task_id, rank: at < 0 ? null : at + 1 }
  }), paired = ranked.filter(r => mine.has(r.task))
  return [model, { answered: scores(ranked), paired: { ...scores(paired), gpuFont: scores(paired.map(r => mine.get(r.task))) } }]
}))

await writeFile('bench/dafont.json', JSON.stringify({
  source: { benchmark: 'https://github.com/MaxHalford/llm-font-recognition', commit: COMMIT, forum: 'https://www.dafont.com/forum/?f=1' },
  coverage, crop: rule, skipped: Object.keys(skipped).length,
  rule: 'Right when the forum\'s answer, compared without case or punctuation, names a row the page shows or a family folded into it; chatbots by the same comparison.',
  catalogs: 'All: every shipped catalog searched as one.', browser, adapter, date: new Date().toISOString().slice(0, 10),
  gpuFont: scores(ours), chatbots,
  requests: ours.map(({ rows, ...r }) => r)
}, null, 1) + '\n')
// gpu-font's answers in the benchmark's own format, to send to its author.
await writeFile(`${ROOT}/gpu-font-guesses.json`, JSON.stringify(ours.map(r => ({ task_id: r.task, model_name: 'gpu-font',
  ...Object.fromEntries([1, 2, 3, 4, 5].map(n => [`candidate_font_${n}`, r.rows[n - 1]?.name ?? '']))})), null, 4) + '\n')
console.log(`${coverage.answered} answered requests, ${coverage.indexed} with an indexed answer, ${cropped.length} cropped`)
const pct = v => v === null ? '–' : `${(100 * v).toFixed(1)}%`
console.log(`gpu-font: top-1 ${pct(scores(ours).top1)}, top-5 ${pct(scores(ours).top5)} on ${ours.length}`)
for (const [model, { answered, paired }] of Object.entries(chatbots))
  console.log(`${model}: top-5 ${pct(answered.top5)} on ${answered.requests} answered; on its ${paired.requests} indexed ones ${pct(paired.top5)}, gpu-font ${pct(paired.gpuFont.top5)}`)
