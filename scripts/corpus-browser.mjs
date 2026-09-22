// Independent Chromium holdout, prepared through the same code as training.
import { readFile, writeFile, mkdir, open } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { chromium } from 'playwright'
import { prepareBatch } from './corpus-prepare.mjs'

const read = async p => JSON.parse(await readFile(p)), hash = b => createHash('sha256').update(b).digest('hex')
const dir = '.data/corpus', catalog = await read('bench/corpus.json'), families = catalog.families.filter(f => !f.excluded)
const pins = { catalog: hash(await readFile('bench/corpus.json')), runner: hash(await readFile('scripts/corpus-browser.mjs')), preparation: hash(await readFile('scripts/corpus-prepare.mjs')), line: hash(await readFile('src/line.mjs')), input: hash(await readFile('src/input.mjs')), normalizer: hash(await readFile('src/prepare.mjs')) }
await mkdir(`${dir}/browser`, { recursive: true })
const browser = await chromium.launch()
let cursor = 0, done = 0
try {
  await Promise.all(Array.from({ length: 4 }, async () => {
    while (cursor < families.length) {
      const label = cursor++, family = families[label], prefix = `${dir}/browser/${family.id}`
      const source = await read(`${dir}/shards/${family.id}.json`)
      if (source.pins['bench/corpus.json'] !== pins.catalog) throw new Error('Stale corpus text plan')
      const plans = source.samples.filter(s => s.role === 'test' && s.condition === 'clean')
      let old
      try { old = await read(`${prefix}.json`) } catch (error) { if (error.code !== 'ENOENT') throw error }
      if (old && JSON.stringify(old.pins) === JSON.stringify(pins) && hash(await readFile(`${prefix}.u8`)) === old.sha256) { done++; continue }
      const bytes = await readFile(`.data/google/${catalog.commit}/${family.selected}`)
      const face = family.faces.find(f => f.path === family.selected)
      if (createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex') !== face.blob) throw new Error('Changed source font')
      const page = await browser.newPage(), samples = [], windows = [], chunks = []
      let offset = 0
      try {
        await page.route('**/corpus-check', route => route.fulfill({contentType:'text/html',body:'<title>Corpus check</title>'}))
        await page.route('**/corpus-font.ttf', route => route.fulfill({contentType:'font/ttf',body:bytes}))
        await page.goto('http://localhost:4179/corpus-check')
        const images = await page.evaluate(async plans => {
          const face = await new FontFace('query', 'url(/corpus-font.ttf)').load(); document.fonts.add(face)
          const result = []
          for (const plan of plans) {
            const canvas = document.createElement('canvas'); let context = canvas.getContext('2d')
            context.font = `${plan.size}px query`; const box = context.measureText(plan.text)
            canvas.width = Math.ceil(box.actualBoundingBoxLeft + box.actualBoundingBoxRight) + 12
            canvas.height = Math.ceil(box.actualBoundingBoxAscent + box.actualBoundingBoxDescent) + 12
            context = canvas.getContext('2d'); context.font = `${plan.size}px query`
            context.fillStyle = 'white'; context.fillRect(0,0,canvas.width,canvas.height); context.fillStyle = 'black'
            context.fillText(plan.text,6+box.actualBoundingBoxLeft,6+box.actualBoundingBoxAscent)
            for (const condition of ['clean','rotate','small']) {
              const out = document.createElement('canvas'), scale = condition === 'small' ? .45 : 1, angle = condition === 'rotate' ? Math.PI/30 : 0
              out.width = Math.max(1,Math.ceil(scale*(canvas.width*Math.cos(angle)+canvas.height*Math.sin(angle))))
              out.height = Math.max(1,Math.ceil(scale*(canvas.height*Math.cos(angle)+canvas.width*Math.sin(angle))))
              const c = out.getContext('2d'); c.fillStyle='white'; c.fillRect(0,0,out.width,out.height)
              c.translate(out.width/2,out.height/2); c.rotate(angle); c.scale(scale,scale); c.drawImage(canvas,-canvas.width/2,-canvas.height/2)
              const data = c.getImageData(0,0,out.width,out.height).data; let pixels=''
              for (let i=0;i<data.length;i+=4) pixels+=String.fromCharCode(plan.light ? 255-data[i] : data[i])
              result.push({width:out.width,height:out.height,pixels:btoa(pixels),plan,condition})
            }
          }
          return result
        }, plans)
        const prepared = prepareBatch(images)
        for (const [source, image] of images.entries()) {
          samples.push({...image.plan,condition:image.condition,label,family:family.id,renderer:'chromium'})
          if (!prepared[source].length) throw new Error(`Blank Chromium input: ${family.id}`)
          for (const w of prepared[source]) {
            const bytes=Buffer.from(w.pixels,'base64'); chunks.push(bytes)
            windows.push({width:w.width,height:w.height,offset,source}); offset+=bytes.length
          }
        }
        const packed=Buffer.concat(chunks); await writeFile(`${prefix}.u8`,packed)
        await writeFile(`${prefix}.json`,JSON.stringify({pins,sha256:hash(packed),samples,windows,browser:browser.version()}))
      } finally { await page.close() }
      if (++done % 50 === 0 || done === families.length) console.log(`Chromium holdout ${done}/${families.length}`)
    }
  }))
} finally { await browser.close() }
const output=await open(`${dir}/browser.u8`,'w'), samples=[], windows=[]
let offset=0
try {
  for (const family of families) {
    const prefix=`${dir}/browser/${family.id}`, shard=await read(`${prefix}.json`), bytes=await readFile(`${prefix}.u8`)
    if (JSON.stringify(shard.pins)!==JSON.stringify(pins) || hash(bytes)!==shard.sha256) throw new Error('Changed browser shard')
    const owner=samples.length; let end=0
    for (const w of shard.windows) {
      if (w.offset!==end || w.source<0 || w.source>=shard.samples.length) throw new Error('Invalid browser window')
      end+=w.width*w.height; windows.push({...w,offset:w.offset+offset,source:w.source+owner})
    }
    if (end!==bytes.length) throw new Error('Truncated/trailing browser pixels')
    samples.push(...shard.samples); await output.writeFile(bytes); offset+=bytes.length
  }
} finally { await output.close() }
await writeFile(`${dir}/browser.json`,JSON.stringify({fonts:families.map(f=>f.id),pins,sha256:hash(await readFile(`${dir}/browser.u8`)),samples,windows}))
console.log(`${samples.length} independent Chromium queries`)
