import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { prepareInput } from '../src/input.mjs'
import { prepareLine } from '../src/line.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const manifest = JSON.parse(await readFile('.data/preparation/prepared.json')), raw = await readFile('.data/preparation/source.gray')
if (hash(raw) !== manifest.sha256 || hash(await readFile('src/line.mjs')) !== manifest.lineSha256) throw new Error('Changed coverage inputs')
const inside = (x,y,p) => p.every((a,i) => { const b=p[(i+1)%4]; return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]) >= -1e-6 })
const polygon = r => [[r.x,r.y],[r.x+r.width,r.y],[r.x+r.width,r.y+r.height],[r.x,r.y+r.height]]
const report = {}
for (const method of ['current','groups','deskew','deskew-windows','oracle']) {
  let total=0,retained=0,words=0,whole=0
  for (const sample of manifest.samples) {
    const gray=raw.subarray(sample.offset,sample.offset+sample.width*sample.height),data=new Uint8Array(gray.length*4),points=[]
    for(let i=0;i<gray.length;i++) {
      data.fill(gray[i],i*4,i*4+3);data[i*4+3]=255
      if(gray[i]<240)points.push([i%sample.width+.5,Math.floor(i/sample.width)+.5])
    }
    const image={...sample,data}
    const input=method==='current'?prepareInput(image):prepareLine(image,{deskew:method.startsWith('deskew'),sampler:method==='deskew-windows'?'windows':'groups',...(method==='oracle'?{angle:sample.angle,regions:sample.regions}:{})})
    const regions=input.windows.map(w=>w.polygon||polygon(w.rect))
    total+=points.length
    for(const [x,y] of points)if(regions.some(p=>inside(x,y,p)))retained++
    for(const word of sample.regions){
      const ink=points.filter(([x,y])=>inside(x,y,word))
      if(!ink.length)throw new Error('Empty renderer word region')
      words++
      if(regions.some(p=>ink.every(([x,y])=>inside(x,y,p))))whole++
    }
  }
  report[method]={sourceInkPixels:total,retainedSourceInkPixels:retained,retainedFraction:retained/total,words,wholeWordsInOneWindow:whole,wholeWordFraction:whole/words}
  console.log(method,report[method])
}
await writeFile('bench/preparation-coverage.json',JSON.stringify({manifestSha256:hash(await readFile('.data/preparation/prepared.json')),scope:'Geometric pixel-center coverage at gray < 240, including neighbor ink. Complete-word counts use renderer boxes and require one window containing every word ink pixel. Does not measure information lost by rescaling or neural recognition.',methods:report},null,2)+'\n')
