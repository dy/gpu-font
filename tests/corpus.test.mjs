import test from 'node:test'
import assert from 'node:assert/strict'
import { prepareBatch } from '../scripts/corpus-prepare.mjs'
import { prepareLine } from '../src/line.mjs'
import { readNetwork, inferCPU, corpusArchitecture, largeArchitecture, widerArchitecture } from '../src/network.mjs'

test('corpus batch preparation is byte-identical to browser input, including blank and repeated batches', () => {
  const width = 35, height = 20, gray = Buffer.alloc(width * height, 255)
  for (let y = 3; y < 17; y++) for (const x of [4,5,9,10,20,21,26,27]) gray[y*width+x] = 0
  const source = { width, height, pixels: gray.toString('base64') }, data = new Uint8Array(gray.length*4)
  for (let i=0;i<gray.length;i++) { data.fill(gray[i],i*4,i*4+3); data[i*4+3]=255 }
  const expected = prepareLine({width,height,data}, {deskew:true,sampler:'windows'}).windows.map(w=>({width:w.width,height:w.height,pixels:Buffer.from(Uint8Array.from(w.pixels,p=>Math.round(p*255))).toString('base64')}))
  assert.ok(expected.length)
  const blank = {width:1,height:1,pixels:Buffer.from([255]).toString('base64')}
  assert.deepEqual(prepareBatch([]), [])
  for (const batch of [[source], [source,source], [source,blank,source]]) assert.deepEqual(prepareBatch(batch), batch.map(s=>s===source?expected:[]))
  assert.deepEqual(prepareBatch([blank]), [[]])
  for (const pixels of ['', gray.subarray(0,-1).toString('base64'), Buffer.concat([gray,Buffer.from([255])]).toString('base64'), '!']) assert.throws(()=>prepareBatch([{...source,pixels}]))
  for (const invalid of [null, {}, [null], [{...source,width:0}], [{...source,height:1.5}]]) assert.throws(()=>prepareBatch(invalid))
  assert.deepEqual(prepareBatch([source]), [expected])
})

for (const [architecture, channels] of [[corpusArchitecture,[1,32,64,96,128,128]], [largeArchitecture,[1,64,128,192,256,256]], [widerArchitecture,[1,96,192,288,384,384]]]) test(`${architecture} uses every feature and the last of 2,055 output rows`, () => {
  const features=channels.at(-1), fonts=Array.from({length:2055},(_,i)=>`f${i}`)
  const layers=Array.from({length:6},(_,i)=>{
    const shape=i===5?[fonts.length,features]:[channels[i+1],channels[i],3,3]
    return {shape,scale:Array(shape[0]).fill(1),bias:Array(shape[0]).fill(0),weights:Buffer.alloc(shape.reduce((a,b)=>a*b)).toString('base64')}
  })
  layers[4].bias[features-1]=3
  const packed=Buffer.from(layers[5].weights,'base64'); packed[packed.length-1]=7; layers[5].weights=packed.toString('base64')
  const model=readNetwork({version:1,architecture,fonts,preparation:{width:128,height:48,windows:3},layers})
  for (const [width,height] of [[1,1],[1,1],[7,3],[1,1]]) {
    const scores=inferCPU(model,{width,height,pixels:new Float32Array(width*height)})
    assert.equal(scores.length,2055); assert.equal(scores.at(-1),21); assert.ok(scores.slice(0,-1).every(v=>v===0))
  }
})

test('an RGB image with readings is read as the page reads a crop: coloured letters on a coloured wall give the line as it stands and cleaned, each with windows; a two-tone one only as it stands', () => {
  const w = 120, h = 40, rgb = paint => { const b = Buffer.alloc(w * h * 3); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) b.set(paint(x, y), 3 * (y * w + x)); return { width: w, height: h, channels: 3, readings: true, pixels: b.toString('base64') } }
  const bars = (x, y) => y >= 10 && y < 30 && x >= 10 && x < 110 && x % 12 < 7
  let seed = 3; const noise = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31
  const wall = prepareBatch([rgb((x, y) => bars(x, y) ? [20, 20, 20] : [100 + 150 * noise(), 120, 140])])[0]
  assert.deepEqual(wall.map(r => [r.kind, r.line]), [['as it is', 0], ['cleaned', 0]])
  for (const r of wall) { assert.ok(r.windows.length); for (const win of r.windows) assert.equal(Buffer.from(win.pixels, 'base64').length, win.width * win.height) }
  const plain = prepareBatch([rgb((x, y) => bars(x, y) ? [0, 0, 0] : [255, 255, 255])])[0]
  assert.deepEqual(plain.map(r => r.kind), ['as it is'])
  assert.throws(() => prepareBatch([{ ...rgb(() => [0, 0, 0]), channels: 2 }]))
  assert.throws(() => prepareBatch([{ ...rgb(() => [0, 0, 0]), channels: 1 }]), 'RGB bytes declared grayscale are trailing pixels')
})
