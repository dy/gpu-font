import test from 'node:test'
import assert from 'node:assert/strict'
import { prepareBatch } from '../scripts/corpus-prepare.mjs'
import { prepareLine } from '../src/line.mjs'
import { readNetwork, inferCPU, corpusArchitecture, largeArchitecture } from '../src/network.mjs'

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

for (const [architecture, channels] of [[corpusArchitecture,[1,32,64,96,128,128]], [largeArchitecture,[1,64,128,192,256,256]]]) test(`${architecture} uses every feature and the last of 2,055 output rows`, () => {
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
