import test from 'node:test'
import assert from 'node:assert/strict'
import { prepareInput } from '../src/input.mjs'
import { prepareLine } from '../src/line.mjs'

function fixture(width=120,height=30,boxes=[[5,5,7,20],[18,5,7,20],[50,5,8,20],[65,5,8,20],[100,5,8,20]]) {
  const data=new Uint8Array(width*height*4).fill(255)
  for(const [x,y,w,h] of boxes)for(let r=y;r<y+h;r++)for(let c=x;c<x+w;c++)data.fill(0,(r*width+c)*4,(r*width+c)*4+3)
  return {width,height,data}
}
test('word grouping retains every ink pixel without cutting glyph columns',()=>{
  const image=fixture(), grouped=prepareLine(image)
  assert.equal(grouped.windows.length,3)
  assert.equal(grouped.angle,0)
  for(let y=0;y<image.height;y++)for(let x=0;x<image.width;x++)if(image.data[(y*image.width+x)*4]===0)assert.ok(grouped.windows.some(({rect:r})=>x>=r.x&&x<r.x+r.width&&y>=r.y&&y<r.y+r.height))
  for(const {rect:r,pixels,width,height} of grouped.windows){
    assert.ok(width<=128&&height<=48)
    for(const x of [r.x,r.x+r.width-1])for(let y=r.y;y<r.y+r.height;y++)assert.equal(image.data[(y*image.width+x)*4],255)
    for(const p of pixels)assert.equal(p,Math.fround(Math.round(p*255)/255))
  }
  const one=prepareLine(image,{windows:1}).windows
  assert.equal(one.length,1)
  assert.deepEqual(one[0].rect,{x:4,y:4,width:105,height:22})
})
test('line preparation keeps detached dots, no-op deskew, and A → A → B → blank → A independent',()=>{
  const a=fixture(), b=fixture(22,40,[[8,3,4,3],[8,12,4,20]])
  const first=prepareLine(a), saved=first.windows[0].pixels.slice()
  assert.deepEqual(prepareLine(a,{deskew:true}),first)
  assert.deepEqual(prepareLine(a),first)
  const dot=prepareLine(b)
  assert.equal(dot.windows.length,1)
  assert.deepEqual(dot.windows[0].rect,{x:7,y:2,width:6,height:31})
  assert.deepEqual(prepareLine(fixture(1,1,[])),{status:'blank',windows:[],angle:0})
  assert.deepEqual(prepareLine(a),first)
  first.windows[0].pixels.fill(.5)
  assert.deepEqual(prepareLine(a).windows[0].pixels,saved)
})
test('line preparation rejects malformed geometry and oracle regions before blank-image early return',()=>{
  const image=fixture(1,1,[])
  for(const invalid of [null,{}, {...image,data:new Uint8Array(3)}])assert.throws(()=>prepareLine(invalid))
  for(const options of [{width:129},{height:0},{windows:0},{windows:4},{windows:1.5},{deskew:1},{angle:NaN},{angle:9},{regions:[]},{regions:[null]},{regions:[[[0,0],[1,0],[1,1],[0,NaN]]]}])assert.throws(()=>prepareLine(image,options))
  const smallest=prepareLine(fixture(2,1,[[0,0,1,1]]))
  assert.equal(smallest.status,'ok')
  assert.ok(smallest.windows[0].pixels.some(p=>p<1))
})
test('known-angle resampling exports the inverse source polygon and bounded exact tensors',()=>{
  const image=fixture()
  for(const angle of [-8,-6,6,8]){
    const result=prepareLine(image,{angle})
    assert.equal(result.angle,angle)
    for(const input of result.windows){
      assert.equal(input.polygon.length,4)
      assert.equal(input.pixels.length,input.width*input.height)
      assert.ok(input.pixels.every(p=>Number.isFinite(p)&&p>=0&&p<=1))
      assert.ok(input.rect.x>=0&&input.rect.y>=0&&input.rect.x+input.rect.width<=image.width&&input.rect.y+input.rect.height<=image.height)
      const [a,b]=input.polygon
      assert.ok(Math.abs(Math.atan2(b[1]-a[1],b[0]-a[0])*180/Math.PI-angle)<1e-10)
    }
  }
})

test('deskew with unchanged angle preserves every deployed tensor and region exactly',()=>{
  const image=fixture(), original=prepareInput(image)
  const result=prepareLine(image,{deskew:true,sampler:'windows'})
  assert.equal(result.angle,0)
  assert.deepEqual(result.windows,original.windows)
})

test('automatic deskew recovers both baseline directions and agrees with the oracle tensors',()=>{
  const base=fixture(180,70,Array.from({length:7},(_,i)=>[15+i*23,25,10,20]))
  for(const angle of [-6,6]){
    const data=new Uint8Array(base.data.length).fill(255), a=angle*Math.PI/180, c=Math.cos(a), s=Math.sin(a)
    for(let y=0;y<base.height;y++)for(let x=0;x<base.width;x++){
      const sx=Math.round(c*(x-90)+s*(y-35)+90), sy=Math.round(-s*(x-90)+c*(y-35)+35)
      if(sx>=0&&sy>=0&&sx<base.width&&sy<base.height&&base.data[(sy*base.width+sx)*4]===0)data.fill(0,(y*base.width+x)*4,(y*base.width+x)*4+3)
    }
    const image={...base,data}, automatic=prepareLine(image,{deskew:true,sampler:'windows'})
    assert.ok(Math.abs(automatic.angle-angle)<=.5,`Estimated ${automatic.angle} for ${angle}`)
    assert.deepEqual(automatic,prepareLine(image,{angle:automatic.angle,sampler:'windows'}))
    assert.equal(automatic.windows.length,3)
    assert.ok(automatic.windows.every(w=>w.pixels.some(p=>p===0)&&w.pixels.some(p=>p===1)))
    assert.equal(prepareLine(base,{deskew:true,sampler:'windows'}).angle,0)
  }
})

test('a crop that touches the text prepares as one with a margin, regions mapped back to the crop', () => {
  const boxes = [[0, 0, 7, 20], [13, 0, 7, 20], [45, 0, 8, 20], [60, 0, 8, 20], [95, 0, 8, 20]]
  const tight = fixture(103, 20, boxes), margin = fixture(123, 40, boxes.map(([x, y, w, h]) => [x + 10, y + 10, w, h]))
  for (const sampler of ['groups', 'windows']) {
    const a = prepareLine(tight, { sampler }), b = prepareLine(margin, { sampler })
    assert.deepEqual(a.windows.map(w => w.pixels), b.windows.map(w => w.pixels), sampler)
    for (const [i, window] of a.windows.entries()) {
      const r = b.windows[i].rect, x = Math.max(0, r.x - 10), y = Math.max(0, r.y - 10)
      assert.deepEqual(window.rect, { x, y, width: Math.min(tight.width, r.x - 10 + r.width) - x, height: Math.min(tight.height, r.y - 10 + r.height) - y }, sampler)
      if (window.polygon) assert.deepEqual(window.polygon, b.windows[i].polygon.map(([px, py]) => [px - 10, py - 10]), sampler)
    }
  }
  // Word regions given in the crop's coordinates move with it into the frame.
  const regions = [[[0, 0], [20, 0], [20, 20], [0, 20]], [[45, 0], [68, 0], [68, 20], [45, 20]]]
  const a = prepareLine(tight, { regions }), b = prepareLine(margin, { regions: regions.map(points => points.map(([x, y]) => [x + 10, y + 10])) })
  assert.equal(a.windows.length, 2)
  assert.deepEqual(a.windows.map(w => w.pixels), b.windows.map(w => w.pixels))
  assert.deepEqual(a.windows.map(w => w.polygon), b.windows.map(w => w.polygon.map(([x, y]) => [x - 10, y - 10])))
})
