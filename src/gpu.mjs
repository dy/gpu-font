import { channels, checkPixels } from './model.mjs'
import { unit } from './rank.mjs'

const bindings = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
`

function convolution(layer) {
  const w = 128 >> layer, h = 32 >> layer, ow = w / 2, oh = h / 2
  const ci = channels[layer], co = channels[layer + 1]
  return `${bindings}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= ${co * ow * oh}u) { return; }
  let x = i32(p % ${ow}u); let y = i32((p / ${ow}u) % ${oh}u); let c = p / ${ow * oh}u;
  var sum = bias[c];
  for (var k = 0u; k < ${ci}u; k++) {
    for (var dy = 0; dy < 3; dy++) { for (var dx = 0; dx < 3; dx++) {
      let ix = x * 2 + dx - 1; let iy = y * 2 + dy - 1;
      if (ix >= 0 && ix < ${w} && iy >= 0 && iy < ${h}) {
        let v = input[(k * ${h}u + u32(iy)) * ${w}u + u32(ix)];
        sum += ${layer === 0 ? '(1.0 - v)' : 'v'} * weight[((c * ${ci}u + k) * 3u + u32(dy)) * 3u + u32(dx)];
      }
    } }
  }
  output[p] = max(sum, 0.0);
}`
}
const projection = `${bindings}
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x; if (d >= 32u) { return; }
  var sum = bias[d];
  for (var c = 0u; c < 48u; c++) {
    var pooled = 0.0;
    for (var p = 0u; p < 16u; p++) { pooled += input[c * 16u + p] / 16.0; }
    sum += pooled * weight[d * 48u + c];
  }
  output[d] = sum;
}`

export async function createGPU(model, gpu = globalThis.navigator?.gpu) {
  if (!gpu) throw new Error('WebGPU is unavailable in this browser')
  const adapter = await gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter is available')
  const device = await adapter.requestDevice()
  const owned = []
  let lost = false, stopped = false, tail = Promise.resolve()
  device.lost.then(() => { lost = true })
  function buffer(size, usage, values) {
    const b = device.createBuffer({ size, usage, mappedAtCreation: !!values })
    owned.push(b)
    if (values) { new Float32Array(b.getMappedRange()).set(values); b.unmap() }
    return b
  }
  const destroy = () => { stopped = true; owned.forEach(b => b.destroy()); device.destroy() }
  try {
    const input = buffer(4096 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    const readback = buffer(32 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)
    const stages = []
    let previous = input
    for (let stage = 0; stage < 5; stage++) {
      const count = stage === 4 ? 32 : channels[stage + 1] * (128 >> (stage + 1)) * (32 >> (stage + 1))
      const name = stage === 4 ? 'project' : `features.${stage * 2}`
      const output = buffer(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
      const weights = model.tensors[`${name}.weight`], bias = model.tensors[`${name}.bias`]
      const weightBuffer = buffer(weights.byteLength, GPUBufferUsage.STORAGE, weights)
      const biasBuffer = buffer(bias.byteLength, GPUBufferUsage.STORAGE, bias)
      const module = device.createShaderModule({ code: stage === 4 ? projection : convolution(stage) })
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [previous, weightBuffer, biasBuffer, output].map((b, binding) => ({ binding, resource: { buffer: b } })) })
      stages.push({ pipeline, group, count }); previous = output
    }
    async function run(pixels) {
      if (stopped || lost) throw new Error('WebGPU device is no longer available')
      device.queue.writeBuffer(input, 0, pixels)
      const commands = device.createCommandEncoder()
      for (const { pipeline, group, count } of stages) {
        const pass = commands.beginComputePass()
        pass.setPipeline(pipeline); pass.setBindGroup(0, group)
        pass.dispatchWorkgroups(Math.ceil(count / (count === 32 ? 32 : 64)))
        pass.end()
      }
      commands.copyBufferToBuffer(previous, 0, readback, 0, 128)
      device.queue.submit([commands.finish()])
      try {
        await readback.mapAsync(GPUMapMode.READ)
        return unit(new Float32Array(readback.getMappedRange()).slice())
      } finally {
        if (readback.mapState === 'mapped') readback.unmap()
      }
    }
    return {
      info: adapter.info?.description || adapter.info?.architecture || 'WebGPU adapter',
      encode(pixels) {
        checkPixels(pixels)
        // Snapshot at call time and serialize use of the shared readback buffer.
        const copy = pixels.slice()
        const next = tail.then(() => run(copy))
        tail = next.catch(() => {})
        return next
      },
      destroy,
    }
  } catch (error) { destroy(); throw error }
}
