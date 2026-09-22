import { checkInput } from './network.mjs'

const bindings = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> shape: vec4<u32>;
`
function shader(layer, model) {
  const { channels, strides, dilations } = model, classes = model.fonts.length, dilation = dilations[layer]
  if (layer === strides.length) return `${bindings}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x; if (d >= ${classes}u) { return; }
  let count = shape.x * shape.y; var sum = bias[d];
  for (var c = 0u; c < 64u; c++) {
    var pooled = 0.0;
    for (var p = 0u; p < count; p++) { pooled += input[c * count + p]; }
    sum += pooled / f32(count) * weights[d * 64u + c];
  }
  output[d] = sum;
}`
  return `${bindings}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x; let count = shape.z * shape.w;
  if (p >= ${channels[layer + 1]}u * count) { return; }
  let x = i32(p % shape.z); let y = i32((p / shape.z) % shape.w); let c = p / count;
  var sum = bias[c];
  for (var k = 0u; k < ${channels[layer]}u; k++) {
    for (var dy = 0; dy < 3; dy++) { for (var dx = 0; dx < 3; dx++) {
      let ix = x * ${strides[layer]} + (dx - 1) * ${dilation}; let iy = y * ${strides[layer]} + (dy - 1) * ${dilation};
      if (ix >= 0 && ix < i32(shape.x) && iy >= 0 && iy < i32(shape.y)) {
        let v = input[(k * shape.y + u32(iy)) * shape.x + u32(ix)];
        sum += ${layer === 0 ? '(1.0 - v)' : 'v'} * weights[((c * ${channels[layer]}u + k) * 3u + u32(dy)) * 3u + u32(dx)];
      }
    } }
  }
  output[p] = max(sum, 0.0);
}`
}

export async function createNetworkGPU(model, gpu = globalThis.navigator?.gpu) {
  const { channels, strides } = model, depth = strides.length
  if (!gpu) throw new Error('WebGPU is unavailable')
  const adapter = await gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')
  const device = await adapter.requestDevice(), owned = []
  let stopped = false, lost = false, tail = Promise.resolve()
  device.lost.then(() => { lost = true })
  function buffer(size, usage, values) {
    const b = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage, mappedAtCreation: !!values })
    owned.push(b)
    if (values) { new Float32Array(b.getMappedRange()).set(values); b.unmap() }
    return b
  }
  const destroy = () => { stopped = true; owned.forEach(b => b.destroy()); device.destroy() }
  try {
    const input = buffer(128 * 48 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    const readback = buffer(model.fonts.length * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)
    const stages = []
    let previous = input, width = 128, height = 48
    for (let i = 0; i <= depth; i++) {
      const w = i === depth ? model.fonts.length : Math.ceil(width / strides[i])
      const h = i === depth ? 1 : Math.ceil(height / strides[i])
      const output = buffer(w * h * (i === depth ? 1 : channels[i + 1]) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
      const layer = model.layers[i]
      const weights = buffer(layer.weights.byteLength, GPUBufferUsage.STORAGE, layer.weights)
      const bias = buffer(layer.bias.byteLength, GPUBufferUsage.STORAGE, layer.bias)
      const shape = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      const module = device.createShaderModule({ code: shader(i, model) })
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [previous, weights, bias, output, shape].map((buffer, binding) => ({ binding, resource: { buffer } })) })
      stages.push({ shape, pipeline, group }); previous = output; width = w; height = h
    }
    async function run(window) {
      if (stopped || lost) throw new Error('WebGPU device is no longer available')
      device.queue.writeBuffer(input, 0, window.pixels)
      let { width, height } = window
      const commands = device.createCommandEncoder()
      for (const [i, stage] of stages.entries()) {
        const w = i === depth ? model.fonts.length : Math.ceil(width / strides[i]), h = i === depth ? 1 : Math.ceil(height / strides[i])
        device.queue.writeBuffer(stage.shape, 0, new Uint32Array([width, height, w, h]))
        const pass = commands.beginComputePass()
        pass.setPipeline(stage.pipeline); pass.setBindGroup(0, stage.group)
        pass.dispatchWorkgroups(Math.ceil(w * h * (i === depth ? 1 : channels[i + 1]) / 64)); pass.end()
        width = w; height = h
      }
      commands.copyBufferToBuffer(previous, 0, readback, 0, model.fonts.length * 4)
      device.queue.submit([commands.finish()])
      try {
        await readback.mapAsync(GPUMapMode.READ)
        return new Float32Array(readback.getMappedRange()).slice()
      } finally { if (readback.mapState === 'mapped') readback.unmap() }
    }
    return {
      info: adapter.info?.description || adapter.info?.architecture || 'WebGPU adapter',
      infer(window) {
        checkInput(window)
        const snapshot = { width: window.width, height: window.height, pixels: window.pixels.slice() }
        const result = tail.then(() => run(snapshot)); tail = result.catch(() => {})
        return result
      }, destroy,
    }
  } catch (error) { destroy(); throw error }
}
