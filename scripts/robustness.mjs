import { readFile, writeFile, open, mkdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { prepare } from '../src/prepare.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
function python(module, ...args) {
  const result = spawnSync(process.execPath, ['scripts/python.mjs', '-m', module, ...args], { stdio: 'inherit' })
  if (result.error) throw result.error
  if (result.status !== 0) throw new Error(`Robustness ${args.join(" ")} failed`)
}
python('train.robustness', 'render')
const dir = '.data/robustness'
const config = JSON.parse(await readFile('bench/robustness.json'))
const manifest = JSON.parse(await readFile(`${dir}/samples.json`))
for (const [name, input] of Object.entries(config.inputs)) {
  const folder = `${dir}/${name}`
  await mkdir(folder, { recursive: true })
  const raw = await open(`${dir}/source.rgba`)
  const count = input.width * input.height
  const binary = Buffer.alloc(manifest.samples.length * count * 4)
  const rawHash = createHash('sha256')
  let offset = 0
  try {
    for (const [i, sample] of manifest.samples.entries()) {
      const data = Buffer.alloc(sample.width * sample.height * 4)
      if (sample.offset !== offset) throw new Error('Invalid raw sample offset')
      const { bytesRead } = await raw.read(data, 0, data.length, offset)
      if (bytesRead !== data.length) throw new Error('Truncated raw image')
      rawHash.update(data)
      offset += data.length
      const prepared = prepare({ ...sample, data }, input)
      if (prepared.status !== 'ok') throw new Error(`Blank augmentation: ${sample.family}/${i}`)
      for (let p = 0; p < count; p++) binary.writeFloatLE(prepared.pixels[p], (i * count + p) * 4)
    }
    if ((await raw.stat()).size !== offset) throw new Error('Trailing raw image bytes')
    if (rawHash.digest('hex') !== manifest.rawSha256) throw new Error('Raw image checksum mismatch')
  } finally { await raw.close() }
  await writeFile(`${folder}/prepared.f32`, binary)
  await writeFile(`${folder}/prepared.json`, JSON.stringify({ ...manifest, ...input,
    preparationSha256: hash(await readFile('src/prepare.mjs')), tensorSha256: hash(binary) }) + '\n')
  console.log(`Prepared ${manifest.samples.length} images at ${input.width} × ${input.height} through the shared normalizer.`)
  python('train.robustness', 'train', name)
}
python('scripts.robustness_preview')
