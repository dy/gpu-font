import test from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { catalogs } from '../src/match.mjs'

const root = new URL('../', import.meta.url), relative = url => url.href.slice(root.href.length)

// Relative imports from a module, followed to the end.
async function reachable(path, seen = new Set()) {
  if (seen.has(path)) return seen
  seen.add(path)
  for (const [, spec] of (await readFile(new URL(path, root), 'utf8')).matchAll(/from '(\.[^']+)'/g)) await reachable(relative(new URL(spec, new URL(path, root))), seen)
  return seen
}

test('the npm package holds the entry, every module it imports, its types and every catalog it names, and no other code', async () => {
  const [{ files }] = JSON.parse(execFileSync('npm', ['pack', '--dry-run', '--json', '--ignore-scripts'], { cwd: fileURLToPath(root), encoding: 'utf8' }))
  const packed = new Set(files.map(f => f.path)), modules = await reachable('src/match.mjs')
  const data = [new URL('models/encoder/encoder.json', root), ...Object.values(catalogs)].map(relative)
  for (const path of [...modules, 'src/match.d.ts', ...data, 'README.md', 'LICENSE', 'package.json']) assert.ok(packed.has(path), `${path} is not packed`)
  assert.deepEqual([...packed].filter(path => path.endsWith('.mjs')).sort(), [...modules].sort())
})

test('the types list the catalogs the module exports', async () => {
  const types = await readFile(new URL('src/match.d.ts', root), 'utf8')
  assert.deepEqual([...types.match(/catalogs: Record<([^>]+)>/)[1].matchAll(/'([^']+)'/g)].map(m => m[1]), Object.keys(catalogs))
})
