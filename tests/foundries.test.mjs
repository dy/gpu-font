import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { validateRegistry } from '../scripts/foundries.mjs'

const entry = (patch = {}) => ({
  id: 'example', name: 'Example Foundry', url: 'https://example.invalid/', kind: 'foundry', access: 'previews',
  licence: 'commercial', terms: { status: 'none-found', checked: '2026-09-23' }, work: { status: 'excluded' },
  log: [{ date: '2026-09-23', event: 'Surveyed' }], ...patch,
})

test('the committed foundry registry is valid', async () => {
  const registry = JSON.parse(await readFile('bench/foundries.json', 'utf8'))
  assert.deepEqual(validateRegistry(registry), [])
  assert.ok(registry.sources.length > 40)
})

test('a ban needs its quoted clause, and blocks work without recorded permission', () => {
  assert.deepEqual(validateRegistry({ sources: [entry()] }), [])
  const unquoted = validateRegistry({ sources: [entry({ terms: { status: 'ban' } })] })
  assert.match(unquoted.join(), /quoted clause/)
  const banned = { status: 'ban', quote: 'no robots', url: 'https://example.invalid/terms' }
  assert.match(validateRegistry({ sources: [entry({ terms: banned, work: { status: 'planned' } })] }).join(), /without recorded permission/)
  assert.deepEqual(validateRegistry({ sources: [entry({ terms: banned, work: { status: 'indexed', permission: 'email 2026-10-01' } })] }), [])
  assert.deepEqual(validateRegistry({ sources: [entry({ terms: banned, work: { status: 'permission-to-request' } })] }), [])
})

test('ids, vocabularies, URLs and the log are checked', () => {
  const problems = validateRegistry({ sources: [
    entry(), entry(),
    entry({ id: 'Bad Id', url: 'http://x', kind: 'shop', access: 'maybe', work: { status: 'soon' }, log: [] }),
  ] })
  for (const pattern of [/duplicate id/, /lowercase slug/, /must be https/, /kind must be/, /access must be/, /work\.status must be/, /dated events/])
    assert.ok(problems.some(problem => pattern.test(problem)), `${pattern} in ${problems.join('; ')}`)
  assert.deepEqual(validateRegistry({}), ['sources must be an array'])
})
