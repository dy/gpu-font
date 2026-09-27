import test from 'node:test'
import assert from 'node:assert/strict'
import { allowed } from '../scripts/web_screens.mjs'

test('no robots.txt, or rules for other paths, leave the homepage open', () => {
  assert.equal(allowed(''), true)
  assert.equal(allowed('User-agent: *\nDisallow: /private\nDisallow: /search?'), true)
})

test('a refusal of every agent or of ClaudeBot closes the homepage', () => {
  assert.equal(allowed('User-agent: *\nDisallow: /'), false)
  assert.equal(allowed('User-agent: *\nDisallow: /*'), false)
  assert.equal(allowed('User-agent: *\nAllow: /\n\nUser-agent: ClaudeBot\nDisallow: /'), false)
  assert.equal(allowed('# comment\nUser-agent: GPTBot\nUser-agent: ClaudeBot  # AI crawlers\nDisallow: /'), false) // two agents share one group
})

test('a group naming another bot does not bind this tool', () => {
  assert.equal(allowed('User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nAllow: /'), true)
})

test('the longest matching rule wins, and a tie is read as a refusal', () => {
  assert.equal(allowed('User-agent: *\nDisallow: /\nAllow: /$'), true)
  assert.equal(allowed('User-agent: *\nAllow: /\nDisallow: /'), false)
})
