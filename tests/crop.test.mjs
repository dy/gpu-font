import test from 'node:test'
import assert from 'node:assert/strict'
import { editCrop, newCrop } from '../demo/crop.mjs'

const bounds = { width: 100, height: 60 }, rect = { x: 20, y: 10, width: 40, height: 30 }
test('each crop handle retains its opposite edge; moves retain size and clamp to the image', () => {
  const expected = {
    nw: [25, 13, 35, 27], n: [20, 13, 40, 27], ne: [20, 13, 45, 27], e: [20, 10, 45, 30],
    se: [20, 10, 45, 33], s: [20, 10, 40, 33], sw: [25, 10, 35, 33], w: [25, 10, 35, 30],
  }
  for (const [handle, values] of Object.entries(expected)) assert.deepEqual(Object.values(editCrop(rect, handle, 5, 3, bounds)), values)
  assert.deepEqual(editCrop(rect, 'move', -999, 999, bounds), { ...rect, x: 0, y: 30 })
  assert.deepEqual(editCrop(rect, 'move', 999, -999, bounds), { ...rect, x: 60, y: 0 })
  for (const handle of Object.keys(expected)) for (const d of [-999, 0, 999]) {
    const r = editCrop(rect, handle, d, d, bounds)
    assert.ok(r.width >= 1 && r.height >= 1 && r.x >= 0 && r.y >= 0 && r.x + r.width <= 100 && r.y + r.height <= 60)
  }
  assert.deepEqual(rect, { x: 20, y: 10, width: 40, height: 30 })
})
test('one-pixel crops, reverse selection, and repeated gestures never accumulate drift', () => {
  const one = { x: 0, y: 0, width: 1, height: 1 }
  for (const handle of ['move', 'nw', 'se']) assert.deepEqual(editCrop(one, handle, 0, 0, one), one)
  assert.deepEqual(newCrop({ x: 110, y: 80 }, { x: 110, y: 80 }, bounds), { x: 99, y: 59, width: 1, height: 1 })
  assert.deepEqual(newCrop({ x: 60, y: 40 }, { x: 20, y: 10 }, bounds), rect)
  const a = editCrop(rect, 'nw', 2.4, 3.7, bounds)
  assert.deepEqual(editCrop(rect, 'nw', 2.4, 3.7, bounds), a)
  editCrop(rect, 'se', -999, 999, bounds)
  assert.deepEqual(editCrop(rect, 'nw', 2.4, 3.7, bounds), a)
})
