import { test } from 'node:test'
import assert from 'node:assert/strict'
import { catalogFaces, styleName } from '../scripts/catalog-faces.mjs'

const family = { id: 'pp-kyoto', family: 'PP Kyoto', letters: { Latn: 250, Hira: 86 }, selected: 'pp/Kyoto-Upright.ttf', sourceUrl: 'https://pp.example/kyoto',
  faces: [{ path: 'pp/Kyoto-Upright.ttf', weight: 100, italic: false, axes: { wght: {} } }, { path: 'pp/Kyoto-Italic.ttf', weight: 100, italic: true, axes: { wght: {} } }, { path: 'pp/Kyoto-Extra.ttf', weight: 700, italic: false, axes: {} }] }
const instances = { families: { 'pp-kyoto': { default: 'pp-kyoto/Kyoto-Upright@regular' } }, faces: {
  'pp/Kyoto-Upright.ttf': [{ id: 'pp-kyoto/Kyoto-Upright@thin', path: 'instances/pp-kyoto/Kyoto-Upright@thin.ttf', styleName: 'Thin', weight: 100, italic: false },
    { id: 'pp-kyoto/Kyoto-Upright@regular', path: 'instances/pp-kyoto/Kyoto-Upright@regular.ttf', styleName: 'Regular', weight: 400, italic: false }],
  'pp/Kyoto-Italic.ttf': [{ id: 'pp-kyoto/Kyoto-Italic@bold-italic', path: 'instances/pp-kyoto/Kyoto-Italic@bold-italic.ttf', styleName: 'Bold Italic', weight: 700, italic: true }] } }

test('a variable file with named instances gives one face per instance, the default on the selected file\'s instance nearest Regular, with the twins', () => {
  const faces = catalogFaces(family, { instances, twins: { 'pp-kyoto': ['pp-kyoto-vf'] } })
  assert.deepEqual(faces.map(f => f.id), ['pp-kyoto/Kyoto-Upright@thin', 'pp-kyoto/Kyoto-Upright@regular', 'pp-kyoto/Kyoto-Italic@bold-italic', 'pp-kyoto/Kyoto-Extra'])
  assert.deepEqual(faces.filter(f => f.default).map(f => f.id), ['pp-kyoto/Kyoto-Upright@regular'])
  assert.deepEqual(faces[1], { id: 'pp-kyoto/Kyoto-Upright@regular', familyId: 'pp-kyoto', family: 'PP Kyoto', styleName: 'Regular', weight: 400, style: 'normal', scripts: ['Hira', 'Latn'],
    sourceUrl: 'https://pp.example/kyoto', default: true, twins: ['pp-kyoto-vf'], path: 'instances/pp-kyoto/Kyoto-Upright@regular.ttf' })
  assert.deepEqual(faces[2], { id: 'pp-kyoto/Kyoto-Italic@bold-italic', familyId: 'pp-kyoto', family: 'PP Kyoto', styleName: 'Bold Italic', weight: 700, style: 'italic', scripts: ['Hira', 'Latn'],
    sourceUrl: 'https://pp.example/kyoto', path: 'instances/pp-kyoto/Kyoto-Italic@bold-italic.ttf' })
  assert.deepEqual(faces[3], { id: 'pp-kyoto/Kyoto-Extra', familyId: 'pp-kyoto', family: 'PP Kyoto', styleName: 'Bold', weight: 700, style: 'normal', scripts: ['Hira', 'Latn'], sourceUrl: 'https://pp.example/kyoto', path: 'pp/Kyoto-Extra.ttf' })
})

test('without instances every file is a face as before, the selected file the default, a family without a page linking to its source\'s', () => {
  const faces = catalogFaces({ ...family, sourceUrl: undefined }, { home: 'https://pp.example', twins: { 'pp-kyoto': ['x'] } })
  assert.deepEqual(faces.map(f => [f.id, f.styleName, f.style, f.default ?? false, f.twins ?? null, f.sourceUrl]),
    [['pp-kyoto/Kyoto-Upright', 'Thin', 'normal', true, ['x'], 'https://pp.example'], ['pp-kyoto/Kyoto-Italic', 'Thin Italic', 'italic', false, null, 'https://pp.example'], ['pp-kyoto/Kyoto-Extra', 'Bold', 'normal', false, null, 'https://pp.example']])
})

test('a selected file without instances stays the default even when another file of the family is instanced', () => {
  const faces = catalogFaces(family, { instances: { families: {}, faces: { 'pp/Kyoto-Italic.ttf': instances.faces['pp/Kyoto-Italic.ttf'] } } })
  assert.deepEqual(faces.filter(f => f.default).map(f => f.id), ['pp-kyoto/Kyoto-Upright'])
  assert.deepEqual(faces.map(f => f.id), ['pp-kyoto/Kyoto-Upright', 'pp-kyoto/Kyoto-Italic@bold-italic', 'pp-kyoto/Kyoto-Extra'])
})

test('a hosted file gives its faces the address, and an instance of a variable file its axes, so the page draws the instance from the one file', () => {
  const hosted = new Map([['pp/Kyoto-Upright.ttf', 'https://pp.example/cdn/Kyoto-Upright.ttf'], ['pp/Kyoto-Extra.ttf', 'https://pp.example/cdn/Kyoto-Extra.ttf']])
  const withAxes = { ...instances, faces: { ...instances.faces, 'pp/Kyoto-Upright.ttf': instances.faces['pp/Kyoto-Upright.ttf'].map((i, k) => ({ ...i, axes: { wght: k ? 400 : 100 } })) } }
  const faces = catalogFaces(family, { instances: withAxes, hosted })
  assert.deepEqual(faces.map(f => [f.id, f.file ?? null, f.axes ?? null]), [
    ['pp-kyoto/Kyoto-Upright@thin', 'https://pp.example/cdn/Kyoto-Upright.ttf', { wght: 100 }], ['pp-kyoto/Kyoto-Upright@regular', 'https://pp.example/cdn/Kyoto-Upright.ttf', { wght: 400 }],
    ['pp-kyoto/Kyoto-Italic@bold-italic', null, null], ['pp-kyoto/Kyoto-Extra', 'https://pp.example/cdn/Kyoto-Extra.ttf', null]])
  assert.deepEqual(catalogFaces(family, { instances }).filter(f => f.file || f.axes), [], 'nothing hosted, nothing carried')
})

test('style names round weights to the nearest hundred and name the slope', () => {
  assert.deepEqual([styleName(450, false), styleName(400, true), styleName(950, true), styleName(50, false)], ['Medium', 'Italic', 'Black Italic', 'Thin'])
})
