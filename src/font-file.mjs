// Faces in font files: family, style and weight from their name and OS/2 tables. A collection (.ttc, .otc) splits into
// standalone faces, since a FontFace loads only the first face of one. WOFF2 tables are Brotli-compressed, which
// browsers cannot inflate, so a WOFF2 face is named from its file name.
export const FONT_FILE = /\.(ttf|otf|ttc|otc|woff2?)$/i

// Weight and slope from a style name such as "Semibold Italic"; longer names first, so "Extra Bold" is not "Bold".
const WEIGHTS = [[/thin|hairline/, 100], [/(extra|ultra)[ -]?light/, 200], [/light/, 300], [/medium/, 500], [/(semi|demi)[ -]?bold/, 600], [/(extra|ultra)[ -]?bold/, 800], [/black|heavy/, 900], [/bold/, 700]]
export function styleOf(name) {
  const s = name.toLowerCase()
  return { weight: WEIGHTS.find(([pattern]) => pattern.test(s))?.[1] ?? 400, style: /italic|oblique/.test(s) ? 'italic' : 'normal' }
}

const tag = (view, at) => String.fromCharCode(view.getUint8(at), view.getUint8(at + 1), view.getUint8(at + 2), view.getUint8(at + 3))
const pad4 = n => (n + 3) & ~3

// Name IDs 16/17 (typographic family and style) when present, else 1/2; Windows English first, then any Unicode, then Mac Roman.
function names(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), count = view.getUint16(2), strings = view.getUint16(4), found = {}
  for (let i = 0; i < count; i++) {
    const r = 6 + i * 12, platform = view.getUint16(r), language = view.getUint16(r + 4), id = view.getUint16(r + 6)
    const rank = platform === 3 && language === 0x409 ? 3 : platform === 3 || platform === 0 ? 2 : platform === 1 && language === 0 ? 1 : 0
    if (!rank || (found[id]?.rank ?? 0) >= rank) continue
    const text = bytes.subarray(strings + view.getUint16(r + 10), strings + view.getUint16(r + 10) + view.getUint16(r + 8))
    found[id] = { rank, text: platform === 1 ? String.fromCharCode(...text) : new TextDecoder('utf-16be').decode(text) }
  }
  return id => found[id]?.text?.trim() || undefined
}
function describe(tables, fallback) {
  const name = tables.name ? names(tables.name) : () => undefined, os2 = tables['OS/2']
  const family = name(16) ?? name(1) ?? fallback.family, styleName = name(17) ?? name(2) ?? fallback.styleName
  const guessed = styleOf(styleName), view = os2 && new DataView(os2.buffer, os2.byteOffset, os2.byteLength)
  const weight = view?.byteLength >= 6 ? view.getUint16(4) : 0, italic = view?.byteLength >= 64 && (view.getUint16(62) & 0x201)
  return { family, styleName, postscriptName: name(6), weight: weight >= 1 && weight <= 1000 ? weight : guessed.weight, style: italic ? 'italic' : guessed.style }
}

// The face whose table directory starts at `at`, as a standalone font; tables are copied, offsets rewritten.
function standalone(bytes, at) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), n = view.getUint16(at + 4)
  const records = Array.from({ length: n }, (_, i) => ({ at: at + 12 + i * 16, offset: view.getUint32(at + 20 + i * 16), length: view.getUint32(at + 24 + i * 16) }))
  const out = new Uint8Array(12 + 16 * n + records.reduce((sum, r) => sum + pad4(r.length), 0)), put = new DataView(out.buffer)
  out.set(bytes.subarray(at, at + 12))
  let offset = 12 + 16 * n
  for (const [i, r] of records.entries()) {
    out.set(bytes.subarray(r.at, r.at + 8), 12 + i * 16); put.setUint32(20 + i * 16, offset); put.setUint32(24 + i * 16, r.length)
    out.set(bytes.subarray(r.offset, r.offset + r.length), offset); offset += pad4(r.length)
  }
  return out
}
function sfntTables(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), tables = {}
  for (let i = 0; i < view.getUint16(4); i++) { const r = 12 + i * 16; tables[tag(view, r)] = bytes.subarray(view.getUint32(r + 8), view.getUint32(r + 8) + view.getUint32(r + 12)) }
  return tables
}
async function inflate(bytes) {
  return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer())
}
async function woffTables(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), tables = {}
  for (let i = 0; i < view.getUint16(12); i++) {
    const r = 44 + i * 20, name = tag(view, r)
    if (name !== 'name' && name !== 'OS/2') continue
    const data = bytes.subarray(view.getUint32(r + 4), view.getUint32(r + 4) + view.getUint32(r + 8))
    tables[name] = view.getUint32(r + 8) < view.getUint32(r + 12) ? await inflate(data) : data
  }
  return tables
}

// Every face in a file: { family, styleName, postscriptName, weight, style, source }, where `source` is what a FontFace loads.
export async function readFontFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer()), base = file.name.replace(/\.[^.]+$/, '')
  // "Inter-BoldItalic" names the family Inter and the style Bold Italic.
  const dash = base.lastIndexOf('-'), fallback = dash > 0
    ? { family: base.slice(0, dash).replace(/[_]+/g, ' '), styleName: base.slice(dash + 1).replace(/([a-z])([A-Z])/g, '$1 $2') }
    : { family: base.replace(/[_]+/g, ' '), styleName: 'Regular' }
  if (bytes.length < 12) throw new Error(`${file.name} is not a font`)
  const kind = tag(new DataView(bytes.buffer), 0)
  if (kind === 'ttcf') {
    const view = new DataView(bytes.buffer)
    return Array.from({ length: view.getUint32(8) }, (_, i) => { const face = standalone(bytes, view.getUint32(12 + i * 4)); return { ...describe(sfntTables(face), fallback), source: face.buffer } })
  }
  if (kind === 'wOFF') return [{ ...describe(await woffTables(bytes), fallback), source: bytes.buffer }]
  if (kind === 'wOF2') return [{ ...describe({}, fallback), source: bytes.buffer }]
  if (['\0\x01\0\0', 'OTTO', 'true'].includes(kind)) return [{ ...describe(sfntTables(bytes), fallback), source: bytes.buffer }]
  throw new Error(`${file.name} is not a font`)
}
