// The faces a file-built catalog indexes for a family of bench/open-fonts.json: one per font file, or, for a variable file
// with named instances (scripts/instances.py, .data/fonts-open/instances.json), one per instance, each a static file of
// its own, so every style the family offers has a row and a picture. The default face, which carries the family's twins,
// is the selected file, or its instance nearest Regular when it is instanced. `path` is store-relative: what the file
// route serves, and what scripts/previews.py draws.
const WEIGHTS = { 100: 'Thin', 200: 'ExtraLight', 300: 'Light', 400: 'Regular', 500: 'Medium', 600: 'SemiBold', 700: 'Bold', 800: 'ExtraBold', 900: 'Black' }

export function styleName(weight, italic) {
  const name = WEIGHTS[Math.min(900, Math.max(100, Math.round(weight / 100) * 100))]
  return italic ? (name === 'Regular' ? 'Italic' : `${name} Italic`) : name
}

// `hosted`: the address its source serves a font file from, by the file's store path, for a source whose web fonts the
// page may draw: such a face carries `file`, and an instance of a variable file its `axes` too, so the page draws the
// instance from the one file (src/references.mjs drawLine, app.mjs loadFont) for its preview and the same-text stage.
export function catalogFaces(family, { instances = { faces: {}, families: {} }, twins = {}, home, hosted = new Map() } = {}) {
  const scripts = Object.keys(family.letters).sort(), sourceUrl = family.sourceUrl ?? home, defaultId = instances.families[family.id]?.default  // A family without its own page links to its source's.
  const entry = (id, path, style, isDefault, file, axes) => ({ id, familyId: family.id, family: family.family, ...style, scripts, sourceUrl, ...(isDefault ? { default: true, ...(twins[family.id] ? { twins: twins[family.id] } : {}) } : {}), ...(file ? { file, ...(axes ? { axes } : {}) } : {}), path })
  return family.faces.flatMap(face => {
    const found = instances.faces[face.path], file = hosted.get(face.path)
    if (found?.length) return found.map(i => entry(i.id, i.path, { styleName: i.styleName, weight: i.weight, style: i.italic ? 'italic' : 'normal' }, i.id === defaultId, file, i.axes))
    const stem = face.path.split('/').at(-1).replace(/\.[^.]+$/, '')
    return [entry(`${family.id}/${stem}`, face.path, { styleName: styleName(face.weight, face.italic), weight: face.weight, style: face.italic ? 'italic' : 'normal' }, face.path === family.selected && !defaultId, file)]
  })
}
