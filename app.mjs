import { editCrop, newCrop } from './crop.mjs'
import { prepareInput } from './src/input.mjs'
import { prepareLine } from './src/line.mjs'
import { readNetwork, inferCPU, rankWindows } from './src/network.mjs'
import { createNetworkGPU } from './src/network-gpu.mjs'
import { readCatalog, unionCatalogs, subsetCatalog, matchCatalog, foldTwins, embedWindows, sha256, preparationHash, readHeads, verdict, writeStyle, readStyle } from './src/catalog.mjs'
import { readMyFonts } from './my-fonts.mjs'
import { previewPath } from './previews.mjs'

const $ = id => document.getElementById(id)
// The address the page opened with: ?catalog= names what to search, ?style= a shared crop's style, ?embed shows the
// matcher alone. Read once: the page rewrites its own address as the matches change.
const params = new URLSearchParams(location.search), embedded = params.has('embed')
// Dropdowns are absolutely positioned in page coordinates: they open inside the viewport, then scroll with their trigger.
const place = (popover, left, top) => Object.assign(popover.style, { left: `${left + scrollX}px`, top: `${top + scrollY}px` })
const source = $('source'), ctx = source.getContext('2d', { willReadFrequently: true })
let model, heads = null, catalog, gpu, gpuReason = '', current = null, crop = null, last = null
let revision = 0, analysis = 0, dragging = null, armed = null
let scheduled = 0, running = false, pending = false
// The sample the page opens with, and the text every sample and preview draws until edited.
const SAMPLE = 'lora', TEXT = 'Quiet rivers flow'
let previewText = TEXT, previewValid = true
// Preview text the previews accept: up to 80 printable ASCII characters, but ~ and ^.
const validText = text => text.length <= 80 && /^[\x20-\x7e]+$/.test(text) && !/[~^]/.test(text)
let searchCatalog = null, catalogRevision = 0
// More matches grows the list in place, with previews, from the top five to the top fifty.
const SHORT = 5, LONG = 50
let expanded = false
const fonts = new Map()
const isEncoder = () => model?.kind === 'font-encoder'

function message(text = '', error = false) {
  $('message').textContent = text
  $('message').toggleAttribute('data-error', error)
}
function controls() {
  $('resolution-control').hidden = !current
  for (const kind of document.querySelectorAll('.source-kind')) kind.hidden = !current
  $('sample').hidden = !current
  $('tools').hidden = !current
  $('empty').hidden = !!current
  $('replace-image').hidden = !current
}
function resolution() {
  const scale = Number($('resolution').value) / 100
  const width = Math.max(1, Math.round(crop.width * scale)), height = Math.max(1, Math.round(crop.height * scale))
  return { scale, width, height }
}
function invalidate(retain = false) {
  analysis++; last = null; pending = false
  cancelAnimationFrame(scheduled); scheduled = 0
  if (!retain) {
    $('results').replaceChildren(); hideTwins(); $('more-matches').inert = true
    previewValid = true; $('preview-error').hidden = true
  }
  shareable()
  // A retained source keeps its last windows until new ones replace them, so the page does not jump.
  if (!retain) $('normalized').replaceChildren()
  $('input-regions').replaceChildren()
  $('result-summary').textContent = ''; $('input-count').textContent = ''; $('detection-time').textContent = ''
  $('results').removeAttribute('aria-busy')
  controls()
}
function schedule() {
  pending = !!current && (!current.drawing || current.strokes > 0) && !!model && (!isEncoder() || !!searchCatalog)
  if (!pending) return
  $('results').setAttribute('aria-busy', 'true')
  if (!running && !scheduled) scheduled = requestAnimationFrame(() => { scheduled = 0; analyze() })
}
function backend() {
  $('backend').textContent = gpu ? 'WebGPU' : 'CPU'
  $('backend').dataset.backend = gpu ? 'webgpu' : 'cpu'
  $('device').textContent = gpu ? `WebGPU, ${gpu.info}` : gpuReason || 'CPU (JavaScript)'
}
function showInput(windows, region, sampled) {
  $('normalized').replaceChildren(); $('input-regions').replaceChildren()
  for (const [index, input] of windows.entries()) {
    const canvas = document.createElement('canvas'), image = new ImageData(input.width, input.height)
    canvas.width = input.width; canvas.height = input.height; canvas.style.cssText = `--w: ${input.width}; --h: ${input.height}` // CSS fits it to the row
    canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `Input ${index + 1}, ${input.width} by ${input.height} pixels`)
    for (let i = 0; i < input.pixels.length; i++) {
      image.data.fill(Math.round(input.pixels[i] * 255), i * 4, i * 4 + 3); image.data[i * 4 + 3] = 255
    }
    canvas.getContext('2d').putImageData(image, 0, 0); $('normalized').append(canvas)
    const outline = document.createElement('span'), r = input.rect
    const sx = region.width / sampled.width, sy = region.height / sampled.height
    Object.assign(outline.style, { left: `${(region.x + r.x * sx) / source.width * 100}%`, top: `${(region.y + r.y * sy) / source.height * 100}%`, width: `${r.width * sx / source.width * 100}%`, height: `${r.height * sy / source.height * 100}%` })
    $('input-regions').append(outline)
  }
  $('input-count').textContent = `${windows.length} window${windows.length === 1 ? '' : 's'}`
}
function updateCrop(next) {
  if (crop && ['x', 'y', 'width', 'height'].every(key => crop[key] === next[key])) return
  crop = next
  const selection = $('selection')
  Object.assign(selection.style, { left: `${crop.x / source.width * 100}%`, top: `${crop.y / source.height * 100}%`, width: `${crop.width / source.width * 100}%`, height: `${crop.height / source.height * 100}%` })
  $('crop-shade').style.cssText = selection.style.cssText
  invalidate(true); message(); schedule()
}
function resetCrop() {
  if (!current) return
  armed = null; dragging = null
  updateCrop({ x: 0, y: 0, width: source.width, height: source.height })
}
function setImage(image, name, known = null, drawing = false) {
  source.width = image.width; source.height = image.height
  ctx.clearRect(0, 0, source.width, source.height); ctx.drawImage(image, 0, 0)
  current = { name, known, width: source.width, height: source.height, ...(drawing ? { drawing } : {}), strokes: 0 }
  const base = document.createElement('canvas'); base.width = source.width; base.height = source.height; base.getContext('2d').drawImage(source, 0, 0)
  edits = { base, strokes: [] }; $('undo').disabled = true
  if (known) lastFont = known
  setTool(drawing ? 'pen' : 'crop')
  sampleLabel()
  crop = null; $('resolution').value = '100'
  $('image-frame').hidden = false
  source.setAttribute('aria-label', name)
  resetCrop()
}
// Pencil and eraser edit any source; strokes replay over an untouched copy, so undo is exact.
const pen = { size: 12 }
let lastFont = SAMPLE, tool = 'crop', edits = { base: null, strokes: [] }
function penSize() {
  const input = $('pen-size'); pen.size = Number(input.value)
  input.parentElement.style.setProperty('--fill', `${(input.value - input.min) / (input.max - input.min) * 100}%`)
}
penSize()
$('pen-size').addEventListener('input', () => { penSize(); brushCursor() })
function setTool(next) {
  tool = next
  for (const [id, kind] of tools) $(id).setAttribute('aria-pressed', String(tool === kind))
  $('image-frame').dataset.tool = $('tools').dataset.tool = tool
  brushCursor()
}
const tools = [['crop-tool', 'crop'], ['pen-tool', 'pen'], ['erase', 'erase']]
for (const [id, kind] of tools) $(id).addEventListener('click', () => setTool(kind))
// While drawing, the cursor is the brush at its on-screen size; dashed while erasing.
function brushCursor() {
  const frame = $('image-frame')
  if (tool === 'crop') { frame.style.cursor = ''; return }
  const scale = source.getBoundingClientRect().width / source.width || 1
  const d = Math.max(4, Math.min(124, Math.round(pen.size * scale))), s = d + 2, c = s / 2
  const ring = `cx="${c}" cy="${c}" r="${d / 2}" fill="none"`
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}"><circle ${ring} stroke="#fff" stroke-width="2"/><circle ${ring} stroke="#000"${tool === 'erase' ? ' stroke-dasharray="3 2"' : ''}/></svg>`
  frame.style.cursor = `url("data:image/svg+xml,${encodeURIComponent(svg)}") ${Math.round(c)} ${Math.round(c)}, crosshair`
}
window.addEventListener('resize', brushCursor)
// The input panel's height, so CSS can pin a panel taller than the window by its bottom edge while matches scroll.
new ResizeObserver(([entry]) => entry.target.style.setProperty('--body-height', `${entry.borderBoxSize[0].blockSize}px`)).observe(document.querySelector('.source-body'))
function startDrawing() {
  revision++; invalidate()
  const sheet = document.createElement('canvas'), c = sheet.getContext('2d')
  sheet.width = 720; sheet.height = 240; c.fillStyle = '#ffffff'; c.fillRect(0, 0, sheet.width, sheet.height)
  setImage(sheet, 'Drawing', null, true)
  brushCursor()
}
// Paints a stroke's segments from point `from` on, one segment at a time, so replay after undo
// reproduces live drawing pixel for pixel; a single point is a dot.
function paint(stroke, from = 0) {
  const p = stroke.points
  Object.assign(ctx, { strokeStyle: stroke.erase ? '#ffffff' : '#000000', lineWidth: stroke.size, lineCap: 'round', lineJoin: 'round' })
  for (let i = from; i < p.length; i++) {
    const a = p[Math.max(0, i - 1)]
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(p[i].x, p[i].y); ctx.stroke()
  }
}
function edited() {
  current.strokes = edits.strokes.length; $('undo').disabled = !current.strokes
  if (current.drawing && !current.strokes) { invalidate(); return }
  message(); invalidate(true); schedule()
}
function addPoint(at) { const stroke = edits.strokes.at(-1); stroke.points.push(at); paint(stroke, stroke.points.length - 1); edited() }
function undo() {
  if (!current || !edits.strokes.length || dragging?.draw) return
  edits.strokes.pop()
  ctx.clearRect(0, 0, source.width, source.height); ctx.drawImage(edits.base, 0, 0)
  for (const stroke of edits.strokes) paint(stroke)
  edited()
}
$('undo').addEventListener('click', undo)
document.addEventListener('keydown', event => {
  if (!(event.metaKey || event.ctrlKey) || event.shiftKey || event.key.toLowerCase() !== 'z' || event.target.closest('input[type="text"], textarea')) return
  if (edits.strokes.length) { event.preventDefault(); undo() }
})
// Fonts load from Google Fonts when first shown; none are shipped. A face is { family, weight, style } and resolves to
// the CSS family to render with. Installed faces in My fonts render by PostScript name ({ local }); the legacy classifier
// build passes local files ({ file }). Their descriptors name the exact face, so it is never synthesized bolder or slanted.
function googleCss({ family, weight = 400, style = 'normal' }) {
  const axes = style === 'italic' ? `:ital,wght@1,${weight}` : weight === 400 ? '' : `:wght@${weight}`
  return `https://fonts.googleapis.com/css2?family=${encodeURIComponent(family).replaceAll('%20', '+')}${axes}&display=block`
}
// A face loaded from a file or by local name takes a CSS identifier of its own: `specimen-` and its key, every other
// character escaped by its code. FontFace hands back any other name quoted (local:Georgia, a file's address), and quoted
// again it names no font, so the preview would fall back to another face.
const specimenFamily = key => `specimen-${key.replace(/[^A-Za-z0-9-]/g, c => `_${c.charCodeAt(0).toString(16)}_`)}`
async function loadFont(key, face) {
  if (!face) throw new Error('Font is absent from this catalog')
  const src = face.file ? `url("${face.file}")` : face.local && `local("${face.local}")`, family = specimenFamily(key)
  if (!fonts.has(key)) fonts.set(key, (src
    ? new FontFace(family, src, { weight: String(face.weight ?? 400), style: face.style ?? 'normal' }).load().then(loaded => { document.fonts.add(loaded); return family })
    : new Promise((resolve, reject) => document.head.append(Object.assign(document.createElement('link'), { rel: 'stylesheet', href: googleCss(face), onload: resolve, onerror: () => reject(new Error(`Google Fonts does not serve ${face.family}`)) })))
      .then(() => document.fonts.load(`${face.style ?? 'normal'} ${face.weight ?? 400} 16px "${face.family}"`))
      .then(loaded => { if (!loaded.length) throw new Error(`Google Fonts does not serve ${face.family}`); return face.family })
  ).catch(error => { fonts.delete(key); throw error }))
  return fonts.get(key)
}
// Rows past the fifth load only the glyphs of their text (Google Fonts text=), under a family of their own, so a subset
// never stands in for the full face another row uses. Each face and text loads once.
const subsets = new Map()
let subsetCount = 0
function subsetFont(face, text) {
  const glyphs = [...new Set(text)].sort().join(''), key = `${face.id}\n${glyphs}`
  if (!subsets.has(key)) subsets.set(key, fetch(`${googleCss(face)}&text=${encodeURIComponent(glyphs)}`)
    .then(response => response.ok ? response.text() : Promise.reject(new Error(`Google Fonts does not serve ${face.family}`)))
    .then(css => {
      const url = css.match(/src:\s*url\(([^)]+)\)/)?.[1]
      if (!url) throw new Error(`Google Fonts does not serve ${face.family}`)
      return new FontFace(`subset-${++subsetCount}`, `url(${url})`, { weight: String(face.weight ?? 400), style: face.style ?? 'normal' }).load()
    })
    .then(loaded => { document.fonts.add(loaded); return loaded.family })
    .catch(error => { subsets.delete(key); throw error }))
  return subsets.get(key)
}
const sampleFont = id => { const font = catalog.fonts.find(f => f.id === id); return font && { ...font, family: font.family ?? font.name } }
async function sample(id) {
  if (!catalog) return
  if (!previewValid) { $('results').querySelector('[aria-invalid="true"]')?.focus(); return }
  const text = previewText
  const version = ++revision
  invalidate(); message('Loading sample…')
  try {
    const family = await loadFont(id, sampleFont(id))
    if (version !== revision) return
    const font = catalog.fonts.find(f => f.id === id)
    const canvas = document.createElement('canvas'), c = canvas.getContext('2d')
    c.font = `56px "${family}"`
    const m = c.measureText(text), pad = 28
    canvas.width = Math.ceil(m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + pad * 2
    canvas.height = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + pad * 2
    c.fillStyle = '#ffffff'; c.fillRect(0, 0, canvas.width, canvas.height)
    c.fillStyle = '#000000'; c.font = `56px "${family}"`
    c.fillText(text, pad + m.actualBoundingBoxLeft, pad + m.actualBoundingBoxAscent)
    // Sample rendering is training evidence: exact monochrome, independent of UI tokens.
    setImage(canvas, `${font.name} sample`, id); current.text = text
  } catch (error) {
    if (version === revision) message(error.message, true)
  }
}
async function openImage(file) {
  const version = ++revision
  invalidate()
  if (!file || !file.type.startsWith('image/')) { message('Choose an image file, such as PNG, JPEG or WebP.', true); return }
  if (file.size > 20 * 1024 * 1024) { message('Choose an image smaller than 20 MB.', true); return }
  let bitmap
  try {
    bitmap = await createImageBitmap(file)
    if (version !== revision) return
    if (bitmap.width * bitmap.height > 16_777_216) throw new Error('This image exceeds 16 megapixels. Save a smaller crop first.')
    setImage(bitmap, file.name || 'Pasted image')
    message()
  } catch (error) {
    if (version === revision) message(error.message.includes('megapixels') ? error.message : 'Could not read this image. Try a PNG, JPEG or WebP file.', true)
  } finally { bitmap?.close() }
}
function background(data) {
  let alpha = 0, luminance = 0, transparent = false
  for (let p = 0; p < data.length; p += 4) {
    const a = data[p + 3]
    transparent ||= a < 255
    alpha += a
    luminance += a * (data[p] * .2126 + data[p + 1] * .7152 + data[p + 2] * .0722)
  }
  return transparent && alpha && luminance / alpha > 127.5 ? 0 : 255
}
async function analyze() {
  if (running || !pending || !current || !model) return
  running = true; pending = false
  const run = analysis
  const region = { ...crop }, specimen = { ...current }
  try {
    const start = performance.now()
    const sampled = resolution()
    let image
    if (sampled.width === region.width && sampled.height === region.height) image = ctx.getImageData(region.x, region.y, region.width, region.height)
    else {
      const canvas = document.createElement('canvas'); canvas.width = sampled.width; canvas.height = sampled.height
      const c = canvas.getContext('2d'); c.imageSmoothingQuality = 'high'
      c.drawImage(source, region.x, region.y, region.width, region.height, 0, 0, sampled.width, sampled.height)
      image = c.getImageData(0, 0, sampled.width, sampled.height)
    }
    const matte = background(image.data)
    source.style.backgroundColor = matte === 0 ? '#000000' : '#ffffff'
    const prepared = isEncoder() ? prepareLine(image, { ...model.preparation, background: matte, deskew: true, sampler: 'windows' }) : prepareInput(image, { ...model.preparation, background: matte })
    if (prepared.status !== 'ok') {
      invalidate()
      if (specimen.drawing) return
      message(prepared.status === 'blank' ? 'No visible text in this crop.' : 'Not enough contrast in this crop.', true)
      return
    }
    let logits, used = 'CPU'
    if (gpu) {
      const executor = gpu
      try { logits = await Promise.all(prepared.windows.map(input => executor.infer(input))); used = 'WebGPU' }
      catch (error) {
        executor.destroy()
        if (gpu === executor) { gpu = null; gpuReason = `CPU fallback: ${error.message}`; backend() }
      }
    }
    if (run !== analysis) return
    if (!logits) logits = prepared.windows.map(input => inferCPU(model, input))
    const embedding = isEncoder() ? embedWindows(logits) : null, judged = embedding && heads ? verdict(embedding, heads) : null
    const matches = embedding ? matchCatalog(embedding, searchCatalog, judged) : rankWindows(logits, model.fonts, catalog.calibration?.temperature ?? 1), elapsed = performance.now() - start
    if (!embedding) await Promise.all(matches.slice(0, 5).map(m => loadFont(m.family, sampleFont(m.family))))
    if (run !== analysis) return
    last = { accepted: embedding ? null : matches[0].score >= (catalog.calibration?.threshold ?? 1.01), calibration: embedding ? null : catalog.calibration, mode: embedding ? 'encoder' : 'neural', backend: used, milliseconds: elapsed, model: catalog.modelSha256, source: specimen, crop: region, resolution: sampled, background: matte,
      ...(embedding ? { embedding: Array.from(embedding), verdict: judged, catalog: { id: searchCatalog.id, sha256: searchCatalog.sha256 }, scoreType: 'cosine', inferenceMilliseconds: elapsed } : {}),
      inputs: prepared.windows.map(input => ({ ...input, pixels: Array.from(input.pixels) })), matches }
    showInput(prepared.windows, region, sampled)
    renderResults()
    showVerdict(embedding ? labelsFor(judged, matches) : [last.accepted ? 'Above threshold' : 'Below threshold'])
    $('detection-time').textContent = `${elapsed.toFixed(1)}ms`
    $('detection-time').title = 'Image preparation, inference and ranking'
    shareable()
  } catch (error) { if (run === analysis) { invalidate(); message(`Could not analyze this crop: ${error.message}`, true) } }
  finally {
    running = false
    if (run === analysis) { controls(); $('results').removeAttribute('aria-busy') }
    if (pending) schedule()
  }
}
function renderResults() {
  hideTwins() // Its note is about to be replaced.
  $('more-matches').inert = true
  if (!last) return
  $('more-matches').inert = !isEncoder() || foldTwins(last.matches, searchCatalog, { script: last.verdict?.script?.[0]?.label, limit: SHORT + 1 }).length <= SHORT
  $('more-matches').setAttribute('aria-expanded', String(expanded)); $('more-matches').firstChild.textContent = expanded ? 'Fewer matches' : 'More matches'
  subsetRows.clear()
  previewValid = true; $('preview-error').hidden = true
  const fragment = document.createDocumentFragment()
  // Kinds of one family identical in the crop's script (IBM Plex Sans and its KR, Arabic… families) fold into one row; the saved result keeps every family.
  const shown = isEncoder() ? foldTwins(last.matches, searchCatalog, { script: last.verdict?.script?.[0]?.label, limit: expanded ? LONG : SHORT }) : last.matches.slice(0, SHORT)
  for (const [i, match] of shown.entries()) {
    if (isEncoder()) { fragment.append(encoderResult(match, i)); continue }
    const font = catalog.fonts.find(f => f.id === match.family)
    const item = document.createElement('li'); item.className = 'result'
    item.innerHTML = '<div class="result-top"><span class="rank"></span><span class="font-name"><a target="_blank" rel="noopener"></a><span class="font-style pill" title="Preview face; weight and style are not detected">Regular, 400</span></span><span class="result-score"></span></div><input class="result-preview" type="text" maxlength="80" spellcheck="false" autocomplete="off" aria-describedby="preview-error">'
    item.querySelector('.rank').textContent = String(i + 1).padStart(2, '0')
    const score = item.querySelector('.result-score')
    score.textContent = match.score < .001 ? '<0.1%' : match.score > .999 ? '>99.9%' : `${(match.score * 100).toFixed(1)}%`
    score.setAttribute('aria-label', `Model score: ${score.textContent}`)
    const link = item.querySelector('a'); link.textContent = font.name + ' ↗'
    link.href = `https://fonts.google.com/specimen/${encodeURIComponent(font.name).replaceAll('%20', '+')}`
    link.setAttribute('aria-label', `${font.name} on Google Fonts`)
    link.title = font.name
    const preview = item.querySelector('.result-preview')
    preview.value = previewText
    preview.setAttribute('aria-label', `Preview ${font.name}`)
    if (i === 0) preview.id = 'preview-text'
    preview.style.setProperty('--font-specimen', `"${specimenFamily(font.id)}"`)
    fragment.append(item)
  }
  $('results').replaceChildren(fragment)
}

const CATEGORY = { SANS_SERIF: 'Sans serif', SERIF: 'Serif', DISPLAY: 'Display', HANDWRITING: 'Handwriting', MONOSPACE: 'Monospace' }
const SCRIPT = new Intl.DisplayNames('en', { type: 'script' }) // Latn → Latin
// The typed verdict for the crop itself: category, the likeliest fine class, upright or italic, script.
// The verdict pills, plus a warning when the best match scores below the model's rejection threshold: the crop's font
// is then probably in no catalog, though the closest look-alike still shows.
const labelsFor = (judged, matches) => [...describe(judged), ...(model.threshold != null && matches[0] && matches[0].score < model.threshold ? ['Probably in no catalog'] : [])]
function describe(judged) {
  if (!judged) return []
  const fine = judged.fine[0].p >= .5 && judged.fine[0].label.split('/').at(-1)
  return [CATEGORY[judged.category[0].label] ?? judged.category[0].label, fine, judged.italic >= .5 ? 'Italic' : 'Upright', SCRIPT.of(judged.script[0].label)].filter(Boolean)
}
// One pill per label; the spaces between them keep a screen reader from running the words together.
const showVerdict = labels => $('result-summary').replaceChildren(...labels.flatMap((label, i) => {
  const pill = document.createElement('span'); pill.className = 'pill'; pill.textContent = label
  return i ? [' ', pill] : [pill]
}))

// One match on one line: rank, linked name, face, ≈ twins, score.
function matchLine(match, index) {
  const { face, score, siblings } = match, line = document.createElement('div'); line.className = 'result-top'
  line.innerHTML = '<span class="rank"></span><span class="font-name"><a target="_blank" rel="noopener"></a><span class="font-style pill"></span></span><span class="result-score"></span>'
  line.querySelector('.rank').textContent = String(index + 1).padStart(2, '0')
  const link = line.querySelector('a'); link.textContent = face.family
  // A Google Fonts family links to its specimen from any shipped catalog it's searched in, All included.
  const url = face.sourceUrl || (searchCatalog.builtin && catalog.previews?.[face.familyId] ? `https://fonts.google.com/specimen/${encodeURIComponent(face.family)}` : null)
  if (url && /^https?:\/\//i.test(url)) { link.href = url; link.textContent += ' ↗' }
  const style = line.querySelector('.font-style')
  style.textContent = face.styleName || [face.style, face.weight].filter(v => v != null).join(', ')
  style.title = `Matched face${face.weight ? `, weight ${face.weight}` : ''}`
  if (siblings.length) {
    // Folded kinds of this family with identical letters: yields space first, so the matched name and face stay readable.
    const note = document.createElement('span'); note.className = 'font-twins'
    note.textContent = `≈ ${siblings[0]}${siblings.length > 1 ? ` +${siblings.length - 1}` : ''}`
    note.tabIndex = 0; note.dataset.twins = JSON.stringify(siblings); note.setAttribute('aria-label', `Same letters as ${siblings.join(', ')}`)
    style.after(note)
  }
  const value = line.querySelector('.result-score')
  value.textContent = score.toFixed(3); value.title = 'Cosine similarity, not certainty'; value.setAttribute('aria-label', `Cosine similarity: ${score.toFixed(3)}`)
  return line
}
function encoderResult(match, index) {
  const { face } = match, item = document.createElement('li'); item.className = 'result'
  item.append(matchLine(match, index), previewElement(face, previewSources(face), index))
  return item
}
// Where a match can be seen, in order: its own font setting the preview text (Google Fonts, or installed), else the
// picture stored with the site for every other face of a shipped catalog.
function previewSources(face) {
  const google = searchCatalog.builtin && catalog.previews[face.familyId]
  return google ? [{ text: google }] : face.local ? [{ text: { latin: true } }] : searchCatalog.builtin ? [{ stored: face.id }] : []
}
// The first source that shows: one that fails to load gives way to the next, the last to a note. Never a fallback face
// displayed as if it were a matching specimen.
function previewElement(face, [candidate, ...rest], index) {
  const next = element => () => { if (element.isConnected) element.replaceWith(previewElement(face, rest, index)) }
  if (!candidate) {
    const note = document.createElement('p'); note.className = 'result-unavailable'; note.textContent = 'No preview available'
    return note
  }
  if (candidate.stored) {
    const image = new Image(); image.className = 'result-image'; image.alt = `Preview of ${face.family}`; image.loading = 'lazy'; image.decoding = 'async'
    image.addEventListener('error', next(image)); previewPath(candidate.stored).then(path => { image.src = path })
    return image
  }
  const { text } = candidate
  const input = document.createElement('input'); input.className = 'result-preview'; input.type = 'text'; input.maxLength = 80
  input.spellcheck = false; input.autocomplete = 'off'; input.style.visibility = 'hidden' // Keeps its row while the font loads.
  input.setAttribute('aria-label', `Preview ${face.family}`); input.setAttribute('aria-describedby', 'preview-error')
  input.readOnly = !text.latin; input.value = text.latin ? previewText : text.sampleText
  input.style.fontWeight = face.weight ?? 400; input.style.fontStyle = face.style ?? 'normal'
  if (!index && text.latin) input.id = 'preview-text'
  const subset = index >= SHORT && !face.local && !face.file
  if (subset) { input.readOnly = true; input.tabIndex = -1; if (text.latin) subsetRows.set(input, face) }
  ;(subset ? subsetFont(face, input.value) : loadFont(face.id, face)).then(family => { input.style.setProperty('--font-specimen', `"${family}"`); input.style.visibility = '' })
    .catch(next(input))
  return input
}

// Shipped catalogs, each fetched once, checked against the checksum site.json records for it and read: switching back
// is instant, and All reuses the ones already read. `received` counts bytes as they arrive, or a read catalog's size.
const shipped = new Map()
function readShipped(option, received) {
  if (shipped.has(option.file)) return shipped.get(option.file).then(data => { received(option.bytes); return data })
  const reading = download(option.file, received, 'Could not load this catalog.').then(async bytes => {
    if (await sha256(bytes) !== option.sha256) throw new Error('Catalog checksum failed.')
    return readCatalog(JSON.parse(new TextDecoder().decode(bytes)), catalog)
  })
  shipped.set(option.file, reading); reading.catch(() => shipped.delete(option.file))
  return reading
}
// The catalog being loaded: the address names it before it lands, so a reload in between asks for it again.
let loading = null
async function selectCatalog(option, file = null, { focus = true } = {}) {
  const version = ++catalogRevision
  $('catalog-menu').hidePopover(); if (focus) $('catalog-button').focus({ preventScroll: true })
  $('catalog-error').hidden = true; $('catalog-button').setAttribute('aria-busy', 'true')
  // The matches wait behind a loader, as the matcher waits for the model: the bytes received of the files the catalog needs.
  const files = option.parts ?? (option.group ? [option.group] : file || option.data ? [] : [option]), total = files.reduce((n, f) => n + f.bytes, 0), bar = $('catalog-loading')
  let got = 0
  const received = n => {
    if (version !== catalogRevision) return
    got += n; bar.value = got; $('catalog-loading-percent').textContent = `${Math.floor(got / total * 100)}%`
  }
  if (total) { bar.max = total; received(0) } else { bar.removeAttribute('value'); $('catalog-loading-percent').textContent = '' }
  $('catalog-loading-name').textContent = option.name; loading = option; $('results-panel').setAttribute('aria-busy', 'true'); shareable()
  try {
    let data, hash
    if (option.parts) {
      // All: every shipped catalog, each verified as when chosen alone, searched as one.
      const parts = await Promise.all(option.parts.map(part => readShipped(part, received)))
      data = unionCatalogs(parts); hash = option.parts.map(part => part.sha256)
    } else if (option.group) {
      // One source of a grouped catalog (Other), searched alone: its faces, which name their source.
      data = subsetCatalog(await readShipped(option.group, received), face => face.sourceId === option.id); hash = option.group.sha256
    } else if (file || option.data) {
      if (file?.size > 20 * 1024 * 1024) throw new Error('Choose a catalog smaller than 20 MB.')
      const bytes = file ? await file.arrayBuffer() : new TextEncoder().encode(JSON.stringify(option.data))
      hash = await sha256(bytes); data = readCatalog(JSON.parse(new TextDecoder().decode(bytes)), catalog)
    } else { data = await readShipped(option, received); hash = option.sha256 }
    if (version !== catalogRevision) return
    const cached = last
    searchCatalog = { ...option, ...data, sha256: hash, builtin: !file && !option.data }
    invalidate(); updateCatalogLabel()
    if (cached?.embedding && (current || cached.shared)) {
      const start = performance.now(), matches = matchCatalog(cached.embedding, searchCatalog, cached.verdict)
      const elapsed = performance.now() - start
      last = { ...cached, matches, catalog: { id: option.id, sha256: hash }, milliseconds: elapsed, cachedEmbedding: true }
      if (last.inputs) showInput(last.inputs, last.crop, last.resolution)
      renderResults()
      $('detection-time').textContent = `Ranked in ${elapsed.toFixed(1)}ms`
      $('detection-time').title = 'Catalog ranking; image embedding reused'
      showVerdict(labelsFor(last.verdict, matches)); shareable()
    } else if (current) schedule()
  } catch (error) {
    if (version === catalogRevision) { $('catalog-error').textContent = error.message; $('catalog-error').hidden = false }
  } finally {
    if (version === catalogRevision) { loading = null; $('catalog-button').removeAttribute('aria-busy'); $('results-panel').removeAttribute('aria-busy') }
  }
}
function intro(lead, top1, scope) {
  $('intro-lead').textContent = lead
  $('intro-scope').textContent = `Experimental: on ${scope}, it names the right family first ${(top1 * 100).toFixed(1)}% of the time.`
}
// A shipped catalog shows its source's favicon; All, Other, My fonts and an opened file show the catalog icon.
const layers = $('catalog-button').querySelector('.catalog-icon').cloneNode(true)
const iconOf = option => option.icon ? Object.assign(new Image(16, 16), { className: 'favicon', src: option.icon, alt: '' }) : layers.cloneNode(true)
function updateCatalogLabel() {
  $('catalog-button').firstElementChild.replaceWith(iconOf(searchCatalog))
  $('catalog-label').textContent = searchCatalog.name
  $('catalog-size').textContent = `${searchCatalog.name}: ${searchCatalog.families.toLocaleString()} families, ${searchCatalog.faces.length.toLocaleString()} faces`
  for (const button of $('catalog-list').children) button.setAttribute('aria-pressed', String(button.dataset.catalog === searchCatalog.id))
}
$('catalog-file').addEventListener('change', event => {
  const file = event.target.files[0]; event.target.value = ''
  if (file) selectCatalog({ id: 'custom', name: file.name }, file)
})
$('import-catalog').addEventListener('click', () => { $('catalog-menu').hidePopover(); $('catalog-file').click() })
function positionCatalog() {
  const menu = $('catalog-menu'), rect = $('catalog-button').getBoundingClientRect()
  // The trigger ends its column; the menu hangs from the same edge.
  place(menu, Math.max(12, Math.min(innerWidth - menu.offsetWidth - 12, rect.right - menu.offsetWidth)), Math.max(12, Math.min(innerHeight - menu.offsetHeight - 12, rect.bottom + 4)))
}
$('catalog-button').addEventListener('click', () => {
  const menu = $('catalog-menu')
  if (menu.matches(':popover-open')) menu.hidePopover()
  else {
    // Placed before focusing, so focusing the selected option never scrolls the page; synchronous, so the next key reaches the menu.
    menu.showPopover(); positionCatalog()
    ;($('catalog-list').querySelector('[aria-pressed="true"]') || $('import-catalog')).focus({ preventScroll: true })
  }
})
$('catalog-menu').addEventListener('beforetoggle', event => { $('catalog-button').setAttribute('aria-expanded', String(event.newState === 'open')) })
$('catalog-menu').addEventListener('toggle', event => { if (event.newState === 'open') positionCatalog() })
$('catalog-menu').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); $('catalog-menu').hidePopover(); $('catalog-button').focus(); return }
  if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const buttons = [...$('catalog-menu').querySelectorAll('button, a:not([hidden])')], at = buttons.indexOf(document.activeElement)
  buttons[event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (at + (event.key === 'ArrowUp' ? -1 : 1) + buttons.length) % buttons.length].focus()
})
window.addEventListener('resize', () => { if ($('catalog-menu').matches(':popover-open')) positionCatalog() })
// Twin notes open the shared tooltip on hover, focus or tap.
function showTwins(note) {
  const tip = $('twins-tip'), list = document.createElement('ul')
  list.append(...JSON.parse(note.dataset.twins).map(name => Object.assign(document.createElement('li'), { textContent: name })))
  tip.replaceChildren(list)
  if (!tip.matches(':popover-open')) tip.showPopover()
  const rect = note.getBoundingClientRect(), below = rect.bottom + 6 + tip.offsetHeight <= innerHeight - 12
  place(tip, Math.max(12, Math.min(innerWidth - tip.offsetWidth - 12, rect.left)), below ? rect.bottom + 6 : Math.max(12, rect.top - 6 - tip.offsetHeight))
}
function hideTwins() { if ($('twins-tip').matches(':popover-open')) $('twins-tip').hidePopover() }
for (const type of ['pointerover', 'focusin']) $('results').addEventListener(type, event => { const note = event.target.closest('.font-twins'); if (note) showTwins(note) })
for (const type of ['pointerout', 'focusout']) $('results').addEventListener(type, event => { if (event.target.closest('.font-twins') && !event.relatedTarget?.closest?.('.font-twins')) hideTwins() })
$('results').addEventListener('keydown', event => { if (event.key === 'Escape' && $('twins-tip').matches(':popover-open')) { event.stopPropagation(); hideTwins() } })
$('more-matches').addEventListener('click', () => { expanded = !expanded; renderResults() })
// Subset rows follow the edited text once its glyphs load, so they never show a fallback face in between.
const subsetRows = new Map()
let subsetTimer = 0
function followText(text) {
  clearTimeout(subsetTimer)
  subsetTimer = setTimeout(() => { for (const [input, face] of subsetRows) subsetFont(face, text).then(family => { if (!input.isConnected || previewText !== text) return; input.style.setProperty('--font-specimen', `"${family}"`); input.value = text }).catch(() => {}) }, 250)
}
function updatePreview(input) {
  const text = input.value.trim() || TEXT
  previewValid = validText(text)
  $('preview-error').hidden = previewValid
  input.setAttribute('aria-invalid', String(!previewValid))
  if (!previewValid) return
  previewText = text
  for (const other of $('results').querySelectorAll('.result-preview')) {
    if (other.readOnly) continue
    other.setAttribute('aria-invalid', 'false')
    if (other !== input) other.value = text
  }
  followText(text)
}
function point(event) {
  const r = source.getBoundingClientRect()
  return { x: Math.max(0, Math.min(source.width, (event.clientX - r.left) * source.width / r.width)), y: Math.max(0, Math.min(source.height, (event.clientY - r.top) * source.height / r.height)) }
}
function applyDrag(drag, end) {
  updateCrop(drag.handle === 'new' ? newCrop(drag.start, end, source) : editCrop(drag.before, drag.handle, end.x - drag.start.x, end.y - drag.start.y, source))
}
$('image-frame').addEventListener('pointerdown', event => {
  if (!current || dragging || !event.isPrimary || event.button !== 0) return
  const start = point(event), handle = event.target.closest('[data-handle]')?.dataset.handle
  if (tool !== 'crop') {
    edits.strokes.push({ erase: tool === 'erase', size: pen.size, points: [] }); dragging = { id: event.pointerId, draw: true }; addPoint(start)
    $('image-frame').focus({ preventScroll: true }); $('image-frame').setPointerCapture(event.pointerId); event.preventDefault(); return
  }
  if (armed) { applyDrag(armed, start); armed = null; event.preventDefault(); return }
  const inside = start.x >= crop.x && start.x <= crop.x + crop.width && start.y >= crop.y && start.y <= crop.y + crop.height
  dragging = { id: event.pointerId, before: { ...crop }, start, handle: handle || (inside ? 'move' : 'new'), x: event.clientX, y: event.clientY, moved: false }
  ;(handle ? event.target : $('image-frame')).focus({ preventScroll: true })
  $('image-frame').setPointerCapture(event.pointerId); event.preventDefault()
})
$('image-frame').addEventListener('pointermove', event => {
  if (dragging?.draw && dragging.id === event.pointerId) { addPoint(point(event)); return }
  if (dragging?.id !== event.pointerId || Math.hypot(event.clientX - dragging.x, event.clientY - dragging.y) < 3) return
  dragging.moved = true; applyDrag(dragging, point(event))
})
$('image-frame').addEventListener('pointerup', event => {
  if (dragging?.id !== event.pointerId) return
  if (dragging.draw) { dragging = null; return }
  if (dragging.moved) applyDrag(dragging, point(event))
  else if (!['new', 'move'].includes(dragging.handle)) armed = { ...dragging }
  dragging = null
})
for (const type of ['pointercancel', 'lostpointercapture']) $('image-frame').addEventListener(type, event => {
  if (dragging?.id === event.pointerId) { dragging = null; armed = null }
})
$('image-frame').addEventListener('keydown', event => {
  if (!current) return
  if (event.key === 'Home' || event.key === 'Escape') { event.preventDefault(); resetCrop(); return }
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return
  event.preventDefault(); armed = null; dragging = null
  const dx = event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : 0
  const dy = event.key === 'ArrowUp' ? -1 : event.key === 'ArrowDown' ? 1 : 0
  const handle = event.target.closest('[data-handle]')?.dataset.handle || (event.shiftKey ? 'se' : 'move')
  updateCrop(editCrop(crop, handle, dx, dy, source))
})
$('resolution').addEventListener('change', () => { invalidate(true); message(); schedule() })
$('draw').addEventListener('click', () => { $('sample-menu').hidePopover(); startDrawing() })
$('source-font').addEventListener('click', () => { if (current?.known) return; sample(lastFont) })
for (const id of ['open-image', 'choose-image', 'replace-image']) $(id).addEventListener('click', () => { $('sample-menu').hidePopover(); $('file').click() })
$('file').addEventListener('change', () => { const file = $('file').files[0]; if (file) openImage(file); $('file').value = '' })
document.addEventListener('paste', event => {
  const file = [...(event.clipboardData?.items || [])].find(item => item.type.startsWith('image/'))?.getAsFile()
  if (file) { event.preventDefault(); openImage(file) }
})
document.addEventListener('dragover', event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault() })
document.addEventListener('drop', event => { event.preventDefault(); $('stage').classList.remove('dragover'); const file = event.dataTransfer.files[0]; if (file) openImage(file) })
$('stage').addEventListener('dragover', event => { event.preventDefault(); $('stage').classList.add('dragover') })
$('stage').addEventListener('dragleave', () => $('stage').classList.remove('dragover'))
$('results').addEventListener('input', event => { if (event.target.matches('.result-preview')) updatePreview(event.target) })
function sampleLabel() {
  const name = catalog?.fonts.find(f => f.id === current?.known)?.name || current?.name || 'Choose font'
  $('sample-label').textContent = name
  const kind = current?.drawing ? 'drawing' : current && !current.known ? 'image' : 'font'
  $('sample').dataset.source = kind
  const kinds = [['source-font', 'font'], ['draw', 'drawing'], ['choose-image', 'image']].map(([id, each]) => { $(id).setAttribute('aria-pressed', String(!!current && kind === each)); return [$(id), each] })
  // The current kind leads its name; the others follow the spacer. DOM order is tab order; a moved button keeps focus.
  const head = $('source-head'), focused = document.activeElement
  head.insertBefore(kinds.find(([, each]) => each === kind)[0], $('sample'))
  for (const [button, each] of kinds) if (each !== kind) head.append(button)
  if (kinds.some(([button]) => button === focused)) focused.focus({ preventScroll: true })
  $('sample').title = name
  $('sample').setAttribute('aria-label', current ? `Choose a font sample. Current ${kind}: ${name}` : 'Choose a font sample')
  for (const button of $('sample-list').children) button.setAttribute('aria-pressed', String(button.dataset.font === current?.known))
}
const specimens = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    const button = entry.target; specimens.unobserve(button)
    loadFont(button.dataset.font, sampleFont(button.dataset.font)).then(family => button.style.setProperty('--font-specimen', `"${family}"`)).catch(() => {})
  }
}, { root: $('sample-menu'), rootMargin: '100px' })
function sampleList() {
  if (!catalog) return
  specimens.disconnect()
  $('sample-list').replaceChildren(...catalog.fonts.map(font => {
    const button = document.createElement('button'); button.className = 'sample-option'; button.dataset.font = font.id; button.title = font.name
    button.setAttribute('aria-pressed', String(current?.known === font.id))
    const name = document.createElement('span')
    name.className = 'sample-name'; name.textContent = font.name
    button.append(name)
    specimens.observe(button)
    button.addEventListener('click', async () => {
      const trigger = sampleTrigger
      $('sample-menu').hidePopover()
      await sample(font.id)
      ;(current ? $('sample') : trigger).focus({ preventScroll: true })
    })
    return button
  }))
  $('sample-empty').textContent = 'No fonts available.'
  $('sample-empty').hidden = !!catalog.fonts.length
}
function focusSample() {
  const button = $('sample-list').querySelector('[aria-pressed="true"]') || $('sample-list').querySelector('button')
  button?.focus({ preventScroll: true })
  button?.scrollIntoView({ block: 'nearest' })
}
function positionSamples() {
  const rect = sampleTrigger.getBoundingClientRect(), menu = $('sample-menu')
  const style = getComputedStyle(menu)
  const width = menu.offsetWidth || parseFloat(style.width), height = menu.offsetHeight || parseFloat(style.maxHeight)
  // The source name ends its line, so its menu hangs from that edge; the empty-state entry opens rightward.
  const x = sampleTrigger === $('empty-sample') ? rect.left : rect.right - width
  place(menu, Math.max(20, Math.min(innerWidth - width - 20, x)), Math.max(20, Math.min(rect.bottom + 6, innerHeight - height - 20)))
}
let sampleTrigger = $('sample')
const sampleTriggers = [$('sample'), $('empty-sample')]
for (const trigger of sampleTriggers) trigger.addEventListener('click', () => { sampleTrigger = trigger })
$('sample-menu').addEventListener('beforetoggle', event => {
  const open = event.newState === 'open'
  for (const trigger of sampleTriggers) trigger.setAttribute('aria-expanded', String(open && trigger === sampleTrigger))
  if (event.newState === 'open') { sampleList(); positionSamples() }
})
$('sample-menu').addEventListener('toggle', event => {
  const open = event.newState === 'open'
  if (!open) return
  positionSamples(); focusSample()
})
$('sample-menu').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); $('sample-menu').hidePopover(); sampleTrigger.focus(); return }
  const buttons = [...$('sample-list').querySelectorAll('button')], index = buttons.indexOf(document.activeElement)
  if (event.key === 'Home' || event.key === 'End') { event.preventDefault(); buttons[event.key === 'Home' ? 0 : buttons.length - 1]?.focus(); return }
  if (['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight'].includes(event.key) && (index >= 0 || ['ArrowDown', 'ArrowUp'].includes(event.key))) {
    event.preventDefault()
    const columns = getComputedStyle($('sample-list')).gridTemplateColumns.split(' ').length
    const delta = event.key === 'ArrowDown' ? columns : event.key === 'ArrowUp' ? -columns : event.key === 'ArrowLeft' ? -1 : 1
    const next = index < 0 ? (delta > 0 ? 0 : buttons.length - 1) : (index + delta + buttons.length) % buttons.length
    buttons[next]?.focus()
  }
})
window.addEventListener('resize', () => { if ($('sample-menu').matches(':popover-open')) positionSamples() })
$('save').addEventListener('click', () => {
  if (!last) return
  const url = URL.createObjectURL(new Blob([JSON.stringify(last, null, 2)], { type: 'application/json' }))
  const a = document.createElement('a'); a.href = url; a.download = 'gpu-font-result.json'; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
})
// A font sample as drawn: not cropped, drawn over or resampled.
const whole = () => !!current?.known && !current.strokes && $('resolution').value === '100' && crop.x === 0 && crop.y === 0 && crop.width === source.width && crop.height === source.height
// The address of what is on show, which opens it again. A whole font sample is its font, ?sample=, and its text, ?text=,
// unless the default: it draws again exactly, and the page's own sample needs neither. Any other source is its crop's
// style, ?style=, the numbers the model reads from it and never its pixels. ?catalog= is the catalog searched or being
// loaded, unless the default or an opened file.
function address({ embed = embedded } = {}) {
  const url = new URL(location.href), query = url.searchParams, set = (key, value) => value ? query.set(key, value) : query.delete(key)
  const id = loading?.id ?? searchCatalog.id, named = whole() && (current.known !== SAMPLE || current.text !== TEXT)
  set('catalog', id !== catalog.catalogs[0].id && id !== 'custom' && id)
  set('sample', named && current.known); set('text', named && current.text !== TEXT && current.text)
  set('style', !whole() && last?.embedding && writeStyle(last.embedding, catalog.modelSha256))
  if (!embed) query.delete('embed')
  return url.href
}
// JSON and Link follow the matches on show, and so does the address bar once they settle: browsers refuse a page that
// rewrites its address many times a second. Only the encoder's matches have a style to link.
let addressTimer = 0
function shareable() {
  $('save').disabled = !last; $('copy-link').disabled = !last?.embedding
  clearTimeout(addressTimer)
  if (searchCatalog) addressTimer = setTimeout(() => { try { history.replaceState(history.state, '', address()) } catch {} }, 500)
}
// A link copies without the embed flag, so it opens the whole page. An iframe without clipboard access still copies
// through the older command, which needs the click alone.
async function copy(text) {
  try { await navigator.clipboard.writeText(text); return true } catch {}
  const button = document.activeElement, field = Object.assign(document.createElement('textarea'), { value: text })
  field.style.cssText = 'position: fixed; opacity: 0'; document.body.append(field); field.select()
  try { return document.execCommand('copy') } catch { return false } finally { field.remove(); button?.focus({ preventScroll: true }) }
}
let copied = 0
$('copy-link').addEventListener('click', async () => {
  const done = await copy(address({ embed: false })), button = $('copy-link')
  button.lastChild.textContent = done ? 'Copied' : 'Copy failed'; button.setAttribute('aria-label', done ? 'Link copied' : 'Copy failed')
  clearTimeout(copied); copied = setTimeout(() => { button.lastChild.textContent = 'Link'; button.setAttribute('aria-label', 'Copy link') }, 2000)
})
// A shared link's matches: its style ranked in this page's catalog. The image stays with whoever shared it. Returns why
// it can't show them, if it can't.
function showShared(text) {
  let style
  try { style = readStyle(text) } catch { return 'This link’s matches can’t be read.' }
  if (!catalog.modelSha256.startsWith(style.model)) return 'This link comes from another version of the model, so its matches can’t be shown.'
  const judged = heads ? verdict(style.embedding, heads) : null
  last = { shared: true, mode: 'encoder', embedding: Array.from(style.embedding), verdict: judged, catalog: { id: searchCatalog.id, sha256: searchCatalog.sha256 }, scoreType: 'cosine', matches: matchCatalog(style.embedding, searchCatalog, judged) }
  renderResults(); showVerdict(labelsFor(judged, last.matches)); shareable()
  message('Matches from a shared link: it holds the crop’s style, not its image.')
  return ''
}
// Opens what the address names: a font sample, else a shared style, else the page's own sample. A link it can't show
// says why over that sample, so the page never opens empty.
async function restore() {
  const id = params.get('sample'), text = params.get('text') ?? TEXT, shared = !id && isEncoder() && params.has('style')
  const failure = shared ? showShared(params.get('style')) : id && !sampleFont(id) ? `No font sample “${id}”.` : ''
  if (shared && !failure) return
  if (validText(text)) previewText = text
  const version = revision + 1
  await sample(sampleFont(id) ? id : SAMPLE)
  if (failure && version === revision) message(failure, true)
}
window.addEventListener('pagehide', () => { invalidate(); gpu?.destroy() })
// Tabs: a click, or an arrow key on a focused tab, shows its panel. The others stay in place, inert and unseen, so the
// block keeps the height of its tallest panel.
for (const list of document.querySelectorAll('[role="tablist"]')) {
  const tabs = [...list.querySelectorAll('[role="tab"]')]
  const select = tab => tabs.forEach(each => { const on = each === tab; each.setAttribute('aria-selected', String(on)); each.tabIndex = on ? 0 : -1; $(each.getAttribute('aria-controls')).inert = !on })
  list.addEventListener('click', event => { const tab = event.target.closest('[role="tab"]'); if (tab) select(tab) })
  list.addEventListener('keydown', event => {
    const at = tabs.indexOf(document.activeElement), to = { ArrowLeft: at - 1, ArrowRight: at + 1, Home: 0, End: tabs.length - 1 }[event.key]
    if (at < 0 || to === undefined) return
    const tab = tabs[(to + tabs.length) % tabs.length]
    event.preventDefault(); select(tab); tab.focus()
  })
}

// Counts a download's bytes as they arrive: decompressed, as site.json sizes them (Content-Length is the gzip size).
async function download(path, received, failure = `Could not load ${path}.`) {
  const response = await fetch(path)
  if (!response.ok) throw new Error(failure)
  return new Response(response.body.pipeThrough(new TransformStream({ transform(chunk, stream) { received(chunk.byteLength); stream.enqueue(chunk) } }))).arrayBuffer()
}
async function initialize() {
  try {
    // site.json names the model and catalogs in the repository; nothing is built or copied. The model and first catalog
    // download together, and the catalog is read once the model checks out. (Safari would fetch a <link rel=preload> twice.)
    const data = JSON.parse(new TextDecoder().decode(await download('./site.json', () => {})))
    const option = data.catalogs?.[0], total = data.modelBytes + (option ? option.bytes : 0), bar = $('loading')
    const received = total ? n => { bar.max = total; bar.value += n; $('loading-percent').textContent = `${Math.floor(bar.value / total * 100)}%` } : () => {}
    const catalogBytes = option && download(option.file, received, 'Could not load the initial catalog.')
    catalogBytes?.catch(() => {}) // Reported when awaited, after the model's own checks.
    const modelBytes = await download(data.model, received), artifact = JSON.parse(new TextDecoder().decode(modelBytes))
    if (await sha256(modelBytes) !== data.modelSha256) throw new Error('Model checksum failed.')
    model = readNetwork(artifact); heads = readHeads(artifact); catalog = data
    if (isEncoder()) {
      if (await preparationHash(artifact.preparation) !== data.preparationSha256) throw new Error('Preparation checksum failed.')
      const bytes = await catalogBytes, hash = await sha256(bytes)
      if (hash !== option.sha256) throw new Error('Catalog checksum failed.')
      const first = readCatalog(JSON.parse(new TextDecoder().decode(bytes)), data)
      shipped.set(option.file, Promise.resolve(first)); searchCatalog = { ...option, ...first, builtin: true }
      $('catalog-control').hidden = false
      // My fonts, indexed on the Catalogs page, join the menu once they fit this model; until then the menu links there.
      const stored = (await readMyFonts())?.catalog, mine = stored?.encoderSha256 === data.modelSha256 && stored.preparationSha256 === data.preparationSha256
        ? { id: 'my-fonts', name: 'My fonts', data: stored, families: new Set(stored.faces.map(f => f.familyId)).size } : null
      // All searches every shipped catalog at once; it loads them when chosen.
      const all = { id: 'all', name: 'All', families: data.families, parts: data.catalogs }
      const options = [all, ...data.catalogs, ...(mine ? [mine] : [])]
      $('catalog-list').replaceChildren(...options.map(option => {
        const button = document.createElement('button'); button.className = 'catalog-option'; button.dataset.catalog = option.id
        const label = document.createElement('span'), count = document.createElement('span')
        label.className = 'catalog-name'; label.textContent = option.name; count.textContent = option.families.toLocaleString()
        button.append(iconOf(option), label, count); button.addEventListener('click', () => selectCatalog(option)); return button
      }))
      $('add-my-fonts').hidden = !!mine
      updateCatalogLabel()
      // A link can name the catalog to search, ?catalog=my-fonts, or one source of a grouped catalog, ?catalog=collletttivo.
      const sources = data.catalogs.flatMap(group => (group.sources ?? []).map(source => ({ ...source, group })))
      const id = params.get('catalog'), wanted = [...options, ...sources].find(each => each.id === id)
      // It loads behind its own loader while the page opens its source; the matches rank again once it lands.
      if (wanted && wanted !== option) selectCatalog(wanted, null, { focus: false })
      else if (id && !wanted) { $('catalog-error').textContent = `${id === 'my-fonts' ? 'My fonts aren’t indexed in this browser' : `No catalog “${id}”`}; searching ${option.name}.`; $('catalog-error').hidden = false }
      // An embed searches the catalog its link names; its label stays, as a plain name.
      if (embedded) { $('catalog-button').disabled = true; $('catalog-button').removeAttribute('aria-label') }
      // Shipped catalogs, not the selected or imported one.
      // Fractions read as percentages; data-format names the rest. A .meter draws its fraction as a bar, or a time
      // against the slowest time beside it.
      const formats = { number: v => Math.round(v).toLocaleString('en-US'), seconds: v => `${v < 1 ? v.toFixed(2) : Math.round(v)} s`, megabytes: v => `${Math.round(v)} MB` }
      for (const figure of document.querySelectorAll('[data-metric]')) {
        const value = data.metrics[figure.dataset.metric], format = figure.dataset.format
        figure.textContent = formats[format]?.(value) ?? `${(value * 100).toFixed(1)}%`
        const peers = format === 'seconds' ? [...figure.parentElement.parentElement.querySelectorAll('[data-format="seconds"]')].map(f => data.metrics[f.dataset.metric]) : [1]
        figure.style.setProperty('--value', value / Math.max(...peers))
      }
      intro(`Finds the closest of ${data.families.toLocaleString()} font families, from ${data.catalogs.length} catalogs, to any line of text.`, data.metrics.top1, 'rendered crops of every Google Fonts family, on text it never trained on')
      // The comparison counts fonts as other finders do: every face of every shipped catalog.
      $('compared-fonts').textContent = data.catalogs.reduce((n, option) => n + option.faces, 0).toLocaleString('en-US')
      if (heads) $('compared-scripts').textContent = String(heads.script.labels.length)
      $('method-note').textContent = `A small neural network turns the crop into ${model.outputs} numbers that describe its style. Every font has its own ${model.outputs} numbers; the nearest are the matches.`
      $('network-label').textContent = 'Encoder'
    }
    try { gpu = await createNetworkGPU(model) } catch (error) { gpuReason = `CPU fallback: ${error.message}` }
    backend(); controls()
    if ($('sample-menu').matches(':popover-open')) { sampleList(); focusSample() }
    if (!isEncoder()) {
      $('catalog-size').textContent = `${data.fonts.length} families`
      const measured = data.metrics.groups.all
      intro(`Finds the closest of the ${data.fonts.length} font families it was trained on to any line of text.`, measured.accuracy, `rendered ${data.metrics.split} text`)
    }
    if (data.metrics.groups?.['length/1']) {
      $('accuracy').hidden = false
      for (const length of ['1', '2-3', '4-7', '8+']) {
        const row = document.createElement('tr')
        for (const value of [length.replace('-', '–'), data.metrics.groups[`${length}/clean`], data.metrics.groups[`length/${length}`]]) {
          const cell = document.createElement(typeof value === 'string' ? 'th' : 'td')
          if (typeof value === 'string') { cell.scope = 'row'; cell.textContent = value }
          else cell.textContent = `${(value.accuracy * 100).toFixed(1)}%`
          row.append(cell)
        }
        $('accuracy').querySelector('tbody').append(row)
      }
    }
    $('parameters').textContent = `${data.parameters.toLocaleString()} parameters, ${(data.weightBits ?? [8]).join('/')}-bit weights`
    $('workbench').removeAttribute('aria-busy'); document.body.dataset.ready = 'true'
    if (revision === 0) await restore()
    else if (current) { invalidate(true); schedule() }
  } catch (error) {
    $('workbench').removeAttribute('aria-busy'); $('backend').textContent = 'Model unavailable'; message(error.message, true)
    if (!catalog) $('sample-empty').textContent = 'Fonts unavailable.'
  }
}
initialize()
