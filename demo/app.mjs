import { editCrop, newCrop } from './crop.mjs'
import { prepareInput } from './src/input.mjs'
import { prepareLine } from './src/line.mjs'
import { readNetwork, inferCPU, rankWindows } from './src/network.mjs'
import { createNetworkGPU } from './src/network-gpu.mjs'
import { readCatalog, rankCatalog, embedWindows, sha256, preparationHash } from './src/catalog.mjs'

const $ = id => document.getElementById(id)
const source = $('source'), ctx = source.getContext('2d', { willReadFrequently: true })
let model, catalog, gpu, gpuReason = '', current = null, crop = null, last = null
let revision = 0, analysis = 0, dragging = null, armed = null
let scheduled = 0, running = false, pending = false
let previewText = 'Quiet rivers flow', previewValid = true
let searchCatalog = null, catalogRevision = 0
const fonts = new Map()
const isEncoder = () => model?.kind === 'font-encoder'

function message(text = '', error = false) {
  $('message').textContent = text
  $('message').toggleAttribute('data-error', error)
}
function controls() {
  $('resolution-control').hidden = !current
  $('clear').hidden = !current
  $('empty').hidden = !!current
  $('open-image').setAttribute('aria-label', current ? 'Replace image' : 'Choose an image')
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
    $('results').replaceChildren(); $('results-empty').hidden = false
    previewValid = true; $('preview-error').hidden = true
  }
  $('save').disabled = true
  $('model-input').hidden = true
  $('normalized').replaceChildren(); $('input-regions').replaceChildren()
  $('result-summary').textContent = ''; $('detection-time').textContent = ''
  $('results-empty').textContent = 'No matches yet.'
  $('timing').textContent = '—'
  $('results').removeAttribute('aria-busy')
  controls()
}
function schedule() {
  pending = !!current && !!model && (!isEncoder() || !!searchCatalog)
  if (!pending) return
  $('results').setAttribute('aria-busy', 'true')
  if (!running && !scheduled) scheduled = requestAnimationFrame(() => { scheduled = 0; analyze() })
}
function backend() {
  $('backend').textContent = gpu ? 'WebGPU' : 'CPU'
  $('backend').dataset.backend = gpu ? 'webgpu' : 'cpu'
  $('device').textContent = gpu ? gpu.info : gpuReason || 'JavaScript CPU'
}
function showInput(windows, region, sampled) {
  $('normalized').replaceChildren(); $('input-regions').replaceChildren()
  for (const [index, input] of windows.entries()) {
    const canvas = document.createElement('canvas'), image = new ImageData(input.width, input.height)
    canvas.width = input.width; canvas.height = input.height; canvas.style.width = `${input.width * 2}px`
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
  $('model-input').hidden = !windows.length
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
function setImage(image, name, known = null) {
  source.width = image.width; source.height = image.height
  ctx.clearRect(0, 0, source.width, source.height); ctx.drawImage(image, 0, 0)
  current = { name, known, width: source.width, height: source.height }
  sampleLabel()
  crop = null; $('resolution').value = '100'
  $('image-frame').hidden = false
  source.setAttribute('aria-label', name)
  resetCrop()
}
function clearImage() {
  revision++
  current = null; crop = null; dragging = null; armed = null
  invalidate(); message(); sampleLabel()
  source.width = source.height = 0
  source.setAttribute('aria-label', 'Source image')
  $('image-frame').hidden = true
  $('resolution').value = '100'
  $('sample-menu').hidePopover()
  $('open-image').focus({ preventScroll: true })
}
async function loadFont(id, file = null) {
  if (!fonts.has(id)) {
    const font = catalog.fonts.find(f => f.id === id)
    if (!file && !font) throw new Error('Font is absent from this catalog')
    fonts.set(id, new FontFace(`specimen-${id}`, `url("${file || font.file}")`).load().then(face => { document.fonts.add(face); return face.family }).catch(error => { fonts.delete(id); throw error }))
  }
  return fonts.get(id)
}
async function sample(id) {
  if (!catalog) return
  if (!previewValid) { $('results').querySelector('[aria-invalid="true"]')?.focus(); return }
  const text = previewText
  const version = ++revision
  invalidate(); message('Loading sample…')
  try {
    const family = await loadFont(id)
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
    setImage(canvas, `${font.name} sample`, id)
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
      $('results-empty').textContent = 'No matches.'
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
    const embedding = isEncoder() ? embedWindows(logits) : null
    const matches = embedding ? rankCatalog(embedding, searchCatalog) : rankWindows(logits, model.fonts, catalog.calibration?.temperature ?? 1), elapsed = performance.now() - start
    if (!embedding) await Promise.all(matches.slice(0, 5).map(m => loadFont(m.family)))
    if (run !== analysis) return
    last = { accepted: embedding ? null : matches[0].score >= (catalog.calibration?.threshold ?? 1.01), calibration: embedding ? null : catalog.calibration, mode: embedding ? 'encoder' : 'neural', backend: used, milliseconds: elapsed, model: catalog.modelSha256, source: specimen, crop: region, resolution: sampled, background: matte,
      ...(embedding ? { embedding: Array.from(embedding), catalog: { id: searchCatalog.id, sha256: searchCatalog.sha256 }, scoreType: 'cosine', inferenceMilliseconds: elapsed } : {}),
      inputs: prepared.windows.map(input => ({ ...input, pixels: Array.from(input.pixels) })), matches }
    showInput(prepared.windows, region, sampled)
    renderResults()
    $('result-summary').textContent = embedding ? 'Uncalibrated' : last.accepted ? 'Above threshold' : 'Below threshold'
    $('detection-time').textContent = `${elapsed.toFixed(1)} ms`
    $('detection-time').title = 'Image preparation, inference and ranking'
    $('timing').textContent = `${elapsed.toFixed(1)} ms (prepare + infer + rank)`
    $('save').disabled = false
  } catch (error) { if (run === analysis) { invalidate(); message(`Could not analyze this crop: ${error.message}`, true) } }
  finally {
    running = false
    if (run === analysis) { controls(); $('results').removeAttribute('aria-busy') }
    if (pending) schedule()
  }
}
function renderResults() {
  if (!last) return
  previewValid = true; $('preview-error').hidden = true
  $('results-empty').hidden = true
  const fragment = document.createDocumentFragment()
  for (const [i, match] of last.matches.slice(0, 5).entries()) {
    if (isEncoder()) { fragment.append(encoderResult(match, i)); continue }
    const font = catalog.fonts.find(f => f.id === match.family)
    const item = document.createElement('li'); item.className = 'result'
    item.innerHTML = '<div class="result-top"><span class="rank"></span><span class="font-name"><a target="_blank" rel="noopener"></a><span class="font-style" title="Preview face; weight and style are not detected">Regular, 400</span></span><span class="result-score"></span></div><input class="result-preview" type="text" maxlength="80" spellcheck="false" autocomplete="off" aria-describedby="preview-error">'
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
    preview.style.setProperty('--font-specimen', `"specimen-${font.id}"`)
    fragment.append(item)
  }
  $('results').replaceChildren(fragment)
}

function encoderResult(match, index) {
  const { face, score } = match, item = document.createElement('li'); item.className = 'result'
  item.innerHTML = '<div class="result-top"><span class="rank"></span><span class="font-name"><a target="_blank" rel="noopener"></a><span class="font-style"></span></span><span class="result-score"></span></div>'
  item.querySelector('.rank').textContent = String(index + 1).padStart(2, '0')
  const link = item.querySelector('a'); link.textContent = face.family
  const url = face.sourceUrl || (searchCatalog.id === 'google-fonts' ? `https://fonts.google.com/specimen/${encodeURIComponent(face.family)}` : null)
  if (url && /^https?:\/\//i.test(url)) { link.href = url; link.textContent += ' ↗' }
  const style = item.querySelector('.font-style')
  style.textContent = face.styleName || [face.style, face.weight].filter(v => v != null).join(', ')
  style.title = 'Nearest reference face; weight and style accuracy are unmeasured'
  const value = item.querySelector('.result-score')
  value.textContent = score.toFixed(3); value.title = 'Cosine similarity, not certainty'; value.setAttribute('aria-label', `Cosine similarity: ${score.toFixed(3)}`)
  const preview = searchCatalog.builtin ? catalog.previews[face.id] : null
  if (preview?.image) {
    const image = document.createElement('img'); image.className = 'reference-preview'; image.src = preview.image
    image.alt = `${face.family}: ${preview.text}`; image.width = preview.width; image.height = preview.height
    item.append(image)
  } else if (preview?.file) {
    // Never display a fallback face as if it were a matching specimen.
    const input = document.createElement('input'); input.className = 'result-preview'; input.type = 'text'; input.maxLength = 80
    input.spellcheck = false; input.autocomplete = 'off'; input.hidden = true
    input.setAttribute('aria-label', `Preview ${face.family}`); input.setAttribute('aria-describedby', 'preview-error')
    input.readOnly = !preview.latin; input.value = preview.latin ? previewText : preview.sampleText
    input.style.setProperty('--font-specimen', `"specimen-${face.id}"`)
    if (!index && preview.latin) input.id = 'preview-text'
    item.append(input)
    loadFont(face.id, preview.file).then(() => { input.hidden = false }).catch(() => { input.remove() })
  }
  return item
}

async function selectCatalog(option, file = null) {
  const version = ++catalogRevision
  $('catalog-menu').hidePopover(); $('catalog-button').focus({ preventScroll: true })
  $('catalog-error').hidden = true; $('catalog-button').setAttribute('aria-busy', 'true')
  try {
    let bytes
    if (file) {
      if (file.size > 20 * 1024 * 1024) throw new Error('Choose a catalog smaller than 20 MB.')
      bytes = await file.arrayBuffer()
    } else {
      const response = await fetch(option.file)
      if (!response.ok) throw new Error('Could not load this catalog.')
      bytes = await response.arrayBuffer()
    }
    const hash = await sha256(bytes)
    if (option.sha256 && hash !== option.sha256) throw new Error('Catalog checksum failed.')
    const data = readCatalog(JSON.parse(new TextDecoder().decode(bytes)), catalog)
    if (version !== catalogRevision) return
    const cached = last
    searchCatalog = { ...option, ...data, sha256: hash, builtin: !file }
    invalidate(); updateCatalogLabel()
    if (cached?.embedding && current) {
      const start = performance.now(), matches = rankCatalog(cached.embedding, searchCatalog)
      const elapsed = performance.now() - start
      last = { ...cached, matches, catalog: { id: option.id, sha256: hash }, milliseconds: elapsed, cachedEmbedding: true }
      showInput(last.inputs, last.crop, last.resolution); renderResults()
      $('detection-time').textContent = `${elapsed.toFixed(1)} ms`
      $('detection-time').title = 'Catalog ranking; image embedding reused'
      $('timing').textContent = `${elapsed.toFixed(1)} ms (rank; embedding reused)`
      $('result-summary').textContent = 'Uncalibrated'; $('save').disabled = false
    } else if (current) schedule()
  } catch (error) {
    if (version === catalogRevision) { $('catalog-error').textContent = error.message; $('catalog-error').hidden = false }
  } finally { if (version === catalogRevision) $('catalog-button').removeAttribute('aria-busy') }
}
function updateCatalogLabel() {
  $('catalog-label').textContent = searchCatalog.name
  $('catalog-size').textContent = `${searchCatalog.families.toLocaleString()} families · ${searchCatalog.faces.length.toLocaleString()} faces`
  for (const button of $('catalog-list').children) button.setAttribute('aria-pressed', String(button.dataset.catalog === searchCatalog.id))
}
$('catalog-file').addEventListener('change', event => {
  const file = event.target.files[0]; event.target.value = ''
  if (file) selectCatalog({ id: 'custom', name: file.name }, file)
})
$('import-catalog').addEventListener('click', () => { $('catalog-menu').hidePopover(); $('catalog-file').click() })
function positionCatalog() {
  const menu = $('catalog-menu'), rect = $('catalog-button').getBoundingClientRect()
  menu.style.left = `${Math.max(12, Math.min(innerWidth - menu.offsetWidth - 12, rect.left))}px`
  menu.style.top = `${Math.max(12, Math.min(innerHeight - menu.offsetHeight - 12, rect.bottom + 4))}px`
}
$('catalog-button').addEventListener('click', () => {
  const menu = $('catalog-menu')
  if (menu.matches(':popover-open')) menu.hidePopover()
  else { menu.showPopover(); positionCatalog() }
})
$('catalog-menu').addEventListener('beforetoggle', event => {
  const open = event.newState === 'open'; $('catalog-button').setAttribute('aria-expanded', String(open))
  if (open) {
    const selected = $('catalog-list').querySelector('[aria-pressed="true"]') || $('import-catalog')
    for (const button of $('catalog-menu').querySelectorAll('button')) button.toggleAttribute('autofocus', button === selected)
  }
})
$('catalog-menu').addEventListener('toggle', event => { if (event.newState === 'open') positionCatalog() })
$('catalog-menu').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); $('catalog-menu').hidePopover(); $('catalog-button').focus(); return }
  if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const buttons = [...$('catalog-menu').querySelectorAll('button')], at = buttons.indexOf(document.activeElement)
  buttons[event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (at + (event.key === 'ArrowUp' ? -1 : 1) + buttons.length) % buttons.length].focus()
})
window.addEventListener('resize', () => { if ($('catalog-menu').matches(':popover-open')) positionCatalog() })
function updatePreview(input) {
  const text = input.value.trim() || 'Quiet rivers flow'
  previewValid = /^[\x20-\x7e]+$/.test(text) && !/[~^]/.test(text)
  $('preview-error').hidden = previewValid
  input.setAttribute('aria-invalid', String(!previewValid))
  if (!previewValid) return
  previewText = text
  for (const other of $('results').querySelectorAll('.result-preview')) {
    if (other.readOnly) continue
    other.setAttribute('aria-invalid', 'false')
    if (other !== input) other.value = text
  }
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
  if (armed) { applyDrag(armed, start); armed = null; event.preventDefault(); return }
  const inside = start.x >= crop.x && start.x <= crop.x + crop.width && start.y >= crop.y && start.y <= crop.y + crop.height
  dragging = { id: event.pointerId, before: { ...crop }, start, handle: handle || (inside ? 'move' : 'new'), x: event.clientX, y: event.clientY, moved: false }
  ;(handle ? event.target : $('image-frame')).focus({ preventScroll: true })
  $('image-frame').setPointerCapture(event.pointerId); event.preventDefault()
})
$('image-frame').addEventListener('pointermove', event => {
  if (dragging?.id !== event.pointerId || Math.hypot(event.clientX - dragging.x, event.clientY - dragging.y) < 3) return
  dragging.moved = true; applyDrag(dragging, point(event))
})
$('image-frame').addEventListener('pointerup', event => {
  if (dragging?.id !== event.pointerId) return
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
$('open-image').addEventListener('click', () => { $('sample-menu').hidePopover(); $('file').click() })
$('clear').addEventListener('click', clearImage)
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
  $('sample').dataset.source = current && !current.known ? 'image' : 'font'
  $('sample').title = name
  $('sample').setAttribute('aria-label', current ? `Choose a font sample. Current ${current.known ? 'font' : 'image'}: ${name}` : 'Choose a font sample')
  for (const button of $('sample-list').children) button.setAttribute('aria-pressed', String(button.dataset.font === current?.known))
}
const specimens = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    const button = entry.target; specimens.unobserve(button)
    loadFont(button.dataset.font).catch(() => {})
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
    button.style.setProperty('--font-specimen', `"specimen-${font.id}"`)
    button.append(name)
    specimens.observe(button)
    button.addEventListener('click', () => { $('sample-menu').hidePopover(); $('sample').focus({ preventScroll: true }); sample(font.id) })
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
  const rect = $('sample').getBoundingClientRect(), menu = $('sample-menu')
  const style = getComputedStyle(menu)
  const width = menu.offsetWidth || parseFloat(style.width), height = menu.offsetHeight || parseFloat(style.maxHeight)
  menu.style.left = `${Math.max(20, Math.min(innerWidth - width - 20, rect.left))}px`
  menu.style.top = `${Math.max(20, Math.min(rect.bottom + 6, innerHeight - height - 20))}px`
}
$('sample-menu').addEventListener('beforetoggle', event => {
  $('sample').setAttribute('aria-expanded', String(event.newState === 'open'))
  if (event.newState === 'open') { sampleList(); positionSamples() }
})
$('sample-menu').addEventListener('toggle', event => {
  const open = event.newState === 'open'
  if (!open) return
  positionSamples(); focusSample()
})
$('sample-menu').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); $('sample-menu').hidePopover(); $('sample').focus(); return }
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
window.addEventListener('pagehide', () => { invalidate(); gpu?.destroy() })

async function initialize() {
  try {
    const [modelBytes, data] = await Promise.all(['model', 'catalog'].map(async name => {
      const response = await fetch(`./assets/${name}.json`)
      if (!response.ok) throw new Error(`Could not load ${name}. Run npm run demo:build and reload.`)
      return name === 'model' ? response.arrayBuffer() : response.json()
    }))
    const artifact = JSON.parse(new TextDecoder().decode(modelBytes))
    if (await sha256(modelBytes) !== data.modelSha256) throw new Error('Model checksum failed.')
    model = readNetwork(artifact); catalog = data
    if (isEncoder()) {
      if (await preparationHash(artifact.preparation) !== data.preparationSha256) throw new Error('Preparation checksum failed.')
      const option = data.catalogs[0], response = await fetch(option.file)
      if (!response.ok) throw new Error('Could not load the initial catalog.')
      const bytes = await response.arrayBuffer(), hash = await sha256(bytes)
      if (hash !== option.sha256) throw new Error('Catalog checksum failed.')
      searchCatalog = { ...option, ...readCatalog(JSON.parse(new TextDecoder().decode(bytes)), data), builtin: true }
      $('catalog-control').hidden = false
      $('catalog-list').replaceChildren(...data.catalogs.map(option => {
        const button = document.createElement('button'); button.className = 'catalog-option'; button.dataset.catalog = option.id
        const label = document.createElement('span'), count = document.createElement('span')
        label.textContent = option.name; count.textContent = option.families.toLocaleString()
        button.append(label, count); button.addEventListener('click', () => selectCatalog(option)); return button
      }))
      updateCatalogLabel()
      $('metrics').textContent = `Experimental encoder. On synthetic crops of 300 unseen families: ${(data.metrics.top1 * 100).toFixed(1)}% top-1, ${(data.metrics.top5Accuracy * 100).toFixed(1)}% top-5. Real-image accuracy is unmeasured.`
      $('score-note').textContent = 'Cosine similarity compares the crop with catalog references. It is not calibrated certainty. Face labels describe references; weight and style accuracy are unmeasured.'
      $('method-note').textContent = 'Up to three tight grayscale windows are deskewed and encoded on WebGPU or CPU. Their normalized vectors are averaged, then compared with the selected catalog. Catalog changes reuse the image embedding.'
      $('scope-note').textContent = 'Crop one font on a plain background. Short crops and fonts outside the selected catalog can produce misleading matches.'
      $('network-label').textContent = 'Encoder'
    }
    try { gpu = await createNetworkGPU(model) } catch (error) { gpuReason = `CPU fallback: ${error.message}` }
    backend(); controls()
    if ($('sample-menu').matches(':popover-open')) { sampleList(); focusSample() }
    if (!isEncoder()) {
      $('catalog-size').textContent = `${data.fonts.length} families`
      const measured = data.metrics.groups.all
      $('metrics').textContent = `Top-1 on ${measured.count.toLocaleString()} synthetic ${data.metrics.split} crops: ${(measured.accuracy * 100).toFixed(1)}%. Real-image accuracy is unmeasured.`
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
    $('parameters').textContent = `${data.parameters.toLocaleString()} parameters`
    document.body.dataset.ready = 'true'
    if (revision === 0) await sample('lora')
    else if (current) { invalidate(true); schedule() }
  } catch (error) {
    $('backend').textContent = 'Model unavailable'; message(error.message, true)
    if (!catalog) $('sample-empty').textContent = 'Fonts unavailable.'
  }
}
initialize()
