import { editCrop, newCrop } from './crop.mjs'
import { prepareInput } from './src/input.mjs'
import { readNetwork, inferCPU, rankWindows } from './src/network.mjs'
import { createNetworkGPU } from './src/network-gpu.mjs'

const $ = id => document.getElementById(id)
const source = $('source'), ctx = source.getContext('2d', { willReadFrequently: true })
let model, catalog, gpu, gpuReason = '', current = null, crop = null, last = null
let revision = 0, analysis = 0, dragging = null, armed = null
let scheduled = 0, running = false, pending = false
let previewText = 'Quiet rivers flow', previewValid = true
const fonts = new Map()

function message(text = '', error = false) {
  $('message').textContent = text
  $('message').toggleAttribute('data-error', error)
}
function controls() {
  $('resolution-control').hidden = !current
}
function resolution() {
  const scale = Number($('resolution').value) / 100
  const width = Math.max(1, Math.round(crop.width * scale)), height = Math.max(1, Math.round(crop.height * scale))
  $('resolution-value').textContent = `${Math.round(scale * 100)}% · ${width} × ${height}`
  return { scale, width, height }
}
function invalidate(retain = false) {
  analysis++; last = null; pending = false
  cancelAnimationFrame(scheduled); scheduled = 0
  if (!retain) { $('results').replaceChildren(); $('results-empty').hidden = false }
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
  pending = !!current && !!model
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
  resolution(); invalidate(true); message(); schedule()
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
  $('image-frame').hidden = false; $('empty').hidden = true
  source.setAttribute('aria-label', name)
  resetCrop()
}
async function loadFont(id) {
  if (!fonts.has(id)) {
    const font = catalog.fonts.find(f => f.id === id)
    if (!font) throw new Error('Font is absent from this catalog')
    fonts.set(id, new FontFace(`specimen-${id}`, `url("${font.file}")`).load().then(face => { document.fonts.add(face); return face.family }).catch(error => { fonts.delete(id); throw error }))
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
    const prepared = prepareInput(image, { ...model.preparation, background: matte })
    if (prepared.status !== 'ok') {
      $('results').replaceChildren(); $('results-empty').hidden = false; $('save').disabled = true; last = null
      $('result-summary').textContent = ''; $('detection-time').textContent = ''
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
    const matches = rankWindows(logits, model.fonts, catalog.calibration?.temperature ?? 1), elapsed = performance.now() - start
    await Promise.all(matches.slice(0, 5).map(m => loadFont(m.family)))
    if (run !== analysis) return
    last = { accepted: matches[0].score >= (catalog.calibration?.threshold ?? 1.01), calibration: catalog.calibration, mode: 'neural', backend: used, milliseconds: elapsed, model: catalog.modelSha256, source: specimen, crop: region, resolution: sampled, background: matte,
      inputs: prepared.windows.map(input => ({ ...input, pixels: Array.from(input.pixels) })), matches }
    showInput(prepared.windows, region, sampled)
    renderResults()
    $('result-summary').textContent = last.accepted ? 'Above threshold' : 'Below threshold'
    $('detection-time').textContent = `${elapsed.toFixed(1)} ms`
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
    const font = catalog.fonts.find(f => f.id === match.family)
    const item = document.createElement('li'); item.className = 'result'
    item.innerHTML = '<div class="result-top"><span class="rank"></span><span class="font-name"><a target="_blank" rel="noopener"></a></span><span class="result-score"></span></div><input class="result-preview" type="text" maxlength="80" spellcheck="false" autocomplete="off" aria-describedby="preview-error">'
    item.querySelector('.rank').textContent = String(i + 1).padStart(2, '0')
    const score = item.querySelector('.result-score')
    score.textContent = match.score < .001 ? '<0.1%' : match.score > .999 ? '>99.9%' : `${(match.score * 100).toFixed(1)}%`
    score.setAttribute('aria-label', `Model score: ${score.textContent}`)
    const link = item.querySelector('a'); link.textContent = font.name + ' ↗'
    link.href = `https://fonts.google.com/specimen/${encodeURIComponent(font.name).replaceAll('%20', '+')}`
    link.setAttribute('aria-label', `${font.name} on Google Fonts`)
    const preview = item.querySelector('.result-preview')
    preview.value = previewText
    preview.setAttribute('aria-label', `Preview ${font.name}`)
    if (i === 0) preview.id = 'preview-text'
    preview.style.setProperty('--font-specimen', `"specimen-${font.id}"`)
    fragment.append(item)
  }
  $('results').replaceChildren(fragment)
}
function updatePreview(input) {
  const text = input.value.trim() || 'Quiet rivers flow'
  previewValid = /^[\x20-\x7e]+$/.test(text) && !/[~^]/.test(text)
  $('preview-error').hidden = previewValid
  input.setAttribute('aria-invalid', String(!previewValid))
  if (!previewValid) return
  previewText = text
  for (const other of $('results').querySelectorAll('.result-preview')) {
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
  if (armed && !handle) { applyDrag(armed, start); armed = null; event.preventDefault(); return }
  armed = null
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
$('resolution').addEventListener('input', () => { resolution(); invalidate(true); message(); schedule() })
$('empty').addEventListener('click', () => $('file').click())
$('upload').addEventListener('click', () => { $('sample-menu').hidePopover(); $('file').click() })
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
  $('sample-label').textContent = catalog?.fonts.find(f => f.id === current?.known)?.name || current?.name || ''
}
const specimens = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    const button = entry.target; specimens.unobserve(button)
    loadFont(button.dataset.font).then(() => { button.dataset.loaded = 'true' }).catch(() => {})
  }
}, { root: $('sample-menu'), rootMargin: '100px' })
function sampleList() {
  if (!catalog) return
  specimens.disconnect()
  const query = $('sample-search').value.trim().toLowerCase()
  const found = catalog.fonts.filter(f => f.name.toLowerCase().includes(query))
  $('sample-list').replaceChildren(...found.map(font => {
    const button = document.createElement('button'); button.className = 'sample-option'; button.dataset.font = font.id; button.title = font.name
    button.setAttribute('aria-pressed', String(current?.known === font.id))
    const name = document.createElement('span'), specimen = document.createElement('span')
    name.className = 'sample-name'; name.textContent = font.name
    specimen.className = 'sample-specimen'; specimen.textContent = 'AaBb'; specimen.setAttribute('aria-hidden', 'true')
    specimen.style.setProperty('--font-specimen', `"specimen-${font.id}"`)
    button.append(specimen, name)
    specimens.observe(button)
    button.addEventListener('click', () => { $('sample-menu').hidePopover(); $('sample').focus({ preventScroll: true }); sample(font.id) })
    return button
  }))
  $('sample-empty').hidden = !!found.length
}
function positionSamples() {
  const rect = $('sample').getBoundingClientRect(), menu = $('sample-menu')
  const style = getComputedStyle(menu)
  const width = menu.offsetWidth || parseFloat(style.width), height = menu.offsetHeight || parseFloat(style.maxHeight)
  menu.style.left = `${Math.max(20, Math.min(innerWidth - width - 20, rect.right - width))}px`
  menu.style.top = `${Math.max(20, Math.min(rect.bottom + 6, innerHeight - height - 20))}px`
}
$('sample-menu').addEventListener('beforetoggle', event => {
  $('sample').setAttribute('aria-expanded', String(event.newState === 'open'))
  if (event.newState === 'open') { $('sample-search').value = ''; sampleList(); positionSamples() }
})
$('sample-menu').addEventListener('toggle', event => {
  const open = event.newState === 'open'
  if (!open) return
  positionSamples()
})
$('sample-search').addEventListener('input', sampleList)
$('sample-menu').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); $('sample-menu').hidePopover(); $('sample').focus(); return }
  const buttons = [...$('sample-list').querySelectorAll('button')], index = buttons.indexOf(document.activeElement)
  if (event.key === 'Enter' && document.activeElement === $('sample-search')) { event.preventDefault(); buttons[0]?.click() }
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
    const [artifact, data] = await Promise.all(['model', 'catalog'].map(async name => {
      const response = await fetch(`./assets/${name}.json`)
      if (!response.ok) throw new Error(`Could not load ${name}. Run npm run demo:build and reload.`)
      return response.json()
    }))
    model = readNetwork(artifact); catalog = data
    try { gpu = await createNetworkGPU(model) } catch (error) { gpuReason = `CPU fallback: ${error.message}` }
    backend(); controls()
    if ($('sample-menu').matches(':popover-open')) sampleList()
    $('catalog-size').textContent = `${data.fonts.length} families`
    const measured = data.metrics.groups.all
    $('metrics').textContent = `Top-1 on ${measured.count.toLocaleString()} synthetic ${data.metrics.split} crops: ${(measured.accuracy * 100).toFixed(1)}%. Real-image accuracy is unmeasured.`
    if (data.metrics.groups['length/1']) {
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
  } catch (error) { $('backend').textContent = 'Model unavailable'; message(error.message, true) }
}
initialize()
