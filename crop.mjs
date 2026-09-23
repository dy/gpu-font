const clamp = (n, low, high) => Math.max(low, Math.min(high, n))

// Delta is always relative to the pointer-down rectangle, avoiding accumulated rounding.
export function editCrop(rect, handle, dx, dy, bounds) {
  dx = Math.round(dx); dy = Math.round(dy)
  if (handle === 'move') return { ...rect, x: clamp(rect.x + dx, 0, bounds.width - rect.width), y: clamp(rect.y + dy, 0, bounds.height - rect.height) }
  let { x, y, width, height } = rect
  const right = x + width, bottom = y + height
  if (handle.includes('w')) { x = clamp(x + dx, 0, right - 1); width = right - x }
  if (handle.includes('e')) width = clamp(width + dx, 1, bounds.width - x)
  if (handle.includes('n')) { y = clamp(y + dy, 0, bottom - 1); height = bottom - y }
  if (handle.includes('s')) height = clamp(height + dy, 1, bounds.height - y)
  return { x, y, width, height }
}

export function newCrop(a, b, bounds) {
  const x = clamp(Math.floor(Math.min(a.x, b.x)), 0, bounds.width - 1)
  const y = clamp(Math.floor(Math.min(a.y, b.y)), 0, bounds.height - 1)
  return { x, y, width: clamp(Math.ceil(Math.max(a.x, b.x)) - x, 1, bounds.width - x), height: clamp(Math.ceil(Math.max(a.y, b.y)) - y, 1, bounds.height - y) }
}
