// Keep source-page labels out of the reference text when meaningful words exist.
export function providedTextAllowed(text, { minChars = 5, excluded = [] } = {}) {
  const clean = value => String(value ?? '').replace(/\s+/g, ' ').trim().toLowerCase()
  const value = clean(text)
  return value.length >= minChars && value.length <= 90 && !excluded.some(item => clean(item) === value)
}

export function hasSearchMargin(ink, box, margin = 8) {
  return ink && Math.min(ink.x - box.x, ink.y - box.y,
    box.x + box.width - ink.x - ink.width, box.y + box.height - ink.y - ink.height) >= margin
}
