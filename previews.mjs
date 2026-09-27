// A shipped face's picture, for every catalog but Google Fonts', whose faces the page sets in their own font. DaFont shows
// each family's name in its own stored preview, loaded from its site; every other face has a 64-pixel black-and-white
// picture stored with the site (scripts/previews.py), filed under the SHA-1 of its id. No catalog carries an address.
export async function previewPath(id) {
  const hex = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-1', new TextEncoder().encode(id))), b => b.toString(16).padStart(2, '0')).join('')
  return `previews/${hex.slice(0, 2)}/${hex.slice(2, 16)}.webp`
}
// DaFont's preview of a family: /img/preview/<1st>/<2nd>/<slug, with _ for ->, then the number of the family's font file it
// shows, which the catalog gives as previewFile where it isn't 0.
const DAFONT = 'dafont:face:'
export const dafontPreview = face => {
  const stem = face.id.slice(DAFONT.length).replaceAll('-', '_')
  return `https://www.dafont.com/img/preview/${stem[0]}/${stem[1]}/${stem}${face.previewFile ?? 0}.png`
}
export const previewAddress = face => face.id.startsWith(DAFONT) ? dafontPreview(face) : previewPath(face.id)
