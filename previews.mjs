// Every face of a shipped catalog but Google Fonts' has a preview stored with the site: a 64-pixel black-and-white
// picture of it (scripts/previews.py), filed under the SHA-1 of its id, so no catalog carries an address.
export async function previewPath(id) {
  const hex = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-1', new TextEncoder().encode(id))), b => b.toString(16).padStart(2, '0')).join('')
  return `previews/${hex.slice(0, 2)}/${hex.slice(2, 16)}.webp`
}
