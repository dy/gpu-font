// The "My fonts" catalog, kept in this browser's IndexedDB: font names and style vectors, never font files.
const KEY = 'my-fonts'
function store(mode, use) {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open('gpu-font', 1)
    open.onupgradeneeded = () => open.result.createObjectStore('catalogs')
    open.onerror = () => reject(open.error)
    open.onsuccess = () => {
      const db = open.result, request = use(db.transaction('catalogs', mode).objectStore('catalogs'))
      request.onsuccess = () => { db.close(); resolve(request.result) }
      request.onerror = () => { db.close(); reject(request.error) }
    }
  })
}
// { catalog, updated } or undefined.
export const readMyFonts = () => store('readonly', s => s.get(KEY)).catch(() => undefined)
export const saveMyFonts = catalog => store('readwrite', s => s.put({ catalog, updated: new Date().toISOString() }, KEY))
export const removeMyFonts = () => store('readwrite', s => s.delete(KEY))
