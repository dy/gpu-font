// The shortlist set in the crop's own words. A crop's embedding is compared with catalog rows made from fixed reference
// lines; set the shortlist's faces in the crop's text instead and the comparison is letter for letter: on 128 held-out
// photographs first place rose from 43.8% to 64.1% and the top five from 75.8% to 83.6% with the model unchanged
// (bench/photos.md). Each face the page can draw renders the text at the reference sizes, prepared and embedded as any
// crop and as the references are, the sizes averaged; its cosine with the crop's embedding joins the catalog score. A
// face the page cannot draw has no text score: its catalog score counts twice, so it neither gains nor loses against a
// face whose text reads exactly as its references do.
import { prepareLine } from './line.mjs'
import { embedWindows, unit } from './catalog.mjs'
import { SIZES } from './references.mjs'

export const MIN_TEXT = 2   // characters a text needs before it ranks anything
export const SHORTLIST = 30 // rows set in the text; the rest keep their order after them

/** `rows` as the catalog ranked them (each with `score`); `draw(row, text, size)` gives the row's face drawing `text` at `size`
 * as ImageData, or null when the page cannot draw that face. The first `limit` rows return sorted by `combined`, catalog
 * score plus text score, each with `textScore` (null when undrawable); the rows past the limit follow unchanged. */
export async function rerank(rows, text, { embedding, encoder, draw, limit = SHORTLIST }) {
  const words = typeof text === 'string' ? text.trim() : ''
  if (words.length < MIN_TEXT || !rows.length) return rows
  const query = unit(embedding), head = rows.slice(0, limit)
  const scored = await Promise.all(head.map(async row => {
    const sum = new Float32Array(query.length); let drawn = 0
    for (const size of SIZES) {
      const image = await draw(row, words, size)
      if (!image) break
      const { status, windows } = prepareLine(image, { ...encoder.preparation, deskew: true, sampler: 'windows' })
      if (status !== 'ok') continue
      const vector = unit(embedWindows(await encoder.project(windows)))
      for (let i = 0; i < sum.length; i++) sum[i] += vector[i]
      drawn++
    }
    if (!drawn) return { ...row, textScore: null, combined: row.score * 2 }
    const reference = unit(sum); let textScore = 0
    for (let i = 0; i < query.length; i++) textScore += query[i] * reference[i]
    textScore = Math.max(-1, Math.min(1, textScore))
    return { ...row, textScore, combined: row.score + textScore }
  }))
  scored.sort((a, b) => b.combined - a.combined || b.score - a.score)
  return scored.concat(rows.slice(limit))
}
