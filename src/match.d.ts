/** RGBA pixels, row by row: an `ImageData`, or any object of the same shape. */
export interface Image {
  data: ArrayLike<number>
  width: number
  height: number
}

/** One font face of a catalog. */
export interface Face {
  id: string
  familyId: string
  /** Family name, such as "Inter". */
  family: string
  /** Such as "Bold Italic". */
  styleName?: string
  /** 100 to 900. */
  weight?: number
  style?: 'normal' | 'italic'
  /** ISO 15924 scripts the face draws, such as "Latn". */
  scripts?: string[]
  /** Where the family is published, for catalogs other than Google Fonts. */
  sourceUrl?: string
  [field: string]: unknown
}

export interface Match {
  /** The family's id in the catalog. */
  family: string
  /** Cosine similarity, -1 to 1: higher is closer. */
  score: number
  /** The family's closest face. */
  face: Face
  /** Same-named families with identical letters, folded into this one. */
  siblings: string[]
}

export interface Matcher {
  /** Families best first; `[]` when the image holds no text. */
  match(image: Image): Promise<Match[]>
  /** The crop's unit embedding, what `match` compares with the catalog and `rerank` with the text drawn in each face; null when the image holds no text. */
  embed(image: Image): Promise<Float32Array | null>
  /** The model behind the matcher, for `rerank`. */
  encoder: Encoder
  /** Releases the GPU device. */
  destroy(): void
}

/** A prepared 128×48 grayscale window of a text line. */
export interface LineWindow {
  width: number
  height: number
  pixels: Float32Array
}

export interface Encoder {
  /** The hashes a catalog must carry to fit this model. */
  binding: { encoderSha256: string; preparationSha256: string }
  preparation: Record<string, unknown>
  /** One projection per window, on WebGPU or the CPU. */
  project(windows: LineWindow[]): Promise<Float32Array[]>
  destroy(): void
}

/** The catalogs this package ships, by id. */
export const catalogs: Record<'google-fonts' | 'dafont' | 'adobe-fonts' | 'github' | 'debian' | 'fontlibrary' | 'whatfontis' | 'pangram-pangram' | 'other', URL>

/** Loads a catalog (Google Fonts by default) and the model it was built with. */
/** Scores below `threshold` mean the crop's font is probably in no catalog; null when the model ships none. */
export function createMatcher(catalog?: string | URL, model?: string | URL): Promise<Matcher>

/** Loads the model alone. */
export function createEncoder(model?: string | URL): Promise<Encoder>

/** The characters a line reader reads from a crop's line: `text` after greedy decoding, the class read at each eight-pixel
 * column (0 blank) and the logits, classes + 1 per column, column-major. */
export interface Reading { text: string; columns: number[]; logits: Float32Array }
export interface Reader {
  model: { classes: string; stride: number; maxWidth: number }
  /** null when the crop holds no line or the line is wider than the reader takes. */
  read(image: Image, options?: { deskew?: boolean }): Reading | null
}
/** The shipped line reader (models/reader/reader.json). */
export const READER: URL
export function createReader(reader?: string | URL): Promise<Reader>

/** A row the same-text stage scored: the catalog score kept, the cosine with the text drawn in the row's face (null when
 * the face could not be drawn) and their sum, which the shortlist is sorted by. */
export interface Reranked extends Match { textScore: number | null; combined: number }
/** The shortlist set in the crop's own text: the first `limit` rows (30) re-sorted by catalog score plus text score, the rest
 * unchanged after them; `draw` returns the row's face drawing `text` at `size` pixels, or null when it cannot. A text under
 * two characters ranks nothing. */
export function rerank(rows: Match[], text: string, options: { embedding: Float32Array | number[]; encoder: Encoder; draw: (row: Match, text: string, size: number) => Image | null | Promise<Image | null>; limit?: number }): Promise<(Match | Reranked)[]>
export const SHORTLIST: number
export const MIN_TEXT: number
