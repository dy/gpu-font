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
  /** Families with identical letters, folded into this one. */
  siblings: string[]
}

export interface Matcher {
  /** Families best first; `[]` when the image holds no text. */
  match(image: Image): Promise<Match[]>
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
export const catalogs: Record<'google-fonts' | 'debian' | 'fontshare' | 'collletttivo' | 'other', URL>

/** Loads a catalog (Google Fonts by default) and the model it was built with. */
export function createMatcher(catalog?: string | URL, model?: string | URL): Promise<Matcher>

/** Loads the model alone. */
export function createEncoder(model?: string | URL): Promise<Encoder>
