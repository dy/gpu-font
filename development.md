# Development

## Before a pull request

- `npm test`: the JavaScript suite (also run by CI on every push) and the Python suite.
- `npm run stamp` after changing a page, stylesheet, module or the version, so the pages pin the new content and link the new release.
- `node checks/catalog-demo.mjs` with `npm run demo` running, after changing the page or the matcher: the browser suite.
- A model or catalog change goes through `node checks/encoder.mjs` and `npm run demo:build`, which rebind `site.json`; the figures in README and bench come from those files, never by hand.
- Benchmarks are frozen: a change to a pinned preparation file needs the byte-for-byte check and a `repins` entry (bench/style.md).

## Run the demo

The page runs straight from the repository, as [GitHub Pages](https://dy.github.io/gpu-font/) serves it: no build and no font files. It reads the model and catalogs from `models/encoder/`, the other catalogs' match previews from `previews/` (DaFont's from DaFont's site), and loads every font from Google Fonts.

```sh
npm run demo          # http://localhost:4179
node scripts/og.mjs   # og.png, the link preview, captured from the running page
node scripts/python.mjs scripts/previews.py   # previews/, a 64-pixel picture of each shipped face, after demo:build
```

`npm run demo:build` refreshes `site.json` (catalog list, checksums, byte sizes, measured figures) after the model or a catalog changes. The page keeps the model and each catalog in Cache Storage under those checksums, so a browser downloads again only the files whose checksum changed. WebGPU needs localhost or HTTPS.

`npm run stamp` pins each page's stylesheets and modules to their content hashes (`?v=` and an import map). GitHub Pages lets browsers cache every file for ten minutes on its own, so without it a page can run another deploy's script. Run it after changing any page, stylesheet or module; `tests/stamp.test.mjs` fails until you do.

The repository holds scripts and JSON, plus `og.png`, the link preview. Fonts, other images and PyTorch checkpoints stay in the ignored `.data/` and `*.pt` files; training and evaluation rebuild them ([corpus](bench/corpus.md), [style references](bench/style.md#reproduce), [sample fonts](bench/hundred.md)). Tested with Node 25.9.0 and Python 3.14.6:

```sh
npm ci
npx playwright install chromium
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

To use another Python, set `GPU_FONT_PYTHON=/absolute/path/to/python`.

On the page:

- Drag the crop to move it, its edges or corners to resize. Arrow keys move it, Shift + arrows resize, Home selects the whole image.
- The pencil and eraser edit any source. ⌘Z or Ctrl+Z undoes a stroke.
- Type in a match's preview to compare your own text.
- Under Model, the resolution menu (100% to 10%) shows how smaller text matches.
- Download JSON exports the exact model input, embedding and ranking.

## Modules

`createMatcher` (`src/match.mjs`) is the package entry. The steps are separate modules if you need them: `prepareLine` (`src/line.mjs`) cuts the crop into up to three 128×48 windows, `src/network.mjs` and `src/network-gpu.mjs` run the network, and `src/catalog.mjs` ranks and folds. `src/references.mjs` and `src/font-file.mjs` build My fonts catalogs on the Catalogs page.

## Catalogs

A catalog is a JSON file of reference vectors, bound to the exact model that made them. Retraining the model means reindexing every catalog, and publishing a new minor version: users pin `^0.x`, so their own catalogs keep loading until they upgrade. Share links (`?style=`) carry the model's first 8 hex digits and stop showing matches under a new model.

- **Google Fonts**: every family except color, emoji and letterless ones, from Chromium renders of each face.
- **Other free sources** (Fontsource, Fontshare, Uncut, Velvetyne and more; each under its own licence): `node scripts/catalog-files.mjs` indexes the font files pinned in `bench/open-fonts.json`, in Chromium, with the same builder My fonts uses. Only names, links and vectors are committed; the fonts stay in `.data/`.
- Sources without verified permission ship only by a decision recorded in `bench/foundries.json` (DaFont, Adobe Fonts); MyFonts stays local. WhatFontIs, whose terms say nothing on collection, ships from its own renders (`.data/previews/whatfontis-archives`). `catalogs.html` shows each source's terms.

A new shipped catalog also needs its entry in `catalogs` (`src/match.mjs`) and its id in `src/match.d.ts`; the tests fail until both list it.

## Data

Everything collected lives under `.data/` and is used for training, for a read, or for a catalog; a set collected for a read not yet written is listed as such. `bench/foundries.json` records each source's terms.

| Set | Where | Used for |
|---|---|---|
| Google Fonts, 2,004 families | `.data/fonts`, `bench/corpus.json` | training; the catalog benchmark; the Google Fonts catalog |
| Open font files, 8,517 families from GitHub, Debian, Font Library and 30 more sources, hand-downloaded ones in `.data/manual` | `.data/fonts-open`, `bench/open-fonts.json`, `bench/open-releases.json` | training (nine families in ten); the held-out read (`train.style_open`, one in ten); the file-built catalogs |
| Adobe Fonts and DaFont captures, the pilot sets | `.data/previews` | the capture-built catalogs (`scripts.preview_catalog`); MyFonts stays local, its terms a ban |
| WhatFontIs-Bench photographs | `.data/whatfontis` | development part: training views and checkpoint selection; final part: the photo read (`scripts/whatfontis.mjs`) |
| Photographs of fonts in no catalog | `.data/style/photos-absent.*` | rejection calibration at export |
| DaFont forum requests, the finder set, the photo controls | `.data/dafont`, `.data/finders`, `.data/photo-controls` | reads (`scripts/dafont.mjs`, `scripts/finders.mjs`, `scripts/photo-controls.mjs`) |
| Web screenshots: lines from the top sites' homepages, labelled by the font Chromium drew | `.data/screens` (`scripts/web_screens.mjs`) | sites split by hash: nine in ten pack as training views (`train.style_screens`), one in ten is the screens read (`scripts/screens.mjs`); other people's designs, never shipped |
| Surface textures, 383 CC0 diffuse maps from Poly Haven, none the benchmark used | `.data/textures` (`scripts/textures.mjs`) | photographed training views (`train.style_data.surface`) |
| Twins across every catalog, per default face and script, from the teacher | `.data/style/twins.json` (`train.style twins`, also written by the catalog export) | the file-built catalogs' faces (`scripts/catalog-files.mjs`), and twin credit in every read (`scripts/photos.mjs`) |
| Wikimedia Commons photographs filed by typeface | `.data/photos/commons` (`scripts/commons_typefaces.py`, `scripts/text_boxes.py`) | the Commons read (`scripts/commons.mjs`) |
| Type specimen books from archive.org | `.data/specimens` (`scripts/specimen_books.py`) | collected for a historical print read; captions still to be parsed into labels |
| Wikidata typefaces, Fonts In Use ids, popularity | `.data/canon` (`scripts/canon.mjs`) | `bench/canon.json`, the typefaces worth collecting |

## Test

```sh
npm test
npm run test:demo     # with the demo running; needs a Chromium with WebGPU
```

## Publish

The npm package holds the entry, the modules it imports, its types, the model and the shipped catalogs; `tests/package.test.mjs` checks the tarball against the imports. `npm publish` runs `node --test` first.

```sh
npm pack --dry-run    # list what ships
npm publish
```

## More

- [bench/style.md](bench/style.md): the current encoder, its benchmark and results.
- [bench/photos.md](bench/photos.md): photos, real forum requests and the shared font-finder set.
- [todo.md](todo.md): what's next. [research.md](research.md): background.
- Earlier experiments, not shipped: [ten fonts](bench/ten.md), [a hundred fonts](bench/hundred.md), [weight and italic](bench/faces.md), [deskew](bench/preparation.md), [retrieval pilot](bench/report.md).
