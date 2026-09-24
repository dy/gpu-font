# Development

## Run the demo

The page runs straight from the repository, as [GitHub Pages](https://dy.github.io/gpu-font/) serves it: no build and no font files. It reads the model and catalogs from `models/encoder/` and loads every font from Google Fonts.

```sh
npm run demo          # http://localhost:4179
node scripts/og.mjs   # og.png, the link preview, captured from the running page
```

`npm run demo:build` refreshes `site.json` (catalog list, checksums, byte sizes, measured figures) after the model or a catalog changes. WebGPU needs localhost or HTTPS.

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
- The resolution menu (100% to 10%) shows how smaller text matches.
- Download JSON exports the exact model input, embedding and ranking.

## Modules

`createMatcher` (`src/match.mjs`) is the package entry. The steps are separate modules if you need them: `prepareLine` (`src/line.mjs`) cuts the crop into up to three 128×48 windows, `src/network.mjs` and `src/network-gpu.mjs` run the network, and `src/catalog.mjs` ranks and folds. `src/references.mjs` and `src/font-file.mjs` build My fonts catalogs on the Catalogs page.

## Catalogs

A catalog is a JSON file of reference vectors, bound to the exact model that made them. Retraining the model means reindexing every catalog.

- **Google Fonts**: every family except color, emoji and letterless ones, from Chromium renders of each face.
- **Other free sources** (Fontsource, Fontshare, Uncut, Velvetyne and more; each under its own licence): `node scripts/catalog-files.mjs` indexes the font files pinned in `bench/open-fonts.json`, in Chromium, with the same builder My fonts uses. Only names, links and vectors are committed; the fonts stay in `.data/`.
- Sources without verified permission (currently MyFonts and Adobe Fonts) are built locally and never shipped. `catalogs.html` shows each source's terms.

A new shipped catalog also needs its entry in `catalogs` (`src/match.mjs`) and its id in `src/match.d.ts`; the tests fail until both list it.

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
- [todo.md](todo.md): what's next. [research.md](research.md): background.
- Earlier experiments, not shipped: [ten fonts](bench/ten.md), [a hundred fonts](bench/hundred.md), [weight and italic](bench/faces.md), [deskew](bench/preparation.md), [retrieval pilot](bench/report.md).
