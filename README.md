# gpu-font

Find the font in an image, in your browser. A small neural network turns a crop of text into a style vector, then ranks the fonts of a catalog by similarity. It runs on WebGPU, or on the CPU without it.

It is experimental. On rendered text in fonts it never saw, the right family is in the top five 87.3% of the time and first 68.6% (5–10 characters). Real screenshots are not measured yet. [How it was trained and measured](bench/style.md).

## What it does

- Crop a line of text from an image, pick a font sample, or draw letters.
- Get the five closest families with their matched weight and style: Bold Italic, Light and so on.
- Search Google Fonts (2,004 families, every weight and italic), Fontshare or Velvetyne, or open your own catalog JSON. New fonts need indexing, not retraining.
- Families with identical letters fold into one row: IBM Plex Sans KR and IBM Plex Sans Arabic show under IBM Plex Sans.

Not yet: hand-drawn letters, icons, symbols and emoji, and matching capitals when a catalog holds only lowercase (54% top five). The page's Goals section lists each aim and where it stands.

## Run the demo

The page runs straight from the repository, as [GitHub Pages](https://dy.github.io/gpu-font/) serves it: no build and no font files. It reads the model and catalogs from `models/encoder/` and loads every font from Google Fonts.

```sh
npm run demo          # http://localhost:4179
```

`npm run demo:build` refreshes `site.json` (catalog list, checksums, measured figures) after the model or a catalog changes. WebGPU needs localhost or HTTPS.

The repository holds scripts and JSON only. Fonts, images and PyTorch checkpoints stay in the ignored `.data/` and `*.pt` files; training and evaluation rebuild them ([corpus](bench/corpus.md), [style references](bench/style.md#reproduce), [sample fonts](bench/hundred.md)). Tested with Node 25.9.0 and Python 3.14.6:

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

## Use in code

```js
import { createMatcher } from './src/match.mjs'

const matcher = await createMatcher('models/encoder/encoder.json', 'models/encoder/google-fonts.json')

// imageData: an RGBA crop of one line of text
const [best] = await matcher.match(imageData)
console.log(best.face.family, best.score)
```

`match` returns fonts best first, identical designs folded into one entry with their `siblings`, or `[]` when the crop holds no text. It uses WebGPU when available and the CPU otherwise. A catalog built for a different model is rejected.

The steps are separate modules if you need them: `prepareLine` (`src/line.mjs`) cuts the crop into up to three 128×48 windows, `src/network.mjs` and `src/network-gpu.mjs` run the network, and `src/catalog.mjs` ranks and folds.

## Catalogs

A catalog is a JSON file of reference vectors, bound to the exact model that made them. Retraining the model means reindexing every catalog.

- **Google Fonts**: every family except color, emoji and letterless ones, from Chromium renders of each face.
- **Fontshare and Velvetyne**: built from captured previews. Only names, links and vectors are committed; the captures stay local.
- Sources without verified permission (currently MyFonts and Adobe Fonts) are built locally and never shipped. `sources.html` shows each source's terms.

## Test

```sh
npm test
npm run test:demo     # with the demo running; needs a Chromium with WebGPU
```

## More

- [bench/style.md](bench/style.md): the current encoder, its benchmark and results.
- [todo.md](todo.md): what's next. [research.md](research.md): background.
- Earlier experiments, not shipped: [ten fonts](bench/ten.md), [a hundred fonts](bench/hundred.md), [weight and italic](bench/faces.md), [deskew](bench/preparation.md), [retrieval pilot](bench/report.md).

## License

[MIT](LICENSE). This work is also offered in the spirit described by the [Krishnized license](https://github.com/krishnized/license), which does not alter the MIT terms.
