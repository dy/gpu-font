# gpu-font

A small neural font encoder running in the browser. The demo searches **2,004 Google Fonts families** or a compatible image-reference catalog using the same frozen 128-dimensional encoder. Adding a font requires indexing its specimens, not retraining the model.

**Recognition quality is still experimental.** The [compact recovery](bench/encoder-quality.md) reaches 39.59%/63.06% top-1/top-five on historical synthetic queries from 300 unseen families. On a fresh phrase bank it reaches only 24.41%/45.72%, up from 21.24%/37.61%. The reported Lora sample now ranks first; Montserrat still ranks twelfth. Model, Google catalog and required modules total 1.72 MB. Useful rejection coverage remains unproven.

The [preview audit and catalog workflow](bench/preview-catalogs.md) cover two collected archives, raster-only compilation, genuine face metadata, and current limits. Earlier [100-family classification](bench/hundred.md), [deskew](bench/preparation.md), and [weight/italic](bench/faces.md) experiments remain available; their calibration is not applied to encoder similarity scores.

## Open the demo

With the pinned Google Fonts preview instances and original sample assets in `.data/` (see the [encoder reproduction steps](bench/encoder.md)):

```sh
npm run demo:build
npm run demo
```

Open **http://localhost:4179**. Click the image area to choose a file, or paste or drop one. With an image loaded, click the surrounding area to replace it or the × to close it. The source control shows Aa and the selected font, or an image icon and the uploaded filename. It opens a font picker with each name rendered in its typeface. Use arrow keys to browse, Home/End to jump, Enter to choose, and Escape to close. Cancelling a file choice keeps the current image. Drag the crop interior to move it; drag any edge or corner to resize. Two diagonal corner grips remain visible. Clicking an edge or corner then its destination also works. Arrow keys move the crop or focused handle; Shift + arrows resize the crop, and Home/Escape selects the full image. Matches update while adjusting the crop. Edit any match preview directly to change the comparison text; this leaves the source image intact. Detection time appears beside Font matches and measures image preparation, inference and ranking. Catalog face labels describe the nearest reference; weight/style accuracy is unmeasured.

The landing page introduces image-to-font matching and explains local inference and the model's limits. The demo is preloaded with a Lora sample. Model & results holds the loaded catalog's actual family count, measured synthetic accuracy and diagnostic export; no accuracy on real screenshots is claimed.

Choose a search catalog below Font matches. The Google Fonts catalog is included by default; compiled preview catalogs appear when available locally. “Open catalog JSON…” accepts version 1/2/3 catalogs bound to this encoder and preparation. Names alone are insufficient. Changing catalogs reuses the current image embedding; its timing then measures ranking only. A failed import retains the current catalog and result.

The Model input previews show the complete arrays actually sent to inference, with their actual aspect ratios. Blank borders are removed from the tensors. A long line produces up to three local windows; these are not recognized words and may cut letters. Normalized window embeddings are averaged and normalized again. The displayed values are cosine similarities, not certainty percentages. Download diagnostic JSON under Model & results exports the actual arrays, dimensions, crop, resolution, model/catalog hashes, embedding and ranked faces. Rejection is explicitly uncalibrated.

WebGPU performs convolution and projection; image preparation and exact cosine ranking run in JavaScript. The same encoder has a CPU fallback for unavailable/lost GPU devices. The static build is in `dist/`; use localhost or HTTPS for WebGPU. Font files load on demand for editable Google specimens. Image-only catalogs display their captured specimens; imported catalogs without local specimens show identity and score without a substitute font. Optional preview assets are separate from the recognition payload.

The demo combines rapid crop updates into the current job and the newest pending crop. Closing or replacing an image and changing resolution invalidate older results. Closing also cancels pending image and sample loads. Transparent light text receives a black matte; transparent dark text receives white. The supported input is a crop containing one font on a substantially plain background.

The percentage dropdown in the image’s bottom-right corner downsamples the selected crop to 100%, 50%, 25% or 10% before normalization. It leaves the original image intact. Returning to 100% restores the original inference pixels. Dashed input outlines are mapped back to the original image coordinates. Font licenses and attribution are available under Model & results.

## Reproduce training

The current shared encoder uses the [full-corpus pipeline](bench/encoder.md), followed by [cross-text refinement and recovery selection](bench/encoder-quality.md). The commands below reproduce the earlier classifiers. After changing the shared encoder, regenerate both Google and image-reference catalogs; previous vectors are incompatible.

Tested with Node 25.9.0 and Python 3.14.6. Dependencies are pinned:

```sh
npm ci
npx playwright install chromium
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
npm run data:import -- --source ~/projects/fontr/data/fonts_collected/google
npm run train:ten
node scripts/python.mjs -m train.hundred_data import --source ~/projects/fontr/data/fonts_collected/google
npm run train:hundred
```

The import pins existing local font files and creates regular weight-400 instances. It does not modify `fontr`. After import, that project is unnecessary. Fonts, generated images, packed tensors and model weights are ignored under `.data/`; configuration, text pools, provenance and measured reports are retained.

`train:hundred` renders Pillow and real Chromium Canvas examples, applies labeled transformations, prepares them through the same JavaScript code as the demo, then trains a PyTorch CNN. The checked-in configuration uses Apple MPS; set `training.device` in `bench/hundred.json` to `cpu` on other machines. Validation selects a checkpoint and checks int8 export. Multi-character strings are disjoint between training, validation and the final synthetic test. Single characters share the alphabet, with held-out rendering plans. Every transformation of a source stays in its split. The ten-font model initializes the 100-font features; all weights remain trainable. The report lists the additional refinement command for the selected checkpoint. The final test is a separate command, to be run only after model selection:

```sh
node scripts/hundred.mjs evaluate
npm run demo:build -- --hundred
```

The experiment report records any additional refinement commands needed to reproduce the selected model. A compatible external Python environment can be selected with `GPU_FONT_PYTHON=/absolute/path/to/python`.

## Verify

```sh
npm test
# With the demo server running:
npm run test:demo
```

The browser check requires a working GPU adapter and uses Chromium’s test flag `--enable-unsafe-webgpu`. For the shared encoder it verifies independent PyTorch/CPU/WebGPU projections, exact input-preview pixels, compatible and invalid catalog imports, repeated switching, stale responses, crop/resolution round trips, clear/replacement, CPU fallback and responsive keyboard navigation. See [bench/catalog-demo.json](bench/catalog-demo.json); screenshots are under `.data/catalog-demo-checks/`. The earlier classifier UI suite remains available after `npm run demo:build -- --hundred`. These are desktop Chromium tests, including mobile viewport widths, not physical-phone measurements.

## Input preparation

```js
import { prepareLine } from './src/line.mjs'

const input = prepareLine(imageData, { deskew: true, sampler: 'windows' })
// input.status: 'ok', 'blank', or 'low-contrast'
// input.windows: up to 3 { width, height, pixels, rect } objects
// pixels: Float32Array, dark ink = 0, white = 1
```

The caller supplies an RGBA crop. Preparation composites alpha, normalizes contrast and polarity, trims outer background, and samples proportionally scaled local windows. Each window is at most 128 × 48, with no all-white outer border. Grayscale antialiasing remains; the pixels are quantized once to byte precision for training, display and inference. Blank/low-contrast crops return no windows.

This does not recognize characters, reconstruct outlines, re-render text or estimate named typography parameters. The encoder maps pixels to vectors; a separate catalog supplies candidate identities. Changing encoder weights requires regenerating catalog vectors.

## Earlier experiments

The [ten-font classifier](bench/ten.md), [retrieval pilot](bench/report.md), [binary robustness trial](bench/robustness.md) and [controlled input comparison](bench/focus.md) remain reproducible through their existing scripts. They are not shipped in the demo. The broader project plan is in [todo.md](todo.md), and primary-source research on the gpu-* packages, JZ and font preparation is in [research.md](research.md).
