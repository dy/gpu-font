# gpu-font

A small neural font classifier running locally in the browser. The current experiment classifies **100 regular font families**, including the original ten. See [the 100-font report](bench/hundred.md) for the training comparison, measured accuracy by text length, and remaining limits.

This is experimental closed-set classification. Short fragments, low resolution, rotation, and fonts outside the collection remain difficult. Synthetic accuracy does not establish accuracy on arbitrary screenshots. The descriptor path is archived.

The [scaling plan](bench/scaling.md) explains the current sampling cuts, model scores, a 10 MB deployment budget, and the preparation/training experiments before expanding to the Google Fonts corpus.

Completed follow-ups: [deskew and repeated training](bench/preparation.md) improved a new synthetic verification set from 57.5% to 62.6%; a [132 KB weight/italic pilot](bench/faces.md) reached 97.4% weight and 97.0% italic accuracy for ten families. [Retrieval continuations](bench/retrieval.md) did not beat the starting encoder. These candidates and their checkpoints are preserved under `models/`; the demo still uses the 100-family regular-only baseline. Independent screenshot accuracy and useful rejection coverage remain unproven.

## Open the demo

With trained artifacts in `.data/`:

```sh
npm run demo:build
npm run demo
```

Open **http://localhost:4179**. The Image menu combines Open image and a searchable font specimen grid. You can also click the empty image area, paste or drop a file. Drag the crop to move it; drag its handles to resize. Clicking a handle then its destination also works. Arrow keys move the crop or focused handle; Shift + arrows resize the crop, and Home/Escape selects the full image. Matches update while adjusting the crop. Edit any match preview directly to change the comparison text; this leaves the source image intact. Detection time appears beside the compute backend.

The Input previews show the complete arrays actually sent to inference, with their actual aspect ratios. Blank borders are removed from the tensors themselves. A long line produces up to three local windows; these are not detected words and can cut letters. Their predicted class probabilities are averaged. The five displayed percentages use the full 100-class distribution, with accessible model-score labels. Under Model & results, Download diagnostic JSON exports these exact arrays, their dimensions, source regions, crop, source resolution, model hash and scores. Scores are temperature-scaled on validation data; they are not a real-world accuracy guarantee or visual-similarity measure. Model & results reports whether the top score passes the validation-selected acceptance threshold. Exports also include that decision and the calibration parameters.

WebGPU performs the convolutions and classifier. Image preparation and score aggregation run in JavaScript. The same model has a CPU fallback for missing/lost GPU devices. Images remain in the browser. The static build is in `dist/`; use localhost or HTTPS for WebGPU. Font preview files are demo assets, separate from the classifier's weights.

The demo combines rapid crop updates into the current job and the newest pending crop. Image replacement and resolution changes invalidate older results. Transparent light text receives a black matte; transparent dark text receives white. The supported input is a crop containing one font on a substantially plain background.

The Resolution slider downsamples the selected source crop from 100% to 10% before normalization. It leaves the original image intact. Returning to 100% restores the original inference pixels. Input outlines are mapped back to the original image coordinates.

## Reproduce training

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
npm run demo:build
```

The experiment report records any additional refinement commands needed to reproduce the selected model. A compatible external Python environment can be selected with `GPU_FONT_PYTHON=/absolute/path/to/python`.

## Verify

```sh
npm test
# With the demo server running:
npm run test:demo
```

The browser check requires a working GPU adapter and uses Chromium's test flag `--enable-unsafe-webgpu`. It compares PyTorch, JavaScript and actual WebGPU logits, including the smallest and largest input sizes, odd dimensions, repeated/concurrent calls and device loss. It also checks exact input-preview pixels, the preview grid, upload/paste/drop, move/resize/cancel crop gestures, resolution round trips, crop races, transparency, blank/corrupt images and responsive layouts. Reports are in [bench/demo.json](bench/demo.json); screenshots are under ignored `.data/demo-checks/`. These are desktop Chromium tests, including mobile viewport widths, not physical-phone measurements.

## Input preparation

```js
import { prepareInput } from './src/input.mjs'

const input = prepareInput(imageData)
// input.status: 'ok', 'blank', or 'low-contrast'
// input.windows: up to 3 { width, height, pixels, rect } objects
// pixels: Float32Array, dark ink = 0, white = 1
```

The caller supplies an RGBA crop. Preparation composites alpha, normalizes contrast and polarity, trims outer background, and samples proportionally scaled local windows. Each window is at most 128 × 48, with no all-white outer border. Grayscale antialiasing remains; the pixels are quantized once to byte precision for training, display and inference. Blank/low-contrast crops return no windows.

This does not recognize characters, reconstruct outlines, re-render text or estimate named typography parameters. This 100-way classifier must be retrained to add classes; an extensible embedding remains a later research goal.

## Earlier experiments

The [ten-font classifier](bench/ten.md), [retrieval pilot](bench/report.md), [binary robustness trial](bench/robustness.md) and [controlled input comparison](bench/focus.md) remain reproducible through their existing scripts. They are not shipped in the demo. The broader project plan is in [todo.md](todo.md), and primary-source research on the gpu-* packages, JZ and font preparation is in [research.md](research.md).
