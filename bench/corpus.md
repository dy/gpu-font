# Google Fonts corpus experiment

The pinned source inventory is complete: **3,886 TTF files, 2,055 source directories, 2,467,345,124 font bytes** at `e44c4b011a820c2cbe2fd2cfa8052037d7edb571`. The recognizer experiment includes **2,004 eligible families and 162 scripts**, using one genuine default face per family.

[The inventory](corpus.json) records every source file's Git blob hash, family metadata, license evidence, available weight/style/axis information, selected face, visible letter coverage, and exclusions. The source repository organizes font families with metadata and licensing files; this experiment pins those files rather than relying on a changing web catalog. [Google Fonts repository](https://github.com/google/fonts)

51 entries are excluded from this text recognizer: 25 lack family metadata, 23 contain color font tables, two have no visible letter glyphs (including Adobe Blank), and one has insufficient letter coverage. Downloading the complete repository does not imply recognition of those excluded entries. The manifest is the coverage contract.

## Training and measurement

- 64 training strings per covered script, with two renderings each; eight validation and eight test strings, each clean, rotated and reduced in resolution. Text is shared across families with the same glyph coverage. Splits are disjoint after case folding.
- **446,976 training images, 83,808 validation images, 83,808 test images**: 614,592 images and 1,184,616 prepared windows. Training uses balanced family sampling. These are synthetic letter strings, not a natural-language or real-screenshot benchmark.
- RAQM shapes complex scripts. Character mappings to missing, empty or zero-area glyphs are excluded. Broken TrueType hint programs trigger an in-memory unhinted copy; each affected shard records that fallback. Original font files remain unchanged.
- Preparation uses the browser's exact `prepareLine(..., { deskew: true, sampler: 'windows' })`, including aspect preservation, grayscale antialiasing, tight bounds, up to three local windows, and conservative deskew. There is no OCR, skeleton extraction, Fourier transform, or descriptor retrieval path.
- The 100-family deskew encoder is widened from 64 to 128 final channels without changing its initial function. The new convolution widths are 32/64/96/128/128. A learned classification head covers the full eligible catalog.
- Identical complete training render banks are recorded per script. Their family labels share a soft target, rather than assigning contradictory hard labels to identical images. These are observed rendering equivalences, not proof that the fonts are identical for every character or style.
- Selection uses mean per-family accuracy on clean long validation strings. Full validation measures all lengths and conditions. Test data is evaluated only after selection. A separate Chromium rendering of held-out strings measures transfer; Chromium was used by the older seed model but not by corpus fine-tuning.
- The export uses int8 weights with a scale per output channel. Its raw JSON, including family metadata, must remain below **10,000,000 bytes**. Float versus int8 validation is reported separately. No softmax score is presented as calibrated certainty.

## Reproduce

From the repository root, with `requirements.txt` and npm dependencies installed:

```sh
node scripts/python.mjs -m scripts.corpus download
node scripts/python.mjs -m scripts.corpus catalog
node scripts/python.mjs -m train.corpus_data render --workers 8
node scripts/python.mjs -m train.corpus_data pack
node scripts/python.mjs -m train.corpus train --steps 40000
node scripts/python.mjs -m train.corpus test
node scripts/corpus-browser.mjs
node scripts/python.mjs -m train.corpus browser
node checks/artifact.mjs .data/corpus/model.json
npm test
```

Downloads verify every cached blob. Rendering reuses only shards whose source/preparation/renderer pins and tensor hashes still match; otherwise it regenerates them. Packing checks window offsets and exact byte boundaries. Training rejects changed data, labels, preprocessing, or initialization artifacts. `--resume` continues from the selected corpus checkpoint with a fresh optimizer; it is not an exact optimizer-state recovery.

The source files and generated tensors stay in `.data/`. They are not part of the browser download. Font preview assets must also be loaded separately and on demand; the 2.47 GB source collection is not a required recognition payload.

## Browser integration

Use the candidate's own labels and metadata. The model's `catalog[].trainingFace` describes what was trained; it is **not a predicted weight or style**. The ten-family weight/italic pilot remains a separate experiment.

```js
import { readNetwork, rankWindows } from './src/network.mjs'
import { createNetworkGPU } from './src/network-gpu.mjs'
import { prepareLine } from './src/line.mjs'

const network = readNetwork(artifact)
const gpu = await createNetworkGPU(network)

const start = performance.now()
const input = prepareLine(imageData, { deskew: true, sampler: 'windows' })
const logits = []
for (const window of input.windows) logits.push(await gpu.infer(window))
const matches = logits.length ? rankWindows(logits, network.fonts).slice(0, 5) : []
const ms = performance.now() - start
// input.windows contains the exact tensors and source polygons used for inference.
// Call gpu.destroy() when the recognizer is no longer needed.
```

This work does not change the demo layout. Its integration must load the new catalog and use the candidate's deskew preparation; simply replacing the old 100-family JSON while retaining different preprocessing would be incorrect.

## Remaining coverage

Default-face classification does not establish reliable bold/italic/continuous-weight detection. Next experiments should add genuine face/axis samples, hard confusable-family examples, tiny/partial crops, combining marks and natural-language shaping, independent screenshots, and unknown-font rejection. Color fonts and the other explicit exclusions need their own rendering coverage before they can be claimed as supported.
