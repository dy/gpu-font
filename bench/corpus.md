# Google Fonts corpus experiment

The pinned source inventory is complete: **3,886 TTF files, 2,055 source directories, 2,467,345,124 font bytes** at `e44c4b011a820c2cbe2fd2cfa8052037d7edb571` (2026-09-20). The recognizer experiment includes **2,004 eligible families and 162 scripts**, using each family's available normal face and axes.

[The inventory](corpus.json) records every source file's Git blob hash, family metadata, license evidence, available weight/style/axis information, selected face, visible letter coverage, and exclusions. The source repository organizes font families with metadata and licensing files; this experiment pins those files rather than relying on a changing web catalog. [Google Fonts repository](https://github.com/google/fonts)

51 entries are excluded from this text recognizer: 25 lack family metadata, 23 contain color font tables, two have no visible letter glyphs (including Adobe Blank), and one has insufficient letter coverage. Downloading the complete repository does not imply recognition of those excluded entries. The manifest is the coverage contract.

## Training and measurement

- 64 training strings per covered script, with two renderings each; eight validation and eight test strings, each clean, rotated and reduced in resolution. Text is shared across families with the same glyph coverage. Splits are disjoint after case folding within each covered alphabet.
- **446,976 training images, 83,808 validation images, 83,808 test images**: 614,592 images and 1,180,933 prepared windows. Training uses balanced family sampling. These are synthetic letter strings, not a natural-language or real-screenshot benchmark.
- RAQM shapes complex scripts. Character mappings to missing, empty or zero-area glyphs are excluded. Broken TrueType hint programs trigger an in-memory unhinted copy; each affected shard records that fallback. Original font files remain unchanged.
- Variable fonts explicitly use normal CSS coordinates: weight 400, width 100, upright style, clamped to the real supported ranges. Other axes retain their defaults. **180 families** differ from their binary-default axes. The initial input audit found Noto CJK binaries defaulting to Thin while Chromium used Regular; [the measured mismatch](corpus-axis-mismatch.json) records the difference. The regression test constructs a genuine variable font and checks that the prepared tensor uses the requested axis value.
- Preparation uses the browser's exact `prepareLine(..., { deskew: true, sampler: 'windows' })`, including aspect preservation, grayscale antialiasing, tight bounds, up to three local windows, and conservative deskew. There is no OCR, skeleton extraction, Fourier transform, or descriptor retrieval path.
- The 100-family deskew encoder is widened from 64 to 128 final channels without changing its initial function. The new convolution widths are 32/64/96/128/128. A learned classification head covers the full eligible catalog.
- Identical complete training render banks are recorded per script. Their family labels share a soft target, rather than assigning contradictory hard labels to identical images. These are observed rendering equivalences, not proof that the fonts are identical for every character or style.
- Selection uses mean per-family accuracy on clean long validation strings. Full validation measures all lengths and conditions. Test data is evaluated only after selection. A separate Chromium rendering of held-out strings measures transfer; Chromium was used by the older seed model but not by corpus fine-tuning.
- The export uses int8 weights with a scale per output channel. Its raw JSON, including family metadata, must remain below **10,000,000 bytes**. Float versus int8 validation is reported separately. No softmax score is presented as calibrated certainty.

The initial run completed 24,000 updates before the axis correction. Its selected 20,000-update checkpoint is preserved as `models/corpus/initial.pt`; [the initial-stage report](corpus-initial.json) records its lineage and validation history. A further 20,000 updates on corrected inputs selected the final checkpoint at update 20,000. That continuation took 786 seconds including validation and export on an Apple M4 Max. The original test inputs informed the rendering audit, so final evaluation uses newly generated v2 strings, disjoint from the original test bank as well as training and validation. No test predictions selected this model. Both final sets are now historical: further tuning requires a fresh final holdout.

## Measured candidate

[The model](../models/corpus/model.json) has **591,924 parameters**. Model plus catalog is **1,475,833 bytes**, or 443,336 bytes with Brotli. Including all five required JavaScript modules, the recognition payload is **1,503,582 bytes raw / 493,623 gzip / 452,089 Brotli** (each file compressed separately). Font previews, training checkpoints and source fonts are separate. This passes the 10 MB storage limit; it does not pass the recognition quality gates.

Validation top-1 changed from 43.013% float to 42.960% int8: a **0.053 percentage-point loss**. [Training and quantization report](corpus-training.json).

Strict exact-family results, with 2,004 candidates on every query:

| Held-out slice | Queries per renderer | Pillow top-1 / top-5 | Chromium top-1 / top-5 |
| --- | ---: | ---: | ---: |
| All | 83,808 | 43.8% / 65.0% | 28.4% / 48.7% |
| Clean | 27,936 | 54.6% / 75.9% | 33.3% / 55.1% |
| Rotated | 27,936 | 44.1% / 66.5% | 27.4% / 47.7% |
| Reduced resolution | 27,936 | 32.8% / 52.4% | 24.6% / 43.3% |
| 2–3 characters, all conditions | 20,952 | 21.6% / 41.6% | 14.4% / 31.0% |
| 4–7 characters, all conditions | 20,952 | 43.6% / 67.2% | 27.8% / 49.6% |
| 8+ characters, all conditions | 41,904 | 55.0% / 75.5% | 35.8% / 57.1% |

The table is sample-weighted. Mean per-family top-1 is 50.1% for Pillow and 32.8% for Chromium. The reports contain all script slices and leading confusions: [Pillow](corpus-test.json), [Chromium](corpus-browser.json). Even clean browser input remains weak; the 75.1% clean long-text **validation family mean** must not be presented as general recognition accuracy. The 123 observed per-script rendering-equivalence groups also create genuine exact-name ambiguity; the strict scores above do not grant credit for alternative names.

CPU and actual Metal WebGPU logits agree with the independent PyTorch reference within 0.000023 and 0.000046 respectively. Four geometries (1×1, 128×48, 127×47 and 7×3) pass the reused-instance sequence A → A → maximum → odd → small → A. On the M4 Max, 20 warm 128×48-window calls measured **0.9 ms median / 2.1 ms p95**, after two warmups; initialization was 58.2 ms. These are single-window inference timings, excluding image decoding, preparation and multi-window aggregation, not end-to-end screenshot latency. [Runtime, payload and numerical checks](corpus-runtime.json).

The next controlled experiment should mix Chromium and Pillow **training** images using shared, disjoint text banks and genuine normal faces. Compare equal update budgets on a renderer-balanced validation set before adding more capacity. Keep short strings and alias families visible in the results. Then add real face/axis variation and independently verified screenshots; the rendering gap is already large enough that simply adding classes or enlarging the head is not an adequate next step.

## Reproduce

From the repository root, with `requirements.txt` and npm dependencies installed:

```sh
node scripts/python.mjs -m scripts.corpus download
node scripts/python.mjs -m scripts.corpus catalog
node scripts/python.mjs -m train.corpus_data render --workers 8
node scripts/python.mjs -m train.corpus_data pack
node scripts/python.mjs -m train.corpus train --steps 20000 --initial models/corpus/initial.pt
node scripts/python.mjs -m train.corpus test
node scripts/corpus-browser.mjs
node scripts/python.mjs -m train.corpus browser
node checks/artifact.mjs .data/corpus/model.json
npm test
```

The selected export and checkpoint are retained in `models/corpus/`; its [README](../models/corpus/README.md) describes loading. The full suite passes 45 JavaScript and 55 Python tests, including an actual variable-font axis fixture, exact preparation bytes, malformed/final-boundary tensors, repeated calls and a 10,000-class last-row check.

Downloads verify every cached blob. Rendering reuses only shards whose source/preparation/renderer pins and tensor hashes still match; otherwise it regenerates them. Packing checks window offsets and exact byte boundaries. Training rejects changed data, labels, preprocessing, or initialization artifacts. `--initial` explicitly allows a new dataset while requiring identical family labels. `--resume` requires the same dataset and continues from the selected corpus checkpoint with a fresh optimizer; it is not an exact optimizer-state recovery. To train from the original 100-family encoder instead, omit `--initial` and use a separately recorded training budget.

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

Normal-face classification does not establish reliable bold/italic/continuous-weight detection. Next experiments should add genuine face/axis samples, hard confusable-family examples, tiny/partial crops, combining marks and natural-language shaping, independent screenshots, and unknown-font rejection. Color fonts and the other explicit exclusions need their own rendering coverage before they can be claimed as supported.
