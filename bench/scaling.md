# Scaling font recognition within 10 MB

2026-09-22. This is the next experiment plan, not a claim that a larger recognizer has been trained. The deployed 100-family model is unchanged: `66b98a49d1bebe9b2e56bc896f1fbe9b8e76ece52e74608aeae58ef3dd4cefbb`.

## What the screenshot exposed

Reproduced in Chromium/WebGPU with the built-in Geist Mono “Quiet rivers flow” sample. The full result is recorded in [demo.json](demo.json), `screenshotExample`:

| Rank | Family | Model score |
| --- | --- | ---: |
| 1 | Geist Mono | 97.526% |
| 2 | Geist | 0.930% |
| 3 | IBM Plex Sans | 0.248% |
| 4 | Bebas Neue | 0.141% |
| 5 | Sora | 0.129% |

The old interface hid useful evidence. Percentages are now visible, labeled **Model score**, rounded to one decimal with `<0.1%` / `>99.9%` at the extremes. They use the distribution over all 100 classes, not a redistribution among the displayed five. Exact scores remain in the export.

These are temperature-scaled per-window softmax probabilities, averaged across windows. They are neither measured real-image correctness probabilities nor visual-similarity distances. Bebas is fourth among very weak alternatives, not a strong match. An identity classifier is trained to put the correct class first; its ordering of wrong classes has no explicit perceptual-similarity supervision. A learned retrieval metric is a separate experiment, not something obtained by renaming these scores.

The example remains “Uncertain match” because the validation-selected acceptance threshold is 98.698%. The strict threshold achieved 99.80% accepted accuracy on the synthetic final set, but accepted only 9.40% of known queries. That is insufficient coverage. [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) does not establish rejection of unseen fonts or calibration on arbitrary screenshots.

The dashed boxes are **sampling windows**, not detected words or letters. `src/input.mjs` uses the ink bounding box and takes up to three evenly spread windows at a maximum 128:48 aspect ratio. It does not find character boundaries. This creates `Quie`, cuts `rivers`, and can leave portions of a long line unused. The input previews do faithfully show those tensors. Displaying whole words while still inferring on clipped windows would conceal the problem.

## Size: more fonts do not duplicate the encoder

Measured current weights: **141,458 bytes JSON / 87,172 bytes Brotli**. Model plus four unminified core modules: **93,417 bytes Brotli**, excluding the demo and preview TTFs. See [hundred-size.json](hundred-size.json).

The five-convolution encoder has 83,088 weights and 224 biases. The classifier adds 64 weights and one bias per family. Holding the encoder fixed:

`parameters = 83,312 + 65 × families`

| Families | Parameters | Int8 binary weights + float32 row scales/biases | Float32 weights/biases |
| ---: | ---: | ---: | ---: |
| 100 | 89,812 | 92,080 B | 359,248 B |
| 1,000 | 148,312 | 156,880 B | 593,248 B |
| 10,000 | 733,312 | 804,880 B | 2,933,248 B |

The binary column is `83,088 + 64N + 8(224 + N)`. This is uncompressed tensor arithmetic; it excludes JSON/base64 overhead, labels, metadata, JavaScript and intermediate buffers. The runtime currently limits artifacts to 100 classes. Neither support for 10,000 classes nor accuracy at that scale is implemented or established by this calculation.

For a retrieval model, one 128-dimensional int8 reference vector per 10,000 fonts costs **1.28 MB**; four cost **5.12 MB**, before scales, metadata and encoder weights. By contrast, 62 character-specific 64-dimensional int8 vectors for every font cost **39.68 MB**. Exhaustive per-character storage is not automatically economical. It can be restricted to a shortlist or loaded in optional packs if later justified.

Use **10,000,000 bytes for the complete recognizer + catalog + runtime before HTTP compression** as the conservative deployment ceiling. Include any shipped text detector/OCR and its vocabulary in that budget. Optional font previews are separate downloads, not a way to hide recognition dependencies. Report compressed transfer, decoded weights and peak CPU/GPU memory separately: the current int8 payload expands to float32 for computation. Preview fonts are not required to run classification.

The likely bottleneck is accuracy and training coverage, not output-layer storage. A wider encoder may be necessary; choose its size from measured accuracy versus bytes rather than multiplying the present model size by the font count.

## What the gpu-* projects actually do

Primary sources checked on 2026-09-22; sizes below have different scopes and are not directly interchangeable.

| Project | Reported deployed size / parameters | Relevant practice |
| --- | --- | --- |
| [gpu-lexer model card](https://github.com/vercel-labs/gpu-lexer/blob/main/MODEL_CARD.md) | 41,321 retained int6 weights; 28,305 B Brotli minified package | Mechanical source splitting/features; Shiki supervision; quantization-aware training; repository-disjoint evaluation. |
| [gpu-time model card](https://github.com/arikchakma/gpu-time/blob/main/MODEL_CARD.md) | 38,745 parameters; 22,501 B Brotli weights; 45,561 B Brotli package | Neural token roles; ordinary TypeScript calendar resolution; generated and teacher-checked data; int6 export. |
| [gpu-query](https://github.com/safzanpirani/gpu-query) | 29,597 parameters; README reports 40 KiB int6 weights | Schema-aware deterministic features, learned roles and a deterministic compiler; generated transfer evaluation. Feasibility spike. |

[gpu-lexer's trainer](https://github.com/vercel-labs/gpu-lexer/blob/main/packages/training/torch/train.py) and [gpu-query's trainer](https://github.com/safzanpirani/gpu-query/blob/main/spike/train.py) use offline PyTorch. These projects show how to isolate a small learned task and verify quantized browser inference. They do not establish that a tiny token-role model can distinguish thousands of visually near-identical fonts. Our 89,812-parameter model is larger than these examples, but still well below 10 MB.

We already store int8 weights. Int6 reduces the weight payload by 25% relative to int8; int4 halves it, before scales/metadata/compression. Lower precision needs quantization-aware training or measured post-training validation. Distillation from a stronger teacher and channel pruning are later options if a demonstrably better model exceeds the budget. Brotli reduces transfer, not the arithmetic or decoded memory. Do not spend accuracy on extra compression while the current model is only 87 KB compressed.

## Preparation that preserves font evidence

Recommended order for a manually selected, single-font line:

1. Estimate background/polarity and foreground geometry. Keep original grayscale antialiasing for recognition; use a binary mask only for locating ink.
2. Estimate the text baseline and deskew when there is enough evidence. Correct the baseline rotation, not italic slant. Keep a no-deskew path for very short or ambiguous crops.
3. Select complete words or groups of complete glyphs. Projection gaps/connected components are cheap candidates for separated Latin text, not universal character detectors: dots, accents, touching letters, ligatures and cursive are counterexamples. Use overlapping groups when a word is too long and verify retained ink coverage.
4. Normalize with a uniform scale and preserve width/height proportions. Do not stretch every letter into the same square or normalize away stroke weight. Keep common line-height/baseline geometry where available; a separately cropped `x` and `H` lose their relative heights. Preserve relevant measurements alongside local pixels if an ablation shows benefit.
5. Aggregate evidence from several varied glyphs in the same font. Repeated windows must not manufacture independent certainty. Calibrate rejection and coverage on separate data, including unknown fonts and mixed-font mistakes.

Uniform rescaling already exists. It cannot recover serifs erased at the original raster size. The [completed deskew experiment](preparation.md) now confirms a gain for long rotated lines: their bounding box height otherwise includes vertical displacement and reduces useful glyph resolution. Added plain background alone already leaves inference pixels unchanged; other marks can change the foreground bounds.

Any preparation change must be versioned and shared by training, inference, source outlines and the actual-input preview. Start with a frozen-model diagnostic, then retrain the selected preparation and its control on equal budgets. Do not silently replace the normalizer under the existing model/calibration.

## Letters, proportions, skeletons and Fourier transforms

**Finding letters is different from recognizing their identity.** Boundary detection can avoid cuts without full OCR. Knowing that a crop is `a` lets us compare `a` with `a` and inspect counters, terminals and proportions more directly. That can reduce content variation, but adds segmentation/transcription errors and possibly another model. Test the benefit first with known synthetic glyph identities, then with automatically recovered identities. An oracle result is an upper bound, not a deployable result.

A cheap neural experiment is a shared encoder with a character-classification head used only during training, while the font/style head remains separate. It requires no second deployed OCR model. It may help preserve shape, but it does not guarantee text-independent style: compare against the same encoder and training budget without the extra loss. Font identity and visual style also need different evaluation: near-clone families are still different identities.

Train same-font/different-text pairs and hard negatives using the same letters in similar fonts. Keep natural proportions, contrast and terminal shapes in the images. Synthetic source fonts supply exact family, glyph and axis labels. Coarse attributes such as serif, width or contrast can support auxiliary supervision where labels are trustworthy, but cannot uniquely identify thousands of fonts.

[Character-independent font identification](https://arxiv.org/abs/2001.08893) demonstrates learned comparison across different characters and unseen fonts, with performance varying by character pair. Its task is binary same/different-font verification, not 10,000-way retrieval; its architecture/accuracy is not a size or quality promise for this project. [DeepFont](https://arxiv.org/abs/1507.03196) demonstrates recognition without requiring character segmentation, and addresses synthetic/real domain mismatch. Both justify experiments, not mandatory OCR.

**Skeleton-only input is a poor default.** Skeletonization removes width, and thresholding may remove fine serifs or alter topology at small raster sizes. A medial axis plus distance-to-boundary can retain local width; see the [scikit-image skeleton/medial-axis example](https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html). Our inference: keep grayscale as primary evidence and test such geometry only as an auxiliary signal if simpler approaches fail. Skeletons are not font-neutral normalization.

**Fourier magnitude is not general font invariance.** Ideal translation changes phase but not magnitude; dropping phase also removes spatial information needed to distinguish shapes. Rotation and scaling still change the spectrum. Fourier/log-polar registration estimates transforms between related images; it does not make different characters of one font equivalent. Cropping, neighboring glyphs, rasterization and shadows also change the signal. The [scikit-image registration example](https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html) documents those transform mechanics. Applying Fourier to a skeleton cannot recover the lost stroke widths. Defer this branch until it beats direct grayscale on a controlled comparison.

## Ordered experiments and decision points

1. **Prepare a fresh development bank before more training.** Reproduce `Quiet`, `rivers`, `ri` with a neighbor sliver, margin changes, descenders/dots, low resolution, rotation and near-clone families across the current 100. Start with 8 texts per font (two per length bucket), two renderers and four views (clean, small, rotated, partial neighbor): 6,400 development images. Balance raster sizes across cases. Use new multi-character strings and seeds; single-character alphabets necessarily overlap. Treat all user-reported examples as development. Reserve new independent screenshots for final evaluation; the old final test is historical.
2. **Isolate preparation at zero training cost.** Compare current windows, boundaries from renderer ground truth (oracle), image-derived gap/component grouping, and that grouping plus deskew. Measure glyph/ink retention, false splitting, top-1/top-5 by length and condition, and preparation + inference latency. Check caps, detached dots, ligatures, and an ambiguous single glyph. Keep baseline geometry/pixels on no-op cases. An oracle advantage with no automatic advantage points to segmentation, not insufficient font classes.
3. **Spend one matched training comparison.** Use the best justified preparation, broad per-glyph size/renderer/context variation, and same-letter hard negatives. Compare a new-preparation model with the current-preparation control from the same starting encoder, seed and 20,000-update budget each. Keep the current model eligible; select on validation only. Previous 20k continuations took roughly five minutes each on the M4 Max, excluding new data generation/evaluation. This is a planning estimate, not a timing guarantee. Repeat a promising comparison with another seed before scaling.
4. **Test learned retrieval before widening the catalog.** First measure the frozen encoder's 64-dimensional features with class prototypes as a cheap baseline. Then compare classification-only training against an added supervised contrastive style objective at matched size/compute. Same font/different content should agree; the same text in similar fonts should separate. Verify withheld-family additions with a frozen encoder. Compare identity recall separately from a small independently judged visual-similarity set. Only add the training-only glyph head if errors still track character content; only consider deployed OCR if the oracle identity experiment shows worthwhile gains and automatic OCR retains them.
5. **Scale 100 → 500 → the pinned Google Fonts corpus.** Record canonical families, aliases, files, axes, licenses and script coverage from the [Google Fonts repository](https://github.com/google/fonts). Separate family identification from exact weight/italic/axis identification. De-duplicate aliases and group related versions before train/test splits. A Latin-only slice is not the full corpus: unsupported scripts, symbol fonts and missing glyphs must remain explicit until script-specific rendering/training/evaluation exists. Add unknown-font development families independently of retrieval families. Recalibrate after catalog changes; score distributions change.
6. **Choose capacity and compression from evidence.** At each scale compare accuracy, correct accepted coverage, unseen-family retrieval, bytes and browser latency. Expand the encoder only when controlled capacity comparisons justify it. Start with 64 or 128 dimensions and one prototype per family; increase prototype count only for demonstrated style/content modes. Measure int8 retrieval before int6/int4. Keep the complete mandatory download within 10 MB; do not ship an exhaustive glyph index or full font binaries to perform recognition.

Steps 1–3 and the first retrieval comparisons are now complete: [preparation and repeated training](preparation.md), [learned retrieval](retrieval.md), and a [genuine weight/italic pilot](faces.md). Deskew improves held-out synthetic accuracy; the tested retrieval continuations do not beat their starting checkpoint. The immediate next decision is a bounded capacity/glyph-supervision comparison on new development data, followed by independently verified screenshots. Full-corpus training remains behind the accuracy and rejection gates. The large-catalog candidate is a **shared neural style encoder plus compact reference vectors**, retaining the stronger current classifier baseline; it is not the archived handcrafted descriptor path.

## Verification of this change

`npm test`: 36 Node + 36 Python tests passed. `node checks/demo.mjs`: real WebGPU/CPU parity and full interaction checks passed. Every saved browser result now verifies all five displayed scores against exported probabilities, including low-score inequalities and the acceptance label. The 100-font sample sweep records the Geist Mono reproduction. Desktop and 320px screenshots were visually inspected; scores fit within the existing layout. No normalizer, runtime inference, trained weights or calibration changed.

Review follow-up: `checks/demo.mjs` now also has an isolated `scoreCases` browser sequence. It substitutes only ranking results through a test-page module route, leaving the actual renderer, export and empty-input path intact. The distributions `[.6, .3, .098, .001, .001]` twice, `[.6, .3, .098, .0010001, .0009999]`, `[.999, .001, 0, 0, 0]`, `[.9990001, .0009999, 0, 0, 0]`, and `[1, 0, 0, 0, 0]` verify exact labels at/beside both display boundaries, repeated results, zero scores and full probability. A 1×1 white PNG then clears every score and acceptance message, disables export, and selecting Lora with the first distribution restores the original labels. The real-model sample sweep separately verifies that displayed values are not renormalized over the five visible candidates. All 72 unit tests and the expanded browser suite pass. This follow-up changes only checks and documentation, with no deployed bundle or inference cost.
