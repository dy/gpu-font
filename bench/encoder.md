# Google Fonts shared-encoder experiment

The product requirement is a frozen encoder plus a caller-provided visual catalog. Adding a font must add reference vectors without changing the encoder. This experiment trains fresh weights on the pinned Google Fonts corpus; it does not continue the classifier that already saw all 2,004 families.

## Frozen protocol

- [Family split](encoder-split.json): 1,403 training, 301 development and 300 final families. Related names, shared upstream repositories, identical binaries and observed identical render banks are connected before assignment. There are 1,099 groups. These are conservative leakage controls, not a complete historical genealogy.
- Script coverage is 161 in training, 31 in development and 26 in final families. Rare scripts concentrated in one lineage cannot appear in all splits. Noto's 228 families remain together in training. Report script slices; do not claim held-out recognition for all 162 corpus scripts.
- All convolution and projection weights start randomly, seed 20260922. No historical checkpoint, classification head or previous corpus gradient is reused.
- Identical 32/64/96/128/128 convolution channels and a learned 128-dimensional projection in both arms. Embeddings are L2-normalized for cosine retrieval. The baseline has an additional training-only classification head; it is discarded before catalog evaluation.
- Per covered alphabet: 32 fresh training strings, eight reference strings, eight development-query strings and eight final-query strings. Lengths are 2 (3 for small alphabets), 4, 8 and 12. All historical strings are excluded, and roles are disjoint after case folding. These are synthetic strings, not a natural-language screenshot benchmark.
- Pillow/RAQM and Chromium render the same selected normal face/axes. The browser receives a verified static font instance so optical size and other variable defaults cannot change the selected face. Originals stay unchanged. Both renderers provide reference and query images; training includes clean and augmented examples, queries include clean/rotated/small conditions.
- Preparation uses the existing grayscale, tight, aspect-preserving deskew/windows code. Preserve all three windows where emitted, average normalized window vectors per source, then normalize. No OCR or handcrafted descriptor is used.
- Each episode selects up to 48 training families and two different text strings per family, with opposite renderers. Prefer the same script and shared text across competing fonts. Sparse scripts fill the remaining candidates with other families. Randomize candidate order. Hard negatives come only from the current training-family reference vectors, refreshed at checkpoint selection.
- Episodic objective: symmetric cross-entropy over the two views' cosine matrix, temperature 0.1. The comparison arm uses global training-family cross-entropy with 0.02 label smoothing on the same episode construction. Observed per-script render aliases share target mass instead of receiving contradictory hard labels.
- First comparison: **20,000 optimizer updates per arm**, AdamW, learning rate 0.001 down to 0.00003, weight decay 0.0001, gradient norm cap 10. Both arms use the same seed and fresh initialization. This is a bounded first comparison, not a claim that 20,000 updates are optimal.
- Checkpoints at step 0 and every 4,000 updates. Select by the harmonic mean of macro family top-5 for development families and 80 fixed training-family controls, on clean 8+ character queries in both renderers. Search all 1,704 non-final families at every checkpoint. No final-family pixels are used during optimization or selection.
- Full development evaluation compares one versus four references per face with the same underlying reference images. The four-reference case groups by source text length. Measure strict identity top-1/top-5 by family, split, renderer, condition, script and length; quantify weight/vector int8 loss separately.
- Freeze the selected encoder and reference recipe before preparing/indexing final families. Evaluate all 2,004 candidates together; confirm adding/removing/reordering entries preserves encoder bytes. Catalogs bind both encoder and preparation hashes. A similarity score is not calibrated certainty.

Quality remains subject to the gates in [todo.md](../todo.md). A useful result requires unseen-family retrieval, independent screenshots and rejection measurements; passing the size limit alone is insufficient. Genuine weight/style variation and a training-only glyph objective remain subsequent measured comparisons.

## Reproduction

```sh
node scripts/python.mjs -m train.encoder_data split
node scripts/python.mjs -m train.encoder_data plan
node scripts/python.mjs -m train.encoder_data faces --workers 6
node scripts/python.mjs -m train.encoder_data render --workers 6
node scripts/encoder-browser.mjs development
node scripts/python.mjs -m train.encoder_data pack
node scripts/python.mjs -m train.encoder episodic --steps 20000
node scripts/python.mjs -m train.encoder classification --steps 20000
npm test
```

The two renderers can run concurrently after normal face instances are ready. Training requires MPS access on the local Mac. The `.data/encoder/` archive is separate from the deployed demo. Each arm writes `progress.json`, an exact optimizer/scheduler/random-state `last.pt` for `--resume`, a selected `best.pt`, and `encoder.json`. Resume rejects changed training code, data, split, method or total step budget. Reports are written to `bench/encoder-episodic.json` and `bench/encoder-classification.json` only after the arm finishes.

Development preparation contains **348,416 images / 689,157 windows / 3,653,928,403 tensor bytes**: 157,376 training images, 47,760 references and 143,280 queries. Both renderers completed all 1,704 non-final families. These training assets do not ship to the browser.

## Encoder and catalog contract

The experimental encoder JSON has `kind: "font-encoder"`, `dimensions: 128`, `normalization: "l2"`, the convolution/projection tensors and preparation hashes. It contains no font labels. The CPU/WebGPU runtime now supports encoder outputs directly. `src/catalog.mjs` validates and ranks compatible catalogs; the experimental demo integrates it. Reliable recognition and the public release remain gated on quality.

The separate JSON catalog has `kind: "font-catalog"`, exact `encoderSha256` and `preparationSha256`, face metadata, and packed int8 vectors with per-row scales and owners. Version 1 indexes one selected normal face per family, with one or four reference vectors. Face weight/style are source metadata, not predicted attributes. The reader rejects incompatible hashes, duplicate identities, malformed dimensions/scales/owners, missing bytes and trailing bytes. Preserve the exact encoder file bytes used to build the catalog.

After both arms finish, selection is frozen before final pixels are generated:

```sh
node scripts/python.mjs -m train.encoder_catalog select
node scripts/python.mjs -m train.encoder_data plan --phase final
node scripts/python.mjs -m train.encoder_data faces --phase final --workers 6
node scripts/python.mjs -m train.encoder_data render --phase final --workers 6
node scripts/encoder-browser.mjs final
node scripts/python.mjs -m train.encoder_data pack --phase final
node scripts/python.mjs -m train.encoder_catalog test
node checks/encoder.mjs
```

Final evaluation compares the 1,704-entry catalog with all 2,004 families, verifies unchanged encoder bytes and reordered scores, and separates float, int8-weight and int8-reference development results. A rejection threshold is chosen on a development catalog with whole family groups removed; a separate final removal measures transfer as catalog membership changes. It is not a calibrated probability or a guarantee for arbitrary user catalogs.

## Matched training result

Both arms completed 20,000 updates and selected their final checkpoint. The [frozen selection](encoder-selection.json) retains the classification-trained encoder with **one reference vector per family**. Its training-only classifier is discarded: the exported encoder has no family labels, and new families require indexing rather than optimization. This is one seeded comparison, not evidence that classification always beats episodic training.

| Training objective / references | Known-family macro top-5 | Unseen-family macro top-5 | All-query top-5 |
| --- | ---: | ---: | ---: |
| Episodic / 1 | 55.28% | 56.37% | 45.95% |
| Episodic / 4 by text length | 54.64% | 56.38% | 45.33% |
| Classification / 1 | **56.59%** | **56.47%** | **47.65%** |
| Classification / 4 by text length | 55.34% | 55.17% | 46.52% |

These are int8 weights and references, searching all 1,704 non-final families on 143,280 development queries (118,032 known-family and 25,248 unseen-family queries). Macro scores give each family equal weight; the all-query column weights every crop equally. The selected encoder's unseen-family query-weighted top-1/top-5 is **27.21% / 49.03%**. Its 74.49% quick-check unseen-family macro top-5 applies only to clean 8+ character queries and is not the full result.

The encoder has **349,920 parameters**, a 128-dimensional output and **489,155 raw JSON bytes**. Training plus per-arm evaluation took 1,234 seconds episodic and 964 seconds classification on the local Apple M4 Max/MPS, excluding data preparation. Reports preserve [episodic](encoder-episodic.json) and [classification](encoder-classification.json) per-family, script, length, renderer and condition slices, float/int8 measurements and checkpoint histories.

A development diagnostic of the selected encoder gives 72.24% macro top-5 across 956 single-script families, versus 36.55% across 748 multi-script families. This association does not establish causation. One cross-script centroid may discard useful distinctions; four centroids grouped by text length did not solve it. A fixed-encoder comparison of script-separated or clustered references is a cheaper next experiment than increasing network size. Use development data for it and reserve new final queries before subsequent selection.

## Frozen final result

The [final report](encoder-test.json) searches **all 2,004 families** with the selected frozen encoder. Final preparation contains 223,488 images / 442,232 windows: 55,872 references and 167,616 separate queries. The 300 final families contribute 24,336 queries and were never used for optimization or selection.

| Family split | Families | Query top-1 | Query top-5 | Macro family top-5 |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,403 | 24.16% | 44.92% | 53.84% |
| Development | 301 | 24.67% | 46.03% | 53.34% |
| Final unseen | 300 | **26.78%** | **47.31%** | **54.75%** |

Final unseen-family macro top-5 has a 95% lineage-bootstrap interval of **51.69–57.64%**. Intervals resample 2,000 whole lineage-group draws and are conditional on this trained seed and synthetic text set. They do not quantify independent screenshot performance. The final unseen-family browser-only query top-5 is 47.55%, versus Pillow's 47.07%.

Across all family splits, top-5 is 25.65% for 2–3 characters, 44.40% for 4–7, and 55.85% for 8+. It is 52.13% on clean inputs, 44.36% on rotated inputs, and 39.82% on reduced-resolution inputs. These are strict family identity results, not visual similarity judgments. No weight/style prediction or arbitrary screenshot reliability is established.

Indexing the 300 final families left the encoder SHA-256 unchanged, and reversed catalog order preserved scores exactly on the checked 64 queries. On the same fresh final queries, adding these distractors changes training-family macro top-5 from 55.69% to 53.84% and development-family macro top-5 from 56.04% to 53.34%. Catalog addition works mechanically; gate B's recognition-quality requirement remains unmet.

Development calibration chose cosine 0.9866603. With 60 final families removed as absent queries, it accepts only **45 of 162,480 present-font queries (0.0277%)**, of which 42 are correct (93.33%). It accepts none of 5,136 absent-font queries. This fails both the accepted-accuracy and useful-coverage requirements; it is not usable certainty. A null threshold in future reports means calibration found no qualifying operating point and all queries are rejected explicitly.

Int8 compression is not the main accuracy loss: development all-query top-5 is 47.726% with float weights/references, 47.654% with int8 weights alone, and 47.653% with int8 weights and references, a combined **0.073 percentage-point** reduction.

## Browser and storage checks

The [CPU/Metal check](encoder-runtime.json) exercised 1×1, 128×48, 127×47 and 7×3 inputs, including repeated A → A → B → C → D → A use. Maximum projection error versus PyTorch is 2.86e-6 on CPU and 4.77e-6 on Metal; maximum normalized embedding error is 1.54e-7.

On Apple M4 Max / Chromium Metal, one warm 128×48 projection takes **1.4 ms median / 2.1 ms p95** over 20 measured runs. GPU initialization with already-loaded JSON takes 37 ms. This excludes image preparation, multi-window aggregation, catalog search and network loading; phone and end-to-end release timings remain unmeasured. The local catalog-demo check measured one initial three-window crop at 30.9 ms, including preparation and ranking; this is not a latency distribution.

| Artifact | Raw bytes | Brotli bytes |
| --- | ---: | ---: |
| Encoder JSON | 489,155 | 348,272 |
| 2,004-family catalog JSON | 908,717 | 358,997 |
| Inference, preparation and catalog modules | 32,578 | 10,376 |
| Measured components | **1,430,450** | **717,645** |

These components fit the 10 MB ceiling. They include catalog parsing/ranking and exclude the website, optional font-preview assets and the training-only checkpoint. Artifacts are in [`models/encoder/`](../models/encoder/); the local demo now searches caller-selected catalogs with this encoder. See [preview catalog integration](preview-catalogs.md).

The training-stage suite passed **56 JavaScript + 70 Python tests**; the catalog integration expands this to **68 JavaScript + 71 Python tests**. The size, parity and catalog-mechanics checks pass; retrieval and rejection gates fail. Preserve this candidate as a measured baseline. Next compare script-separated/clustered references on development data before spending on a larger encoder, then evaluate a training-only glyph objective if text invariance remains weak. Independent screenshots and genuine face coverage remain necessary before a release claim.

The independent preview archive is reference material; none of its captures enter this training run or its reported query tests.
