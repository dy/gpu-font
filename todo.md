# gpu-font — implementation plan

Build a small JavaScript library that turns a cropped image of text into ranked matches from a caller-provided font catalog, with inference in the browser. **Adding a font must require indexing only, never retraining the shared model.** The current fixed-family classifiers are measured baselines; they do not satisfy that requirement. The earlier retrieval experiments have not yet established sufficient accuracy.

The first user is a designer or developer with a screenshot and no access to the original font declaration. The useful result is a short list of plausible families, with an honest indication when the catalog cannot provide a reliable match.

Current local demo: **shared style encoder with interchangeable source collections**, defaulting to every face of 2,004 Google Fonts families. Compiled references are grouped into Fontshare, MyFonts and Velvetyne collections; phrase/alphabet/source-page alternatives stay in the benchmark. Counts reflect indexed coverage. The historical 100-family classifier remains reproducible with `npm run demo:build -- --hundred`. Encoder accuracy and rejection remain below the release gates.

Scope update, 2026-09-22: the user explicitly requested full Google Fonts expansion now. This supersedes the earlier 100/500-family sequencing gates. The **10 MB ceiling for model + catalog + required runtime**, before HTTP compression, still applies. Accuracy and rejection gates measure release readiness, not permission to expand the experiment.

Architecture clarification, 2026-09-22: the user requires interchangeable catalogs (`google-fonts.json`, `myfonts.json`, custom sets) under one frozen encoder. This is the primary acceptance criterion. The next training work below supersedes further fixed-family classification as the product path; the full source inventory, renderer fixes and existing classifier measurements remain useful.

Style-encoder result: the [style encoder](bench/style.md) (3.04M parameters, int8, typed heads) with a per-face Chromium-reference catalog puts the right family in the top five for 87.3% of 5–10 character crops from 300 held-out families (89.4% at 8–10 characters, 68.6% top-1, identical designs counted), against 60.4% for the previous encoder on the same frozen benchmark. All eight reported demo samples rank first. Model/catalog/required modules total 10.7 MB (5.3 MB with Brotli). Hand drawings, other rasterizers, real screenshots and rejection remain below release readiness.

- [x] Collect and audit two raster-preview batches: 309 reference records, 48 families and 56 genuine faces. Preserve original pixels, verified metadata and regions independently of encoder versions.
- [ ] Compare alphabet, phrase and source-page reference recipes using separately collected query screenshots before broad collection. Catalog references remain separate from evaluation queries. Remaining: each recipe has its own catalog and the swap benchmark reads the shipped one; no collected screenshots query them yet.
- [x] Define separate versioned encoder and catalog formats. The [experimental contract](bench/encoder.md) binds packed reference vectors and face metadata to exact encoder/preparation hashes, with strict decoding tests. Changing the encoder requires rebuilding its catalogs; the public JavaScript catalog API remains gated below.
- [x] Freeze lineage/alias-grouped training, development and final **family** splits before optimization: [1,403 / 301 / 300 families](bench/encoder-split.json), with script coverage and grouping evidence. The [shared-encoder experiment](bench/encoder.md) starts from fresh weights and reserves new reference/query strings; the historical 2,004-family checkpoint is not reused.
- [x] Train a 128-dimensional encoder directly for catalog selection: vary candidate membership/order, pair different texts and renderers, and include same-text confusable-font negatives. Both fresh-weight arms completed 20,000 updates. [Development selection](bench/encoder-selection.json) favors classification-feature retrieval with one reference; the training classifier is discarded. Unseen-family macro top-5 is 56.47%, below the gate.
- [x] Select using held-out-family retrieval, then freeze before indexing the 300 final families. The [final report](bench/encoder-test.json) covers all 2,004 candidates, strict top-1/top-5 slices, lineage confidence intervals, rejection and 1,704 → 2,004 catalog growth. Encoder bytes and reordered scores remain unchanged; accuracy/rejection gates fail.
- [x] Compare one versus four references per selected genuine face at equal source-image budgets; one wins this experiment. Preserve actual weight/style metadata without claiming to predict it. Int8 weight/vector loss is 0.073 percentage points; [encoder/catalog/current runtime components](bench/encoder-runtime.json) total 1,430,450 bytes. The complete released package is still gated below.
- [ ] Keep OCR out of the initial runtime. Use known glyph identities during synthetic training to test whether a glyph-aware auxiliary objective helps; only add OCR or glyph alignment after a measured benefit justifies its size and latency. Remaining: the glyph-aware objective is untested; OCR stays out.
- [x] Implement raster-preview → frozen encoder → versioned catalog compilation and a demo selector/import. Preserve genuine faces, exact encoder/preparation bindings and candidate-only ranking. WebGPU computes embeddings; exact cosine ranking currently runs in JavaScript.
- [ ] Measure independent screenshot queries and meet the published quality gates before releasing the catalog API as reliable recognition. Measured on photos and forum requests, 43–54% top five ([photos](bench/photos.md)); the gates are not met.
- [x] Compare script-separated and clustered references on development queries, then measure a cross-text encoder continuation and case-specific reference specimens. [The first refinement](bench/encoder-quality.md) improves aggregate results but worsens the reported Lora crop, so it is held back.
- [x] Train with lowercase, title-case and uppercase strings across both renderers; preserve held-out families and confirm on a second frozen phrase bank. [Top-five improves from 40.56% to 66.89%](bench/encoder-case-quality.json), but Lora/Montserrat remain at ranks 9/26, so the model is held back.
- [x] Evaluate a larger encoder and retain the smaller candidate under a declared recovery rule. The 1.86 MB encoder does not improve both reported failures; the selected compact update passes CPU/Metal parity and stays below 10 MB.
- [ ] Test a training-only glyph objective at a fixed budget to reduce the measured text dependence. Preserve whole glyphs and known synthetic labels; measure fresh text and Montserrat/Roboto confusions before adding runtime OCR.
- [x] Define style from the fonts themselves: a letter-by-letter distance over every face and 22 scripts, renderer-noise twins, Google's taxonomy for broad class. See [the style encoder](bench/style.md).
- [x] Train on fresh text every step (all cases and scripts, screenshot damage, the browser's own preparation) with identity, style-geometry and typed-head losses; freeze a 9,409-query Chromium benchmark first. Held-out top-five 84.3%.
- [x] Index every face (weights, italics) with Chromium references; skip fonts that cannot draw the detected script; name matched faces in the demo and fold families with identical letters in the crop's script into one row.
- [x] Preview each match in its own face (the static instance its references were rendered from), compile collected catalogs with one reference per capture, and measure catalog swap: 100% top-five in each source's catalog, 97.1% among Google's families.
- [ ] Collect script, brush, handwriting and contemporary serif faces for the Adobe Fonts and MyFonts catalogs; the archive has none, and those styles score well among Google families (brush 96.7%, display 90.4% held-out top-five), so the gap is coverage. Blocked: MyFonts' terms forbid collecting and Adobe's are unverified; both await permission.
- [ ] Index a user's own font files into a catalog in one command (Chromium references, as for Google Fonts); today a catalog comes from the Google pipeline or a preview archive. In the browser, My fonts → Download JSON exports one, drawn by whichever browser indexed it.
- [ ] Preview faces outside Google Fonts: Other, Debian and file-indexed My fonts show names only, so a foundry's own catalog (`?catalog=collletttivo`) shows no letters. OFL families pinned at GitHub commits could load from jsDelivr; decide per source whose terms allow serving the files.
- [ ] Make hand-drawn search work: the synthetic sketch slice stays at 5% top-five and 39% category agreement; two sketch fine-tunes failed. Collect real drawings and test a category-level target.
- [ ] Measure Safari, Firefox, Windows ClearType and phone screenshots against the Chromium-reference catalog; keep the averaged catalog as the fallback if they disagree.
- [x] Train longer, then 1.5× wider with the function preserved: held-out top-five 84.3% → 87.3%, top-one 62.5% → 68.6%, degraded 75.0% → 80.0%. A view-to-view loss with opposite-case pairs taught case but cost overall accuracy and was rejected. See [second step](bench/style.md#second-step-longer-then-wider).
- [ ] Train style to cross case and script: held-out capitals found among lowercase-only references 54.0% of the time, other scripts from Latin only 43.6%, Chinese 15.4% ([breakdown](bench/style.md#by-requirement)); collected catalogs often hold one case.
- [ ] Raise non-Latin scripts (70%) and Hanzi (56%) toward Latin, and seen families at 8–10 characters from 91.7% toward the 95% target.
- [x] Measure same-text versus different-text retrieval across renderers. The [opened-development diagnostic](bench/encoder-text-dependence.json) exposes a substantial text-dependence gap; it does not establish the benefit of OCR.

The approach is learned metric/prototype retrieval, not the archived handcrafted descriptor. [Character-independent font identification](https://arxiv.org/abs/2001.08893) provides evidence for learned comparisons on unseen fonts, but its pair-verification accuracy is not equivalent to selecting from thousands of candidates. The earlier [retrieval report](bench/retrieval.md) is historical: both 12,000-update continuations selected step zero. The [fresh Google Fonts encoder comparison](bench/encoder.md) now supplies the relevant held-out-family evidence; it has not met the release gates.

- [x] Download and verify all 3,886 pinned font binaries, metadata and licensing files; account for every family and exclusion in [the coverage manifest](bench/corpus.json).
- [x] Prepare all 2,004 eligible families through the shared browser normalizer; reject invisible glyphs, record hinting fallbacks and identical rendering groups, and keep text splits disjoint.
- [x] Expand CPU/WebGPU support to 10,000 labels and a wider encoder; pool once per input rather than once per candidate. Verify 2,055-class numerical parity on Metal.
- [x] Catch the variable-font default/normal-weight mismatch with a tensor audit and a real variable-font regression fixture; correct 180 families and reserve fresh final-test strings.
- [x] Continue for 20,000 updates from the selected initial corpus checkpoint, compare float/int8 accuracy, and preserve the candidate: **1.50 MB including runtime**, with 0.053 percentage-point quantization loss.
- [x] Run frozen Pillow and Chromium tests (83,808 queries each), publish length/condition/script slices, and verify the final artifact on CPU/Metal WebGPU. Browser top-1/top-5 is **28.4%/48.7%**; the quality gate is not met.
- [x] Compare mixed Chromium/Pillow training with the current renderer-only continuation at equal update budgets. Use renderer-balanced validation and new final text banks; measure short crops and exact-name ambiguity separately before changing model capacity. Superseded: the style encoder mixes generated views with stored Chromium renders in every batch ([training views](bench/style.md#training-views)); no equal-budget comparison was run.
- [x] Add genuine weight/style/axis samples to the full catalog. Use confusable-family validation examples to select the next training improvement; keep the existing final tests historical after opening them. Done: the catalog indexes all 8,132 faces, weights and italics included.
- [ ] Build a verified screenshot benchmark and calibrate unknown/ambiguous results before claiming reliable recognition from arbitrary images. The photo and forum sets exist ([photos](bench/photos.md)); calibrating unknown results remains.

- [x] Commit the baseline and trained checkpoint; simplify the image menu, edit previews inline, show detection time, and move diagnostic export/acceptance out of the primary results.
- [x] Restore model scores beside all five matches and reproduce the Geist Mono/Bebas example with exact exported probabilities. Scores are not visual-similarity measurements.
- [x] Account for 10,000-class storage: a fixed current encoder plus expanded int8 head needs about 0.805 MB of binary tensors, including scales/biases. This is a size projection, not trained accuracy.
- [x] Build a fresh 6,400-image preparation development bank with unseen multi-character strings, two renderers, three source sizes and clean/small/rotated/neighbor views. Exact user screenshots remain development cases to preserve separately.
- [x] Compare existing windows, renderer word boxes, image-derived groups and deskew with frozen weights; measure geometric ink/word coverage, recognition and preparation time. [Results](bench/preparation.md).
- [x] Run the matched 20k-update preparation/control comparison with two seeds. Preserve the selected normalizer and model together; verify on 6,400 new known-family crops plus 384 absent-family crops. The deskew candidate improves top-1 from 57.5% to 62.6%, but accepted coverage is only 5.8%.
- [x] Test compact learned retrieval and seven frozen-encoder catalog additions, including one/four references and a trained-head projection. Compare paired classification and contrastive training at 12k updates each. Both retain the original model. [Results](bench/retrieval.md).
- [x] Train genuine weight-400/700 and upright/italic faces for the original ten families, recover int8 accuracy with QAT, run a separate test and CPU/WebGPU parity, and commit the 132 KB artifact. [Results](bench/faces.md).
- [ ] Compare one wider encoder and a training-only glyph head with matched budgets on new development data; preserve glyph boundaries without sacrificing raster height. The newly opened verification set is now historical. Remaining: the glyph head; the wider encoder is measured ([second step](bench/style.md#second-step-longer-then-wider)).
- [ ] Measure visual similarity separately from exact identity with verified human judgments. Do not interpret classifier probabilities as similarity scores.
- [x] Integrate the shared encoder and its pinned preparation into the demo with explicit user authorization. Preserve exact input previews, raw cosine scores, catalog switching/import and timing. No classifier calibration is transferred.
- [x] Bring forward the pinned Google Fonts experiment with explicit family/face/script coverage. Training, calibration and release readiness are tracked above.

- [x] Import 100 pinned regular families and reserve 13 unknown families across validation/test.
- [x] Train on single glyphs, random short strings, words and phrases in two renderers, with edge cuts, scale changes, rotation, shadows and margins.
- [x] Restore the searchable specimen grid; add icon-based opening, clickable empty state, movable/resizable crop handles and source-resolution control.
- [x] Verify actual input preview pixels, plain-margin invariance, crop/resize/cancel interaction, and resolution round trips.
- [x] Complete the bounded 100-font model comparison and separate 21,600-image final test; publish per-length and per-condition results.
- [x] Calibrate uncertainty on validation only and report acceptance coverage. Final accepted accuracy is 99.80%, but coverage is only 9.40%; the reliability gate remains unmet.
- [x] Inspect rotated/downscaled windows and compare inexpensive line deskew on fresh data before adding OCR; rotated 8+ character verification improves from 43.5% to 85.5%.
- [x] Add single/short strings and partial neighboring-letter views to development; publish the weak length slices rather than assuming any glyph identifies a font.
- [x] Collect verified independent screenshots for the included families, split by source, and reserve a new final test before further tuning. Existing ten-font and 100-font tests are historical. Done with photos rather than screenshots: WhatFontIs-Bench, DaFont forum requests and the finder set, split into development and final ([photos](bench/photos.md)).
- [x] Establish held-out weight/italic evidence for the ten-family pilot.
- [x] Extend genuine face/axis coverage beyond the ten-family pilot, using the full pinned catalog's real available faces and axes. Done: every face of every family, in training and in the catalog.

Historical descriptor scores are archived, not an active product path. Earlier evidence and the reason for local windows remain in [bench/focus.md](bench/focus.md).

## Goals and limits

- [x] Recognize a measured, published subset of Google Fonts first, then expand toward the full catalog. Done: all 2,004 eligible families, measured ([style](bench/style.md)).
- [ ] Add Adobe Fonts, MyFonts, independent foundries, and custom fonts through source-specific catalog packs and permitted local imports. Done for independent foundries (Other) and custom fonts (My fonts, exportable as JSON); Adobe Fonts and MyFonts await permission.
- [ ] Extend to historical typefaces, distinguishing identifiable digital revivals from uncertain printed specimens.
- [ ] Deliver a small runtime plus one self-contained JSON data file for the default model and catalog; accept reproducible PRs that add font entries without retraining. Remaining: model, catalog and modules total 10.7 MB, over the 10 MB ceiling; no pull-request workflow for font entries. The model and each catalog are separate files by design, so catalogs swap.

“All fonts” is a coverage direction, not a verifiable release promise. Maintain a coverage manifest with included, unsupported, and access-blocked entries. Identical outlines, missing distinguishing glyphs, and degraded images can prevent exact identification even when the correct font is indexed.

The conversation's 20–50k parameters, 32 dimensions, approximately 100 KB package, and recognition of unseen fonts are hypotheses to test. A 32-dimensional float32 vector takes 128 bytes; 32 bytes requires int8 quantization, excluding metadata. One JSON file still requires executable JavaScript to interpret it.

## Sequence and decision gates

The phases below preserve the original roadmap. The demo was brought forward on 2026-09-21 and full Google Fonts experimentation on 2026-09-22 at the user's request. Neither expansion by itself satisfies the accuracy, rejection, hardware or release gates. Phases 5–7 describe release work; the current corpus experiment brings part of phase 8 forward.

These are proposed engineering targets, not achieved results or literature benchmarks. Freeze them and the evaluation protocol in phase 0; record any later scope change explicitly.

| Gate | Evidence required to continue |
| --- | --- |
| A: useful retrieval | On the locked browser screenshot set, macro family top-5 recall ≥85% for training families and ≥75% for families withheld from training. On independently collected screenshots, top-5 ≥75%, reported separately. |
| B: addition without retraining | Add 100 withheld families to the frozen encoder's catalog. Reach gate A's unseen-family target while comparing against all 500 indexed families. Report the effect on existing-family recall. |
| C: honest rejection | With thresholds selected on validation data, ≥95% correct top-1 among accepted in-catalog queries at ≥50% acceptance; accept ≤10% of queries whose families are absent from the catalog. Report ambiguous and degraded inputs separately. |
| D: small, usable browser build | Required runtime + model + catalog ≤10,000,000 bytes before HTTP compression, including any OCR/detector used for recognition. Report Brotli bytes and optional font-preview downloads separately; warm end-to-end p95 ≤100 ms on a named laptop and ≤250 ms on a named midrange phone. Measure cold load, decoded/peak memory, and CPU fallback separately. |
| E: release | All automated tests pass; browser interaction checks pass; source provenance, reproduction steps, measured limits, and contribution workflow are published. |

Accuracy gates apply to the declared v0.1 input scope. Report sample counts, per-family results, and confidence intervals alongside aggregate scores. Measure retrieval before rejection, so abstaining cannot inflate recall. Do not tune against the final test set.

## 0. Fix the experiment contract

- [x] Inspect the gpu-* training implementations and existing screenshot-preparation tools; record primary sources and limitations in [research.md](research.md).
- [x] Define initial input: a tight crop of one horizontal line in one Latin-script family, upright regular text, approximately 6–30 visible characters. Caller supplies the crop. Keep automatic page/text-region detection separate from the initial recognition experiment; see [the contract](bench/protocol.md).
- [ ] Encode the declared 12–48 CSS px, device pixel ratios 1 and 2, and rendering conditions in a real benchmark manifest; include moderate rescaling/compression, short text, and tiny text as explicit stress cases. Remaining: 16–64 px are measured ([references at three sizes](bench/style.md#third-step-references-at-three-sizes)); 12 px and pixel ratio 2 are not.
- [x] Define canonical family identity as the primary label; record source/style metadata separately. The [pilot font manifest](bench/fonts.json) now records names, source/instance hashes, embedded licenses, and regular-face axes. Exact face/axis prediction and full alias recovery remain later work.
- [x] Check whether JZ can compile a representative typed-array pixel kernel. The retained [probe](bench/jz-probe.mjs) verifies JS/WASM numerical parity, not speed or font-recognition quality.
- [x] Build the small [input experiment](bench/protocol.md): 20 known families, controlled renderings, independently captured development crops, a shared grayscale `prepare` function, and neural recognition. Preserve antialiasing; test binary thresholding as an alternative, not an assumption. Superseded: the style encoder's frozen benchmark and the photo sets replace it.
- [x] Complete the synthetic portion of that experiment: 20 hash-pinned families, 480 controlled Pillow crops, 320 Chromium queries, 2,240 separate training crops, shared normalization, grayscale/binary descriptor comparison, and an inspected contact sheet. Independent screenshots remain outstanding.
- [ ] Freeze the gates above, split rules, target laptop/phone, supported browser versions to test, and a small experiment budget. Start with at most three architecture sizes and two seeds per size; record hardware time and cost before scaling. Remaining: gates and splits are frozen and the laptop is an M4 Max; the phone, browser versions and budget are not named.
- [x] Confirm PyTorch with finite/nonzero gradients, weight updates, repeated steps, and exact JSON export round trip. Pin trainer/rendering dependencies; use the shared JavaScript function for all training and query preparation.
- [x] Create `train/`, `scripts/`, `tests/`, and `bench/`, pinned dependencies, and documented commands for data preparation, training with evaluation, and the full test suite. Ignore font binaries, generated data, and checkpoints; retain one reviewed sample sheet.

Exit: a checked-in experiment configuration and benchmark protocol that another contributor can follow without guessing the task or success criteria.

## 1. Build the font manifest and evaluation set

- [x] Select 500 diverse Google Fonts families: sans, serif, slab, monospace, and display, including visually similar families. Begin with 50 for pipeline smoke checks, then use the fixed 500-family experiment. Superseded: all 2,004 eligible families (scope update, 2026-09-22).
- [x] Pin source revision and file checksums; record canonical IDs, aliases, glyph coverage, style/axis metadata, provenance, and each font's license. Google Fonts supplies per-family metadata and license files; inspect those files rather than assuming one license for the collection. [Source](https://github.com/google/fonts/blob/main/README.md) Done: 3,886 pinned binaries with their metadata and licence files ([corpus](bench/corpus.md)).
- [x] Check for duplicate files, renamed families, and shared outlines before splitting. Keep aliases and closely related versions together to avoid claiming memorized designs as unseen fonts. Done: lineage and alias groups before the split ([split](bench/encoder-split.json)); identical designs as twins ([style](bench/style.md#ground-truth-for-style)).
- [x] Assign approximately 350 families to encoder training, 50 to unseen-family development, and 100 to final unseen-family evaluation. Preserve related-family groups even if exact counts change. Reserve another 50 families outside the catalog, split equally between rejection development and final testing. Superseded: 1,403 / 301 / 300 families ([split](bench/encoder-split.json)); absent-family sets belong to rejection, below.
- [x] Separate training, prototype-generation, validation, and test text pools and rendering seeds. Keep derivatives of the same source screenshot in one split. Use a fixed development catalog; introduce the final 100 families only after choosing the encoder. Done: training text never repeats and the frozen benchmark has its own text seed.
- [x] Create separate browser and independently sourced validation crops for training families and the 50 development families. Use these for model/prototype selection and rejection calibration; keep the final sets below locked. Done: the development part of the photo sets ([photos](bench/photos.md#development-and-final)).
- [ ] Capture at least 10 test crops per indexed family through a browser pipeline independent of the training renderer. Vary text, size, pixel ratio, light/dark background, and scaling. Verify that the intended font loaded and no glyph fallback occurred. Remaining: Chromium renders both training views and test crops; other browsers are unmeasured.
- [ ] Collect at least 200 additional screenshots from independently produced pages or design files with verified source fonts, including at least 50 final unseen families. Keep these separate from generated fixtures; record font-file identity where possible and exclude uncertain labels from exact-match metrics. Remaining: photos and forum crops exist ([photos](bench/photos.md)); screenshots of pages and design files are not collected.
- [ ] Add absent-family, near-clone, blank, non-text, extremely short, blurred, and out-of-scope fixtures. Record ambiguity instead of inventing certainty for indistinguishable samples. Remaining: blank and low-contrast crops are tested and twins are credited; the absent-family set (1,420 forum requests no catalog holds) is unused.
- [ ] Implement a report with macro family top-1/top-5, sample-weighted scores, trained/unseen splits, independent screenshots, rejection metrics, and a confusion list. Include family-level uncertainty intervals and the complete candidate catalog size. Remaining: rejection metrics and confidence intervals; the rest is in [style](bench/style.md) and [photos](bench/photos.md).

Exit: a versioned manifest and reproducible evaluation command. No training sample, prototype image, or tuning decision uses final evaluation crops.

## 2. Establish preprocessing and input robustness

- [x] Implement a pilot preprocessing specification: alpha compositing, grayscale conversion, foreground polarity, conservative margin trimming, contrast normalization, aspect-preserving resize, and padding to 128 × 32 pixels. Preserve stroke proportions and antialiasing. Window policy still needs comparison before freezing the release contract.
- [x] Inspect normalized samples across font classes. Verify that wide crops retain useful glyph detail; test bounded overlapping windows if a single padded resize loses it. Account for every window in latency measurements. Done: up to three windows a crop ([preparation](bench/preparation.md)), all counted in the timings.
- [x] Specify and test invalid dimensions/buffers/crops, blank input, low contrast, alpha, inversion, detached dots, fractional shrink, subarrays, and independent repeated outputs. Blank/low-contrast tensors have an explicit non-match status.
- [x] Build and measure a non-neural edge/projection descriptor baseline on the 20-family catalog, with separate prototype/query text and grayscale/binary ablation. Archived comparison; no further descriptor development.
- [x] Implement cosine search, deterministic tie-breaking, and family deduplication in Python. Record baseline validation results and time before training. Superseded: benchmarks rank in Python (`train.style final`) and the page in JavaScript (`rankCatalog`), one row per family.
- [x] Save preprocessing fixtures with expected tensors so the later JavaScript implementation can be checked against the offline pipeline. Superseded: Python prepares through the browser's own `prepareLine`, byte for byte, so there is one preprocessing.

Exit: images become reproducible tensors, evaluation runs end to end, and there is a baseline to beat.

## 3. Train and test the tiny embedding

- [x] Run the [one-font verification experiment](bench/robustness.md): matched positive/negative crops, disjoint text pools, nine distortion conditions, separate threshold calibration, and an input-resolution ablation. The gate is not met; prioritize new-text generalization.
- [x] Expand text diversity and run a matched-budget whole-line/local-window comparison with independent browser rendering. [Local windows preserve useful detail](bench/focus.md), but generalization and rotation still fail the gate. Keep the experimental binary models separate from the demo.


- [x] Turn reported crop failures into a fixed evaluation: save source images and verified family labels, separate full-line, 6/12/24-character excerpts, border padding, and renderer conditions. Report recall and false-positive rates for one-target verification; measure top-1/top-5 neural retrieval after expanding the training task. Keep these development cases out of training. Done: the finder set's fixed crops with verified labels ([photos](bench/photos.md)); the reported demo samples rank first.
- [x] Add randomized crop length/position and border padding to training fixtures, preserving the disjoint training text pool. Compare against the frozen current model at the same training budget; separately test height-preserving windows so model gains are distinguishable from preprocessing gains. Done in the style views: 1–24 characters, clipped edges ([training views](bench/style.md#training-views)).
- [x] Generate balanced synthetic batches on demand. Start with roughly 1,000 renderings per training family and increase only when learning curves justify it. Store seeds/configuration rather than a huge committed image corpus. Done: every view comes from a face, a script and a seed; nothing is stored.
- [x] Vary text, size, spacing, subpixel placement, antialiasing, pixel ratio, contrast, light/dark background, modest blur, compression, and screenshot scaling. Keep severe perspective, decorative backgrounds, synthetic styles, and broad variable-axis sampling for later scope expansion. Done: the style views vary each of these ([training views](bench/style.md#training-views)).
- [x] Run the first small CNN: 25,120 parameters, normalized 32-dimensional output, convolution/ReLU/pooling/projection, 400 supervised-contrastive steps on 14 families. [Pilot results](bench/training.json) are below the useful-retrieval gate overall; this is not the final trained model.
- [x] Build provisional prototypes from the separate prototype text pool and evaluate against all 400 development-catalog families. Average normalized specimen embeddings and normalize the result; report the 350 training families and 50 unseen development families separately. Superseded: per-face references, searched across all 2,004 families.
- [x] Start with supervised contrastive learning: multiple strings per family in a batch, same-family positives with different text, and different-family negatives including the same text. Introduce hard negatives from validation confusions; avoid false negatives from aliases or known identical outlines. Superseded: the style objective, identity plus letter-distance geometry, with nearest designs in each batch ([objective](bench/style.md#objective)).
- [ ] Monitor same-family/different-text retrieval, different-family/same-text separation, embedding collapse, and train-to-validation gaps. Benchmark 32 vs. 64 dimensions only if the error analysis warrants the extra capacity.
- [x] Compare the tiny model against the declared accuracy gates. If it fails, run one larger reference model to distinguish insufficient model capacity from faulty data or preprocessing. Try distillation only if that larger model demonstrates a useful advantage. Done: 350K, 1.36M and 3.04M encoders compared ([results](bench/style.md#results)).
- [ ] Record model configuration, seeds, parameters, operation count, wall time, checkpoint hash, and validation metrics for every experiment. Select the model using development data only.

Exit: development results meet gate A's retrieval targets on the development catalog with separately measured crop robustness. Phase 4 runs the final gate checks. If both learned models fail, fix the data or narrow the input scope. If only the larger model works, revise the size target explicitly before committing to a custom runtime.

Font recognition research already identifies the synthetic-to-real domain gap as a central problem; generated labels alone do not establish screenshot accuracy. [DeepFont](https://arxiv.org/abs/1507.03196)

## 4. Prove catalog extension and rejection

- [x] Finalize prototype generation with the frozen encoder and a fixed, versioned set of diverse strings and rendering conditions. Average normalized embeddings, then normalize again. Choose one prototype or a small bounded number using development data, then freeze the recipe before introducing final withheld families. Done: each case and script drawn at 16, 24 and 48 px, averaged per face ([references](bench/style.md#third-step-references-at-three-sizes)).
- [x] Establish how families with multiple prototypes are scored without letting prototype count unfairly favor a family. Return each family once. Done: a family scores by its best face and appears once (`rankCatalog`).
- [x] Use the same offline pipeline to encode the 100 final withheld families. Append their entries and retain before/after catalog snapshots; confirm the model file hash did not change. Done: 1,704 → 2,004 families with unchanged encoder bytes ([test](bench/encoder-test.json)).
- [ ] Fit a simple rejection rule on validation data using nearest similarity and, if useful, the first/second-family margin. Validate separately on absent families and near-identical indexed families. Freeze thresholds before final testing.
- [ ] Define `score` as similarity, not a probability. Use a boolean `unknown` for “no reliable catalog identification”; keep ranked suggestions available when useful. Do not expose a numeric probability of unknown without a separate calibration experiment. Remaining: `unknown`; `score` is cosine similarity.
- [ ] Record rejection thresholds with their encoder and catalog version. Rerun calibration and evaluation when the catalog grows; adding fonts need not retrain the encoder, but it changes the search and rejection problem.
- [x] Evaluate all 500 families together on the locked final sets. Report old-family and newly added-family results, including old-family recall before/after adding the new entries. Publish gate A/B/C results, failure examples, and the supported input boundary in `bench/report.md`. Superseded: all 2,004 families on the frozen benchmark, gates stated in [limits](bench/style.md#limits-and-negative-results); gate C waits on rejection.

Exit: gates A–C pass. If unseen-family retrieval fails, the promise of adding fonts without retraining is unproven: improve the encoder or document a narrower supported domain before proceeding.

Learning an embedding that supports retrieval is an established pattern; applying it to a tiny, text-invariant font encoder remains this project's experiment. [FaceNet](https://arxiv.org/abs/1503.03832)

## 5. Export the single-file data artifact

- [ ] Specify a versioned JSON schema containing preprocessing configuration, a supported architecture ID, tensor shapes and weights, quantization parameters, normalized catalog prototypes, font metadata, and rejection settings. Remaining: rejection settings; the rest is the model and catalog formats ([contract](bench/encoder.md)).
- [x] Include separate schema, encoder, and catalog versions plus checksums. Reject incompatible dimensions, non-finite values, malformed tensor sizes, duplicate IDs, and prototypes generated by a different encoder. Done: `readCatalog` checks version, dimensions, sizes, finite values, unique ids and the model's hashes; site.json records checksums.
- [x] Compare float weights with int8 weight quantization and float prototypes with int8 prototypes. Dequantize consistently and renormalize vectors for cosine search. Start with weight-only quantization; add more complex quantization only if needed. Done: [quantization](bench/style.md#quantization).
- [x] Measure exported accuracy against the frozen float model. Target ≤1 percentage point top-5 loss, and require the rejection gate to hold after validation-only recalibration. Done: 0.24 points below float ([quantization](bench/style.md#quantization)); rejection does not exist yet.
- [x] Compare ordinary numeric arrays with packed bytes encoded inside JSON. Choose using actual raw, gzip, and Brotli size plus parse/load time. Include metadata, scales, and encoding overhead in every size claim. Done: 6-bit weights and 4-bit catalog rows, packed in base64 ([quantization](bench/style.md#quantization)); the catalog parses in 44 ms.
- [x] Export one default `gpu-font.json` with no required weight sidecar. Keep a readable canonical source manifest and deterministic build command so contributions remain reviewable. Superseded: the model and each catalog are separate files, so catalogs swap under one model.
- [x] Measure the 500-family artifact against gate D's total package budget. Record a separate stretch target of ≤150 KiB Brotli for runtime + model + 2,000 families; account for multiple prototypes if used. Measured: 10,675,579 bytes, 5.35 MB with Brotli, over the 10 MB ceiling ([catalog](bench/style.md#catalog)); far from the 150 KiB stretch.
- [x] Treat a new encoder as requiring catalog regeneration unless compatibility is demonstrated. Publish the regeneration path; never silently mix old and new vector spaces. Done: a catalog carries its model's hashes and any other model refuses it; README and development.md give the rule and the rebuild path.

Exit: reproducible JSON export, validated round trip, measured size, and acceptable quantization loss.

## 6. Implement the browser library

- [x] Implement CPU and float32 WGSL inference for the deployed int8 ten-way classifier, with variable-size text windows and probability aggregation. Verify independent PyTorch reference inputs, logits/ranking parity, A → A → B → A reuse, concurrent requests, input snapshots, invalid-input recovery, unavailable GPU, and device-loss fallback. See [demo checks](checks/demo.mjs) and [measurements](bench/demo.json). The release API and broader hardware checks below remain open.
- [x] Implement a small JavaScript CPU reference using typed arrays and the exported data. Match preprocessing tensors, layer outputs, final vectors, rankings, and rejection decisions against Python fixtures. Done: `inferCPU` matches PyTorch within 3.8e-6 (checks/catalog-demo.mjs).
- [ ] Implement only the chosen model's operations in WGSL. Use float32 compute first; unpack/dequantize weights at initialization if that keeps the runtime simpler. Measure the resulting GPU memory separately from download size. Remaining: GPU memory; `src/network-gpu.mjs` runs the encoder and matches PyTorch within 3.8e-6 on Metal.
- [x] Initialize lazily, cache pipelines and weights, reuse buffers, bound crop/window counts, and release owned resources. Test repeated and concurrent requests and GPU device loss. Done: one to three windows a crop, one device reused until `destroy()`, a lost device falls back to the CPU (`src/match.mjs`).
- [x] Keep nearest-prototype search on the CPU initially. Benchmark full image-to-result time, including preprocessing, uploads, dispatch, readback, and search, before moving more work to the GPU. Done: exact cosine search in JavaScript; a whole match takes about 0.1 s on WebGPU ([runtime](bench/style-runtime.json)).
- [x] Handle unavailable adapters, initialization failures, and device loss with the tested CPU path. The WebGPU API includes fallible adapter acquisition and device-loss handling; exercise those paths explicitly. [WebGPU explainer](https://gpuweb.github.io/gpuweb/explainer/) Done: without WebGPU the CPU path runs (checks/catalog-demo.mjs).
- [x] Settle and document the minimal public API below. Accept `ImageData` first; add browser image/canvas inputs through one shared conversion path. Validate `top` and empty catalogs, and keep model initialization reusable. Done: `createMatcher()` and `match(imageData)` (README, `src/match.d.ts`); an empty catalog is refused.
- [x] Allow an explicit local JSON catalog/model to replace the default data without code changes. Check compatibility before merging additional catalog entries. Done: `createMatcher(catalog, model)` and the page's Open catalog JSON.
- [x] Test a clean ESM consumer with type declarations and no browser ML framework dependency. Keep import-time behavior safe outside the browser; image inference remains the documented browser feature. Done: tests/package.test.mjs checks the tarball's modules and types; tests/match.test.mjs runs it in Node, with no dependencies.
- [ ] Run hardware browser checks and gate D benchmarks on the named laptop and phone. Report CPU and GPU cold/warm p50/p95 plus peak memory. Choose the default backend from evidence if GPU overhead dominates small requests. Remaining: only the M4 Max laptop is measured; no phone and no peak memory.

Proposed API, to finalize after the model/data contract is proven:

```js
import font, { fontvec } from 'gpu-font'

const result = await font(image, { top: 5 })
// { matches: [{ id, family, score }], unknown: boolean }

const vector = await fontvec(image)
// L2-normalized Float32Array; length declared by the model version
```

Exit: gates A–D pass through the shipped browser path, with Python/CPU/GPU parity tests and a working fallback. Any hardware-dependent limits are documented with measured evidence.

## 7. Make v0.1 reproducible and usable

- [x] Build the local neural demo: Open, keyboard paste and drop, drag/two-corner/keyboard cropping, ranked font previews, a searchable sample popover beside Image, exact model-input preview, result export, replace/clear, and Google Fonts links. Descriptor selection, the Paste button, numeric crop fields and redundant instructions are removed.
- [x] Match automatically on import and while cropping. Keep one active inference plus only the newest pending crop, read only selected pixels, discard stale results, and infer transparency backing. Remove the background selector, sample metadata, and search button from the interface.
- [x] Check the pilot interaction lifecycle in Chromium: loading, crop changes, stale requests, clear/reset, malformed/blank images, unsupported preview glyphs, unavailable GPU/device loss, dialog keyboard focus, and 320/375/414/768/1440 px layouts. Physical mobile hardware remains untested.
- [ ] Add the calibrated uncertain/unknown result state after phase 4 supplies validated rejection thresholds.
- [ ] Add `scripts/add-font` to read a permitted local font, validate metadata/glyph coverage, render canonical specimens, run the frozen encoder, and emit a compatible catalog entry without invoking training. Same as the one-command index above; the browser path exists.
- [ ] Write a contribution checklist covering source/license evidence, stable IDs, aliases, encoder version, reproducibility, and evaluation impact. Support private local catalogs for fonts that cannot enter the public artifact.
- [ ] Add CI for preprocessing, numerical parity, catalog validation, deterministic export, malformed input, ranking/rejection, CPU fallback, clean package consumption, and compressed-size budgets. Run GPU tests on actual supported hardware; label unavailable coverage explicitly. Remaining: no CI workflow exists.
- [ ] Document installation, one working example, supported input conditions, score/unknown semantics, coverage, offline use, measured size/latency, known failures, and how to add fonts. Publish data/model license and attribution information alongside the package. Remaining: unknown semantics, once rejection exists, and offline use; the rest is in the README.
- [x] Provide commands to regenerate the manifest, train, evaluate, export, and run the complete test suite. Keep large artifacts out of Git and publish their checksums and retrieval instructions. Done: [development](development.md) and each report's Reproduce section.
- [ ] Run the full test suite and final browser checks against the packaged release artifact. Have a fresh checkout reproduce a small end-to-end example, then prepare the v0.1 release with its benchmark report. Remaining: 0.1.0 is on npm, but a fresh checkout fails tests/preview-sources-v2.test.mjs: its scripts/preview-sources.mjs is excluded in .git/info/exclude.

Exit: gate E passes. A developer can identify a supported crop and add a permitted new family using the documented commands.

## 8. Expand toward the original coverage goals

Each expansion gets its own coverage manifest, independent evaluation samples, rejection calibration, size/latency report, and known limits. Growing the index alone does not prove growing recognition quality.

- [ ] **Google Fonts:** expand from 500 to 2,000 families, then the remaining eligible catalog. Track families versus faces separately; automate upstream change detection and prototype regeneration for changed files. Remaining: upstream change detection; all 2,004 eligible families are indexed.
- [x] **Weights and variable fonts:** test regular/bold/italic and representative axis samples; decide whether multiple prototypes solve family retrieval before adding exact-style prediction. Done: every static face and each hundred along weight axes; each match names its face.
- [ ] **Adobe Fonts and MyFonts:** inventory sources and verify permitted font access, training use, derived-artifact distribution, and specimen use for each relevant agreement. A subscription is not evidence of permission for every operation; Adobe directs some uses to separate licenses. Obtain the required permissions before including a source. [Adobe licensing FAQ](https://helpx.adobe.com/fonts/web/font-licensing/font-licensing.html) Inventoried with their terms on the Catalogs page; no permission yet.
- [x] **Independent foundries and custom fonts:** provide the same contribution/import workflow with source provenance and reproducible checks. Distribute compatible catalog packs only where permitted; keep restricted font files local. Done: Fontshare, Collletttivo, Velvetyne and more in Other, only where terms allow; My fonts stay in the browser and export as JSON.
- [ ] **Historical fonts:** first cover documented digital revivals. For scans without a source font, create a separate specimen-retrieval benchmark with provenance and uncertain labels; distinguish original type, revival, and closest available match. Test print wear, ink spread, paper texture, and scanning before claiming support.
- [x] **Other scripts:** add script-specific glyph coverage checks, text corpora, shaping, fixtures, and held-out families. Revalidate whether the current encoder generalizes; retrain and regenerate catalogs if a new script requires it. Done: 22 scripts in the style target and training, with per-script heads and results ([by requirement](bench/style.md#by-requirement)); accuracy is tracked above.
- [ ] **Large catalogs:** measure retrieval as distractor families increase. Add optional source/script packs while preserving a one-JSON default. Keep exact cosine search until measurements justify an approximate index. Remaining: retrieval against more distractors was measured once, 1,704 → 2,004, on an earlier encoder; catalogs split by source and All searches them together with exact cosine.
- [ ] **Known-text assistance:** experiment with `{ text }` by rendering the supplied text in a small candidate shortlist for reranking. Measure gains and extra font-asset cost; the text option must not imply that the existing encoder consumes characters it was never trained to use.
- [ ] **Secondary uses:** validate screenshot similarity, clustering, and font-image search on separate tasks before documenting them as supported. Add descriptive axes only if a labeled evaluation shows they are useful.

Completion is measured coverage per release, not an assertion of universal recognition.

## Start here

- [x] Build the pinned full-corpus data pipeline and wider browser-compatible encoder.
- [x] Finish and evaluate the corpus candidate with the commands in [bench/corpus.md](bench/corpus.md). Done: 28.4% / 48.7% browser top-1/top-5, then superseded by the style encoder.
- [x] Use the measured failures to prioritize genuine face/axis coverage, hard confusable families and short/partial glyph crops. Done: the style encoder added every face, twins for confusable families and 1–24 character views.
- [ ] Resolve reliability and unknown-font rejection on independent screenshots before a release claim or further commercial-source expansion.
- [ ] Soften the script filter: the page skips faces that cannot draw the detected script, and on rare scripts the detection is wrong (Canadian syllabics, Myanmar, N'Ko, Lao, Syriac: 3–16% detected), so the right family is skipped; other scripts read 82.9% top five filtered against 91.3% before the filter. Filter on the scripts whose head probability adds up to most of the mass, or rank with a penalty instead of skipping.
