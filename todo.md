# gpu-font — implementation plan

Build a tiny JavaScript library that turns a cropped image of text into a reusable typography vector and ranked font matches. Inference runs locally in the browser. New fonts enter a versioned data catalog without changing the runtime or retraining the frozen encoder.

The first user is a designer or developer with a screenshot and no access to the original font declaration. The useful result is a short list of plausible families, with an honest indication when the catalog cannot provide a reliable match.

Current state: **100-font neural classification**, with explicit uncertainty and a tested crop UI. The user’s original ten are included. Model selection compares equal-budget continuations and keeps the extra context layer (+2.84 validation points, 89,812 parameters). See [the 100-font report](bench/hundred.md).

Next: [the measured scaling analysis and ordered experiments](bench/scaling.md). The deployment ceiling is now **10 MB for model + catalog + required runtime**, before HTTP compression. The previous 100 KiB target is superseded. Full Google Fonts coverage remains the destination; first isolate lost glyph evidence and train for content-independent style at the current scale.

- [x] Restore model scores beside all five matches and reproduce the Geist Mono/Bebas example with exact exported probabilities. Scores are not visual-similarity measurements.
- [x] Account for 10,000-class storage: a fixed current encoder plus expanded int8 head needs about 0.805 MB of binary tensors, including scales/biases. This is a size projection, not trained accuracy.
- [ ] Build the fresh 6,400-image preparation development bank specified in [the scaling plan](bench/scaling.md#ordered-experiments-and-decision-points), including current user-reported failures and unseen text/size combinations.
- [ ] Compare current windows, oracle complete-glyph groups, automatic groups, and automatic groups + deskew with frozen weights; measure retained evidence, recognition and latency.
- [ ] Run the matched 20k-update preparation/control comparison only after that diagnostic; train and display the same versioned input arrays.
- [ ] Compare compact learned style retrieval with the classifier baseline. Test adding withheld fonts without retraining; measure exact identity and visual similarity separately. Test a training-only glyph head before considering deployed OCR.
- [ ] Advance to 500 families and then a pinned Google Fonts corpus with explicit family/face/script coverage, repeated validation, catalog-specific calibration and a measured download below 10 MB.

- [x] Import 100 pinned regular families and reserve 13 unknown families across validation/test.
- [x] Train on single glyphs, random short strings, words and phrases in two renderers, with edge cuts, scale changes, rotation, shadows and margins.
- [x] Restore the searchable specimen grid; add icon-based opening, clickable empty state, movable/resizable crop handles and source-resolution control.
- [x] Verify actual input preview pixels, plain-margin invariance, crop/resize/cancel interaction, and resolution round trips.
- [x] Complete the bounded 100-font model comparison and separate 21,600-image final test; publish per-length and per-condition results.
- [x] Calibrate uncertainty on validation only and report acceptance coverage. Final accepted accuracy is 99.80%, but coverage is only 9.40%; the reliability gate remains unmet.
- [ ] Inspect rotated/downscaled windows and compare inexpensive text-line deskew/height estimation on fresh validation data before adding OCR or another larger network.
- [ ] Broaden development examples for single letters, near-clone families and partial neighboring letters. Single-glyph identifiability is not guaranteed.
- [ ] Collect verified independent screenshots for the included families, split by source, and reserve a new final test before further tuning. Existing ten-font and 100-font tests are historical.
- [ ] Add weight/italic variation only with corresponding held-out evidence. Do not expand beyond 100 until current failures improve.

Historical descriptor scores are archived, not an active product path. Earlier evidence and the reason for local windows remain in [bench/focus.md](bench/focus.md).

## Goals and limits

- [ ] Recognize a measured, published subset of Google Fonts first, then expand toward the full catalog.
- [ ] Add Adobe Fonts, MyFonts, independent foundries, and custom fonts through source-specific catalog packs and permitted local imports.
- [ ] Extend to historical typefaces, distinguishing identifiable digital revivals from uncertain printed specimens.
- [ ] Deliver a small runtime plus one self-contained JSON data file for the default model and catalog; accept reproducible PRs that add font entries without retraining.

“All fonts” is a coverage direction, not a verifiable release promise. Maintain a coverage manifest with included, unsupported, and access-blocked entries. Identical outlines, missing distinguishing glyphs, and degraded images can prevent exact identification even when the correct font is indexed.

The conversation's 20–50k parameters, 32 dimensions, approximately 100 KB package, and recognition of unseen fonts are hypotheses to test. A 32-dimensional float32 vector takes 128 bytes; 32 bytes requires int8 quantization, excluding metadata. One JSON file still requires executable JavaScript to interpret it.

## Sequence and decision gates

Complete phases 0–4 before committing to a release architecture. Scope update, 2026-09-21: at the user's request, a small CPU/WebGPU demo was brought forward to make the current algorithm inspectable. It does not satisfy the accuracy, size, hardware, or release gates below. Phases 5–7 produce v0.1. Phase 8 expands coverage after that release passes its gates. Dataset collection can continue while experiments run.

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
- [ ] Encode the declared 12–48 CSS px, device pixel ratios 1 and 2, and rendering conditions in a real benchmark manifest; include moderate rescaling/compression, short text, and tiny text as explicit stress cases.
- [x] Define canonical family identity as the primary label; record source/style metadata separately. The [pilot font manifest](bench/fonts.json) now records names, source/instance hashes, embedded licenses, and regular-face axes. Exact face/axis prediction and full alias recovery remain later work.
- [x] Check whether JZ can compile a representative typed-array pixel kernel. The retained [probe](bench/jz-probe.mjs) verifies JS/WASM numerical parity, not speed or font-recognition quality.
- [ ] Build the small [input experiment](bench/protocol.md): 20 known families, controlled renderings, independently captured development crops, a shared grayscale `prepare` function, and neural recognition. Preserve antialiasing; test binary thresholding as an alternative, not an assumption.
- [x] Complete the synthetic portion of that experiment: 20 hash-pinned families, 480 controlled Pillow crops, 320 Chromium queries, 2,240 separate training crops, shared normalization, grayscale/binary descriptor comparison, and an inspected contact sheet. Independent screenshots remain outstanding.
- [ ] Freeze the gates above, split rules, target laptop/phone, supported browser versions to test, and a small experiment budget. Start with at most three architecture sizes and two seeds per size; record hardware time and cost before scaling.
- [x] Confirm PyTorch with finite/nonzero gradients, weight updates, repeated steps, and exact JSON export round trip. Pin trainer/rendering dependencies; use the shared JavaScript function for all training and query preparation.
- [x] Create `train/`, `scripts/`, `tests/`, and `bench/`, pinned dependencies, and documented commands for data preparation, training with evaluation, and the full test suite. Ignore font binaries, generated data, and checkpoints; retain one reviewed sample sheet.

Exit: a checked-in experiment configuration and benchmark protocol that another contributor can follow without guessing the task or success criteria.

## 1. Build the font manifest and evaluation set

- [ ] Select 500 diverse Google Fonts families: sans, serif, slab, monospace, and display, including visually similar families. Begin with 50 for pipeline smoke checks, then use the fixed 500-family experiment.
- [ ] Pin source revision and file checksums; record canonical IDs, aliases, glyph coverage, style/axis metadata, provenance, and each font's license. Google Fonts supplies per-family metadata and license files; inspect those files rather than assuming one license for the collection. [Source](https://github.com/google/fonts/blob/main/README.md)
- [ ] Check for duplicate files, renamed families, and shared outlines before splitting. Keep aliases and closely related versions together to avoid claiming memorized designs as unseen fonts.
- [ ] Assign approximately 350 families to encoder training, 50 to unseen-family development, and 100 to final unseen-family evaluation. Preserve related-family groups even if exact counts change. Reserve another 50 families outside the catalog, split equally between rejection development and final testing.
- [ ] Separate training, prototype-generation, validation, and test text pools and rendering seeds. Keep derivatives of the same source screenshot in one split. Use a fixed development catalog; introduce the final 100 families only after choosing the encoder.
- [ ] Create separate browser and independently sourced validation crops for training families and the 50 development families. Use these for model/prototype selection and rejection calibration; keep the final sets below locked.
- [ ] Capture at least 10 test crops per indexed family through a browser pipeline independent of the training renderer. Vary text, size, pixel ratio, light/dark background, and scaling. Verify that the intended font loaded and no glyph fallback occurred.
- [ ] Collect at least 200 additional screenshots from independently produced pages or design files with verified source fonts, including at least 50 final unseen families. Keep these separate from generated fixtures; record font-file identity where possible and exclude uncertain labels from exact-match metrics.
- [ ] Add absent-family, near-clone, blank, non-text, extremely short, blurred, and out-of-scope fixtures. Record ambiguity instead of inventing certainty for indistinguishable samples.
- [ ] Implement a report with macro family top-1/top-5, sample-weighted scores, trained/unseen splits, independent screenshots, rejection metrics, and a confusion list. Include family-level uncertainty intervals and the complete candidate catalog size.

Exit: a versioned manifest and reproducible evaluation command. No training sample, prototype image, or tuning decision uses final evaluation crops.

## 2. Establish preprocessing and input robustness

- [x] Implement a pilot preprocessing specification: alpha compositing, grayscale conversion, foreground polarity, conservative margin trimming, contrast normalization, aspect-preserving resize, and padding to 128 × 32 pixels. Preserve stroke proportions and antialiasing. Window policy still needs comparison before freezing the release contract.
- [ ] Inspect normalized samples across font classes. Verify that wide crops retain useful glyph detail; test bounded overlapping windows if a single padded resize loses it. Account for every window in latency measurements.
- [x] Specify and test invalid dimensions/buffers/crops, blank input, low contrast, alpha, inversion, detached dots, fractional shrink, subarrays, and independent repeated outputs. Blank/low-contrast tensors have an explicit non-match status.
- [x] Build and measure a non-neural edge/projection descriptor baseline on the 20-family catalog, with separate prototype/query text and grayscale/binary ablation. Archived comparison; no further descriptor development.
- [ ] Implement cosine search, deterministic tie-breaking, and family deduplication in Python. Record baseline validation results and time before training.
- [ ] Save preprocessing fixtures with expected tensors so the later JavaScript implementation can be checked against the offline pipeline.

Exit: images become reproducible tensors, evaluation runs end to end, and there is a baseline to beat.

## 3. Train and test the tiny embedding

- [x] Run the [one-font verification experiment](bench/robustness.md): matched positive/negative crops, disjoint text pools, nine distortion conditions, separate threshold calibration, and an input-resolution ablation. The gate is not met; prioritize new-text generalization.
- [x] Expand text diversity and run a matched-budget whole-line/local-window comparison with independent browser rendering. [Local windows preserve useful detail](bench/focus.md), but generalization and rotation still fail the gate. Keep the experimental binary models separate from the demo.


- [ ] Turn reported crop failures into a fixed evaluation: save source images and verified family labels, separate full-line, 6/12/24-character excerpts, border padding, and renderer conditions. Report recall and false-positive rates for one-target verification; measure top-1/top-5 neural retrieval after expanding the training task. Keep these development cases out of training.
- [ ] Add randomized crop length/position and border padding to training fixtures, preserving the disjoint training text pool. Compare against the frozen current model at the same training budget; separately test height-preserving windows so model gains are distinguishable from preprocessing gains.
- [ ] Generate balanced synthetic batches on demand. Start with roughly 1,000 renderings per training family and increase only when learning curves justify it. Store seeds/configuration rather than a huge committed image corpus.
- [ ] Vary text, size, spacing, subpixel placement, antialiasing, pixel ratio, contrast, light/dark background, modest blur, compression, and screenshot scaling. Keep severe perspective, decorative backgrounds, synthetic styles, and broad variable-axis sampling for later scope expansion.
- [x] Run the first small CNN: 25,120 parameters, normalized 32-dimensional output, convolution/ReLU/pooling/projection, 400 supervised-contrastive steps on 14 families. [Pilot results](bench/training.json) are below the useful-retrieval gate overall; this is not the final trained model.
- [ ] Build provisional prototypes from the separate prototype text pool and evaluate against all 400 development-catalog families. Average normalized specimen embeddings and normalize the result; report the 350 training families and 50 unseen development families separately.
- [ ] Start with supervised contrastive learning: multiple strings per family in a batch, same-family positives with different text, and different-family negatives including the same text. Introduce hard negatives from validation confusions; avoid false negatives from aliases or known identical outlines.
- [ ] Monitor same-family/different-text retrieval, different-family/same-text separation, embedding collapse, and train-to-validation gaps. Benchmark 32 vs. 64 dimensions only if the error analysis warrants the extra capacity.
- [ ] Compare the tiny model against the declared accuracy gates. If it fails, run one larger reference model to distinguish insufficient model capacity from faulty data or preprocessing. Try distillation only if that larger model demonstrates a useful advantage.
- [ ] Record model configuration, seeds, parameters, operation count, wall time, checkpoint hash, and validation metrics for every experiment. Select the model using development data only.

Exit: development results meet gate A's retrieval targets on the development catalog with separately measured crop robustness. Phase 4 runs the final gate checks. If both learned models fail, fix the data or narrow the input scope. If only the larger model works, revise the size target explicitly before committing to a custom runtime.

Font recognition research already identifies the synthetic-to-real domain gap as a central problem; generated labels alone do not establish screenshot accuracy. [DeepFont](https://arxiv.org/abs/1507.03196)

## 4. Prove catalog extension and rejection

- [ ] Finalize prototype generation with the frozen encoder and a fixed, versioned set of diverse strings and rendering conditions. Average normalized embeddings, then normalize again. Choose one prototype or a small bounded number using development data, then freeze the recipe before introducing final withheld families.
- [ ] Establish how families with multiple prototypes are scored without letting prototype count unfairly favor a family. Return each family once.
- [ ] Use the same offline pipeline to encode the 100 final withheld families. Append their entries and retain before/after catalog snapshots; confirm the model file hash did not change.
- [ ] Fit a simple rejection rule on validation data using nearest similarity and, if useful, the first/second-family margin. Validate separately on absent families and near-identical indexed families. Freeze thresholds before final testing.
- [ ] Define `score` as similarity, not a probability. Use a boolean `unknown` for “no reliable catalog identification”; keep ranked suggestions available when useful. Do not expose a numeric probability of unknown without a separate calibration experiment.
- [ ] Record rejection thresholds with their encoder and catalog version. Rerun calibration and evaluation when the catalog grows; adding fonts need not retrain the encoder, but it changes the search and rejection problem.
- [ ] Evaluate all 500 families together on the locked final sets. Report old-family and newly added-family results, including old-family recall before/after adding the new entries. Publish gate A/B/C results, failure examples, and the supported input boundary in `bench/report.md`.

Exit: gates A–C pass. If unseen-family retrieval fails, the promise of adding fonts without retraining is unproven: improve the encoder or document a narrower supported domain before proceeding.

Learning an embedding that supports retrieval is an established pattern; applying it to a tiny, text-invariant font encoder remains this project's experiment. [FaceNet](https://arxiv.org/abs/1503.03832)

## 5. Export the single-file data artifact

- [ ] Specify a versioned JSON schema containing preprocessing configuration, a supported architecture ID, tensor shapes and weights, quantization parameters, normalized catalog prototypes, font metadata, and rejection settings.
- [ ] Include separate schema, encoder, and catalog versions plus checksums. Reject incompatible dimensions, non-finite values, malformed tensor sizes, duplicate IDs, and prototypes generated by a different encoder.
- [ ] Compare float weights with int8 weight quantization and float prototypes with int8 prototypes. Dequantize consistently and renormalize vectors for cosine search. Start with weight-only quantization; add more complex quantization only if needed.
- [ ] Measure exported accuracy against the frozen float model. Target ≤1 percentage point top-5 loss, and require the rejection gate to hold after validation-only recalibration.
- [ ] Compare ordinary numeric arrays with packed bytes encoded inside JSON. Choose using actual raw, gzip, and Brotli size plus parse/load time. Include metadata, scales, and encoding overhead in every size claim.
- [ ] Export one default `gpu-font.json` with no required weight sidecar. Keep a readable canonical source manifest and deterministic build command so contributions remain reviewable.
- [ ] Measure the 500-family artifact against gate D's total package budget. Record a separate stretch target of ≤150 KiB Brotli for runtime + model + 2,000 families; account for multiple prototypes if used.
- [ ] Treat a new encoder as requiring catalog regeneration unless compatibility is demonstrated. Publish the regeneration path; never silently mix old and new vector spaces.

Exit: reproducible JSON export, validated round trip, measured size, and acceptable quantization loss.

## 6. Implement the browser library

- [x] Implement CPU and float32 WGSL inference for the deployed int8 ten-way classifier, with variable-size text windows and probability aggregation. Verify independent PyTorch reference inputs, logits/ranking parity, A → A → B → A reuse, concurrent requests, input snapshots, invalid-input recovery, unavailable GPU, and device-loss fallback. See [demo checks](checks/demo.mjs) and [measurements](bench/demo.json). The release API and broader hardware checks below remain open.
- [ ] Implement a small JavaScript CPU reference using typed arrays and the exported data. Match preprocessing tensors, layer outputs, final vectors, rankings, and rejection decisions against Python fixtures.
- [ ] Implement only the chosen model's operations in WGSL. Use float32 compute first; unpack/dequantize weights at initialization if that keeps the runtime simpler. Measure the resulting GPU memory separately from download size.
- [ ] Initialize lazily, cache pipelines and weights, reuse buffers, bound crop/window counts, and release owned resources. Test repeated and concurrent requests and GPU device loss.
- [ ] Keep nearest-prototype search on the CPU initially. Benchmark full image-to-result time, including preprocessing, uploads, dispatch, readback, and search, before moving more work to the GPU.
- [ ] Handle unavailable adapters, initialization failures, and device loss with the tested CPU path. The WebGPU API includes fallible adapter acquisition and device-loss handling; exercise those paths explicitly. [WebGPU explainer](https://gpuweb.github.io/gpuweb/explainer/)
- [ ] Settle and document the minimal public API below. Accept `ImageData` first; add browser image/canvas inputs through one shared conversion path. Validate `top` and empty catalogs, and keep model initialization reusable.
- [ ] Allow an explicit local JSON catalog/model to replace the default data without code changes. Check compatibility before merging additional catalog entries.
- [ ] Test a clean ESM consumer with type declarations and no browser ML framework dependency. Keep import-time behavior safe outside the browser; image inference remains the documented browser feature.
- [ ] Run hardware browser checks and gate D benchmarks on the named laptop and phone. Report CPU and GPU cold/warm p50/p95 plus peak memory. Choose the default backend from evidence if GPU overhead dominates small requests.

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
- [ ] Add `scripts/add-font` to read a permitted local font, validate metadata/glyph coverage, render canonical specimens, run the frozen encoder, and emit a compatible catalog entry without invoking training.
- [ ] Write a contribution checklist covering source/license evidence, stable IDs, aliases, encoder version, reproducibility, and evaluation impact. Support private local catalogs for fonts that cannot enter the public artifact.
- [ ] Add CI for preprocessing, numerical parity, catalog validation, deterministic export, malformed input, ranking/rejection, CPU fallback, clean package consumption, and compressed-size budgets. Run GPU tests on actual supported hardware; label unavailable coverage explicitly.
- [ ] Document installation, one working example, supported input conditions, score/unknown semantics, coverage, offline use, measured size/latency, known failures, and how to add fonts. Publish data/model license and attribution information alongside the package.
- [ ] Provide commands to regenerate the manifest, train, evaluate, export, and run the complete test suite. Keep large artifacts out of Git and publish their checksums and retrieval instructions.
- [ ] Run the full test suite and final browser checks against the packaged release artifact. Have a fresh checkout reproduce a small end-to-end example, then prepare the v0.1 release with its benchmark report.

Exit: gate E passes. A developer can identify a supported crop and add a permitted new family using the documented commands.

## 8. Expand toward the original coverage goals

Each expansion gets its own coverage manifest, independent evaluation samples, rejection calibration, size/latency report, and known limits. Growing the index alone does not prove growing recognition quality.

- [ ] **Google Fonts:** expand from 500 to 2,000 families, then the remaining eligible catalog. Track families versus faces separately; automate upstream change detection and prototype regeneration for changed files.
- [ ] **Weights and variable fonts:** test regular/bold/italic and representative axis samples; decide whether multiple prototypes solve family retrieval before adding exact-style prediction.
- [ ] **Adobe Fonts and MyFonts:** inventory sources and verify permitted font access, training use, derived-artifact distribution, and specimen use for each relevant agreement. A subscription is not evidence of permission for every operation; Adobe directs some uses to separate licenses. Obtain the required permissions before including a source. [Adobe licensing FAQ](https://helpx.adobe.com/fonts/web/font-licensing/font-licensing.html)
- [ ] **Independent foundries and custom fonts:** provide the same contribution/import workflow with source provenance and reproducible checks. Distribute compatible catalog packs only where permitted; keep restricted font files local.
- [ ] **Historical fonts:** first cover documented digital revivals. For scans without a source font, create a separate specimen-retrieval benchmark with provenance and uncertain labels; distinguish original type, revival, and closest available match. Test print wear, ink spread, paper texture, and scanning before claiming support.
- [ ] **Other scripts:** add script-specific glyph coverage checks, text corpora, shaping, fixtures, and held-out families. Revalidate whether the current encoder generalizes; retrain and regenerate catalogs if a new script requires it.
- [ ] **Large catalogs:** measure retrieval as distractor families increase. Add optional source/script packs while preserving a one-JSON default. Keep exact cosine search until measurements justify an approximate index.
- [ ] **Known-text assistance:** experiment with `{ text }` by rendering the supplied text in a small candidate shortlist for reranking. Measure gains and extra font-asset cost; the text option must not imply that the existing encoder consumes characters it was never trained to use.
- [ ] **Secondary uses:** validate screenshot similarity, clustering, and font-image search on separate tasks before documenting them as supported. Add descriptive axes only if a labeled evaluation shows they are useful.

Completion is measured coverage per release, not an assertion of universal recognition.

## Start here

- [ ] Build the 20-family input experiment in [bench/protocol.md](bench/protocol.md), then expand to a pinned 50-family smoke subset.
- [ ] Produce one verified browser screenshot and one synthetic training crop per family; inspect their normalized tensors.
- [x] Run the descriptor baseline and a tiny-model smoke experiment through the same evaluator.
- [ ] Improve the failed pilot, scale the fixed experiment to 500 families, and resolve gates A–C before release optimization or broad commercial catalogs.
