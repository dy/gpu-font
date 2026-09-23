# First training pilot — 2026-09-21

Historical pilot report. Current training focus and the neural-only demo update are documented in [one-font verification](robustness.md).

The executable pipeline works. The first tiny encoder does **not** beat the untrained descriptor overall. This historical comparison prompted the subsequent neural robustness experiments. The current demo exposes neural inference only; this pilot report records the earlier comparison.

## Data and scope

- 20 local fonts from `fontr`, pinned by source/instance SHA-256 in [fonts.json](fonts.json); each has an embedded OFL notice. Upstream revisions and standalone family license files remain to be recovered before distributing a corpus.
- Explicit upright weight 400, width 100 where available, other variable axes at recorded defaults. Several source files otherwise default to weights 100–300.
- 14 training families; six unseen-development families: Nunito, Nunito Sans, Lora, Merriweather, Bebas Neue, Playfair Display. Same family always has the same split. The similarly named IBM Plex and Source families stay entirely in training.
- 2,240 Pillow training crops: 24 training strings, sizes 12–48, DPR 1/2, both polarities, moderate blur and JPEG compression. These are finite stored samples, not an unlimited augmentation stream.
- 160 Pillow prototype crops: one separate string × eight rendering conditions × 20 fonts. Mean normalized vectors give one prototype per family.
- 320 Pillow queries and 320 Chromium Canvas queries: two further strings, sizes 14/28, DPR 1/2, both polarities. No independent real screenshots yet.
- Shared JS preparation → 128 × 32 grayscale float32 tensors, aspect ratio preserved. The contact sheet (`scripts/fixtures.py`, kept locally) was visually inspected across all 20 families and both renderers. Dots, counters, descenders, and serif/sans distinctions remain visible; fine serif detail is softened. Height-preserving windows remain untested.

The [configuration](pilot.json) was fixed before training: one seed (1729), 400 steps, batches of four samples per training family, Adam at 0.002, four CPU threads. The encoder has 25,120 parameters, four strided 3 × 3 convolutions with ReLU, global average pooling, a 32-dimensional projection, and L2 normalization. Supervised contrastive learning supplies the objective.

## Measured retrieval

Macro-average family recall; all comparisons search the same **20-family catalog**. Each train-family group has 224 queries (14 × 16), each unseen-family group has 96 (6 × 16). Uniform chance is 5% top-1 and 25% top-5. These small, selected development groups do not establish population accuracy or release gates.

| Query renderer / family split | Descriptor top-1 | Descriptor top-5 | CNN top-1 | CNN top-5 |
| --- | ---: | ---: | ---: | ---: |
| Pillow / training | 26.8% | 72.3% | 18.8% | 67.0% |
| Pillow / unseen development | 31.3% | 83.3% | 20.8% | 54.2% |
| Chromium / training | 21.4% | 61.6% | 13.4% | 33.5% |
| Chromium / unseen development | 16.7% | 57.3% | 35.4% | 65.6% |

Across all 20 families, Chromium top-5 is **60.3% for the descriptor and 43.1% for the CNN**. The improvement on six unseen families cannot compensate for the regression on training families. Loss decreased from 3.847 (first 20 steps) to 2.637 (last 20), which proves optimization progress, not useful retrieval.

Hard thresholding at 0.5 reduced descriptor top-5: Pillow training/unseen 55.4%/60.4%; Chromium training/unseen 39.7%/44.8%. Grayscale is the supported pilot input. This is one descriptor ablation, not proof that every model must prefer grayscale.

Full per-family metrics and common descriptor confusions are in [baseline.json](baseline.json); training metadata and learned per-family metrics are in [training.json](training.json). The training report includes dataset/model hashes.

## Cost and verification

Observed on this arm64 Mac, macOS 26.5, Node 25.9.0, Python 3.14.6, PyTorch 2.11.0, Pillow 12.2.0, Playwright 1.62.1 / Chromium 151.0.7922.34:

- Training: 9.7 seconds, CPU, four threads. No cloud run.
- JS preparation: median 0.134 ms, p95 0.493 ms across 3,040 crops, excluding decoding and file I/O. This is not browser end-to-end latency.
- Exported JSON weights and 20 prototypes: 538,252 bytes; 211,134 bytes with Node's default Brotli settings. Runtime excluded, no quantization. The proposed 100 KiB shipped budget has not been met.
- JSON serialize/load → same PyTorch architecture: maximum embedding difference 0 on eight probe crops. This verifies tensor export, not cross-language inference parity.
- Full suite: 23 Node tests and eight Python tests pass. Browser fixture generation and the complete render/prepare/baseline/train path also ran. The review additionally reran the baseline and sample preview against the frozen pilot data; metrics were unchanged.

Relevant regression evidence:

| Claim | Direct test |
| --- | --- |
| Empty/invalid crops cannot become matches | `single-pixel blank and transparent crops take the explicit non-match path`; `rejects empty/null, mismatched buffers, malformed dimensions and out-of-bounds crops` in [prepare.test.mjs](../tests/prepare.test.mjs) |
| Detached marks and uniform scale survive | `preserves a detached dot, reports source coordinates, and uses uniform scale`: a dot/gap/stem raster, checking output rows, source bounds, scale, and finite [0,1] values |
| Shrinking retains thin-edge coverage | `area shrinking keeps thin strokes at fractional crop boundaries`: a one-pixel line in a 101 × 3 raster → 16 × 8; no output row center lands inside the crop, but footprint overlap must preserve total ink = source area × scale² |
| Alpha, inversion, subarray offsets, retained outputs | Four dedicated tests in `prepare.test.mjs`; the last runs A → A → B → blank → A and mutates the final output while retaining the first |
| Training has useful gradients and export preserves output | `test_gradient_update_and_json_round_trip`: three optimizer steps, finite/nonzero gradients, changed weights, unit embeddings, exact output after JSON round trip, batch size one |
| Bad export/data cannot silently enter training | `test_export_rejects_wrong_architecture_names_shape_count_and_nan`; `test_family_and_text_leakage_and_missing_catalog_are_rejected` in [test_training.py](../tests/test_training.py) |
| Evaluation is actually macro-family retrieval | `test_retrieval_uses_prototypes_and_macro_averages_families`: uneven family counts with deliberately wrong queries; expected top-1 exactly 0.5 |
| Readers reject stale data and malformed tensor boundaries | `baseline validates freshness, exact tensor boundaries, and catalog-dependent chance` in [pipeline.test.mjs](../tests/pipeline.test.mjs): 3 × 3 one-pixel prototype/query, one-family chance = 1, each source A → stale B → A, tensor one byte short/long, checksum mismatch, empty/missing roles; `test_preview_rejects_each_stale_source_and_corrupt_tensor` checks preview freshness; `test_python_readers_reject_truncated_and_trailing_tensor_bytes` checks trainer and preview with 0/11/13/16 bytes where 12 are required, even with matching hashes |
| Font configuration changes cannot silently reuse an old catalog | `preparation rejects a changed font selection before reading raw fixtures`: changed family count/split/file fail; restored configuration prepares a one-pixel crop to 4 × 4 with exact expected ink area; an empty source list rejects |

All prepared-data readers check configuration, font-manifest, normalizer, and tensor hashes. Changing any source requires preparing data again. Preparation also checks that the imported fonts match the configured family selection. Chance recall derives from the actual catalog size. The contrastive loss explicitly rejects zero- and one-sample batches instead of producing NaN.

## Browser demo verification

The [browser checks](../checks/demo.mjs) run the real WGSL encoder in Chromium 151.0.7922.34 on a Metal GPU adapter (`metal-3`). Maximum absolute embedding error against independent PyTorch outputs is **2.83 × 10⁻⁷ for JavaScript CPU** and **2.68 × 10⁻⁷ for WebGPU**; all 20 ranking positions agree for white, black, ramp, stripes, Pillow-query, and Chromium-query inputs. These six numerical probes establish implementation parity, not recognition accuracy. [Machine-readable results](demo.json).

Direct regression evidence:

- GPU buffers: concurrent white → white → ramp → white requests retain independent outputs even when the caller mutates the first input after enqueueing. An empty tensor rejects, the next valid request recovers, and use after destruction rejects.
- Model validation: [model.test.mjs](../tests/model.test.mjs) checks incompatible architecture/preparation, malformed tensors/catalogs, non-finite/overflowing weights, null/empty/short/long/out-of-range pixels, and exact synthetic convolution/projection outputs.
- Import lifecycle: PNG upload → inference → JSON download → clear → clipboard paste → inference → immediate clear leaves no stale matches. Keyboard paste and drop are tested separately. A paused paste decode followed by a newer upload or Clear cannot overwrite the newer state. The Paste button and clipboard-read permission path were removed. Corrupt PNG and blank image paths are explicit.
- Sample integrity: unsupported CJK/emoji preview text preserves the previous preview and cannot create a mislabeled known sample; whitespace resets to the default text. The reference exporter verifies printable ASCII coverage in all 20 pinned font files.
- Crop/catalog controls: dragging and reverse-order corner taps produce identical rectangles; arrow-key movement and Shift + arrow resizing stay inside the image, including a one-pixel crop at the bottom-right boundary. Home/Escape restore the full image. Matches update automatically, including before the pointer is released. The sample dropdown lists all 20 families; Clear or an imported image resets its selection. Image actions stay inside the frame; crop coordinates, background selector, sample metadata and search button are absent. The restored input preview is checked pixel by pixel against the exported normalized tensor, including after crop changes and image replacement.
- Live matching: with the full-image crop selected, Home → ArrowLeft → two animation frames performs zero canvas reads and zero GPU readbacks. Hold a real GPU readback, dispatch 50 additional crop changes, then release it. Exactly two inferences run: the active crop and the newest crop. Exported bounds equal the final selection, rankings match CPU inference, and canvas reads contain only those two crop rectangles. Hold another readback → queue a crop → clear → import a different image → release; only the replacement image is matched, with exactly the same normalized pixels and scores as its original run. Exports stay disabled while results are stale.
- Automatic background: 128 × 64 opaque dark/light, transparent dark/light, uniform alpha-128 dark/light, and transparent grayscale-127/128 versions of the same stroke mask produce equal normalized tensors (within 10⁻⁶) and identical rankings. White transparent ink selects black; dark transparent ink selects white. Grayscale 127 selects white, while 128 selects black, pinning the 127.5 decision boundary. A fully transparent image produces no matches. Opacity is applied once to the completed mask, avoiding darker overlaps. These cases test the plain-text heuristic, not arbitrary artwork.
- Gesture ownership: in `checks/demo.mjs`, a mouse drag from 10% to 40% of a 530 × 110 image keeps the exact rectangle `{x:53, y:11, width:159, height:33}` through unrelated touch down/move/up/cancel events; the original mouse then completes the crop to 90%. Both `pointercancel` and actual capture release stop further crop updates, followed by a fresh one-pixel first corner and a correct second corner. The unrelated-pointer assertion failed before the fix.
- Fallback: removing `navigator.gpu` produces CPU matches. Destroying the actual GPU device before rapid matching requests recovers to CPU matches. Clearing an uploaded image before model initialization finishes preserves the empty state; retaining the upload starts matching automatically after initialization.
- Responsive UI: screenshots inspected at 320, 375, 414, 768, and 1440 px; controls have no horizontal clipping. Five text/background combinations exceed 4.5:1 contrast. Mobile viewports were emulated on this Mac; physical phones and other browsers remain untested.

The page reports preparation + inference + ranking time, excluding image decoding, font loading, and cold model initialization. No latency or package-size release gate is claimed. Browser automation enables Chromium's `--enable-unsafe-webgpu` flag; ordinary demo usage does not set browser flags.

## Next experiment

The user reports poor matches after cropping in the demo. The fixture generator currently renders complete lines with blur/JPEG variation; it has no crop-length, crop-position, or padding augmentation. This is a coverage gap, not yet a measured explanation of the failures. First freeze verified source images and a crop matrix (full line, 6/12/24-character excerpts, tight/loose borders, renderer) and score the existing CNN and descriptor. Keep the cases for development evaluation; train augmentations only from the separate training text pool. Compare crop augmentation and height-preserving windows as separate experiments before replacing the demo weights.

1. **Diagnose the failed retrieval before more training.** Freeze these artifacts. Compare same-text vs different-text retrieval, per-size performance, and renderer transfer. Inspect the nearest wrong matches. This distinguishes text/layout dependence from lost detail and rendering dependence.
2. **Test input detail with one controlled ablation.** Compare current line fitting against height-preserving overlapping windows, using the same font split and queries. Include all windows in cost measurements. Do not add character OCR or skeleton reconstruction as a prerequisite.
3. **Make the 50-family smoke set trustworthy.** Recover pinned upstream font/license sources, add family/outline deduplication, preserve related designs in one split, increase independent training/prototype/query texts, and acquire at least 20 source-verified real development crops. Report these separately from Canvas fixtures.
4. **Run the next bounded encoder experiment.** Balance batches for same-text negatives and different-text positives, cover the diagnosed rendering variation, and compare with the descriptor. If the tiny model still fails, run one larger reference to separate capacity limits from dataset/preparation defects. Preserve failures; choose on development metrics only.

Family recognition remains the first task. `fontr`'s 12-axis style representation may later inform weight/width/slant supervision, but its inspected trainer fits styles and glyph topology; it does not establish a ready image-to-font-parameter predictor. Arbitrary-page detection and semantic parameter estimation each need their own labeled evaluation.
