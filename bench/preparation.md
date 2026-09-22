# Preparation and equal-budget continuation

A fresh **6,400-image development bank** tests 100 regular families, eight strings, two renderers and four views: clean, 45%-scale, ±6° rotation and a partial neighboring `v`. Source sizes 18/32/56 px are balanced across the font/text combinations. Multi-character strings do not occur in the historical corpus; single characters share the alphabet. This is synthetic development evidence, not independent screenshot accuracy.

The original trained encoder stays frozen for the first comparison. Every method exports the exact arrays it evaluates. [preparation.json](preparation.json) records hashes, counts, top-1/top-5 and Node preparation time.

| Preparation, frozen weights | All top-1 | Rotated top-1 | Clean top-1 |
| --- | ---: | ---: | ---: |
| Existing windows | 60.08% | 49.75% | 73.75% |
| Image-derived whole-word groups | 53.14% | 41.13% | 66.94% |
| Whole-word groups + deskew | 56.19% | 53.31% | 66.94% |
| Existing windows + deskew | **62.16%** | **58.06%** | **73.75%** |
| Renderer word boxes + known rotation | 58.52% | 60.12% | 66.94% |

Grouping complete words preserves their boundaries but can shrink the letters much more than the existing local windows. It changes the input distribution and performed worse with frozen weights. This does not show that letter detection is useless; it shows that this whole-word replacement is not ready to deploy. The renderer-box result uses a different sampling policy, so it is not an absolute upper bound for the existing sampler.

[Geometric coverage](preparation-coverage.json) confirms this tradeoff: the original windows retain 93.76% of source ink and contain 66.71% of words wholly in one window. Whole-word groups retain 100% of ink and 99.83% of complete words, yet recognize fewer fonts. Deskew/windows retain 91.01% of ink and 60.67% of complete words while improving recognition. Deskew reduces the line's height, so fixed-count windows cover less horizontal extent at higher glyph resolution. Coverage alone is not sufficient; the next glyph-group experiment must preserve both boundaries and useful letter size. These are pixel-center geometry measurements, not OCR or information-retention measurements after rescaling.

The inexpensive deskew estimate uses ink-row concentration, bounded to ±8°. It acts only on sufficiently wide text with a meaningful score improvement. It corrects baseline rotation, not font slant. Upright no-op decisions return the original sampler arrays exactly. Median Node preparation was 0.30 ms for deskew/windows versus 0.26 ms for the original sampler on this bank; these are not browser end-to-end timings.

## Matched training

Both arms start from the same preserved checkpoint, use 20,000 updates, batch 64, AdamW 0.0005 → 0.000025, the same family/renderer sampling, and keep step zero eligible. Only the preparation differs. The experiment was repeated with another seed. Checkpoint selection uses the development bank, then weights are exported to int8.

| Seed | Existing-preparation control | Deskew/windows | Difference |
| --- | ---: | ---: | ---: |
| 20260925 | 61.16% | 63.58% | +2.42 points |
| 20260927 | 60.95% | 63.95% | +3.00 points |

Detailed reports: [control 1](next-current-20260925.json), [deskew 1](next-deskew-20260925.json), [control 2](next-current-20260927.json), [deskew 2](next-deskew-20260927.json). Each run took about 4–6 minutes on the M4 Max, excluding preparation. The second deskew seed is the selected candidate in [models/deskew](../models/deskew/README.md).

This improvement does not satisfy the reliability gate. Selected-candidate development accuracy is 75.63% clean, 59.94% rotated and 45.69% small. Its rejection calibration accepts only **6.39%** of known queries while meeting the fitted unknown-acceptance constraint. Near-perfect accuracy among that small accepted fraction does not mean the recognizer is broadly reliable.

## Verification and integration

New post-selection strings and six absent families were prepared separately under `.data/verification`; thresholds were never fitted there. [verification.json](verification.json) records 6,400 known-family crops plus 384 absent-family crops:

| Frozen model | Known top-1 | Rotated top-1 | Clean 8+ letters | Rotated 8+ letters |
| --- | ---: | ---: | ---: | ---: |
| Deployed baseline | 57.47% | 44.06% | 97.50% | 43.50% |
| Deskew candidate | **62.59%** | **60.00%** | **97.75%** | **85.50%** |

Each length/condition cell contains 400 crops from two texts, 100 fonts and two renderers. Single-glyph and two/three-letter accuracy remain only 39.50% and 40.25% across all conditions. The candidate accepts 373/6,400 known queries, with 372 correct; none of the 384 absent-family crops pass the fixed threshold. That is **5.83% coverage**, far below the 50% gate. The old final tests remain historical; these new generated crops still do not replace independently sourced screenshots.

The experimental normalizer is versioned by hash in the candidate artifact. It returns input tensors and inverse source polygons together. The main demo retains its original model/normalizer while the separate website work proceeds; integrating only the candidate weights would feed it the wrong preparation.

Regression tests cover ink coverage of grouped regions, detached dots, smallest/blank/malformed inputs, opposite deskew directions, source polygons, no-op byte identity, and repeated A → A → B → blank → A calls. A real optimizer-step test proves a worse continuation retains the starting checkpoint and exact exported weights. Candidate CPU/WebGPU parity is recorded in [deskew-browser.json](deskew-browser.json).

```sh
node scripts/preparation.mjs
node scripts/coverage.mjs
node scripts/reprepare.mjs
node scripts/python.mjs -m train.next current --seed 20260925
node scripts/python.mjs -m train.next deskew --seed 20260925
# Repeat both with --seed 20260927; freeze the selected artifact before verification.
node scripts/preparation.mjs verification
node scripts/python.mjs -m train.verification
```
