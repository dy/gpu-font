# Compact learned retrieval

This tests neural features, not the archived handcrafted descriptor. The preserved 100-family network produces 64-dimensional features. Reference vectors are averages of normalized specimen features, quantized to int8 with a float32 scale and decoded to unit length for cosine search. A source's window features are averaged with equal weight before catalog construction/search.

The catalog contains the existing 100 families plus seven development families that never received encoder optimization. Prototype and query multi-character strings are disjoint. Single-character alphabets overlap for the known-font development bank. Added-font prototypes use one half of the historical unknown-validation text pool and queries use the other half. Those families were previously used for rejection calibration, so this is development evidence. The six unknown-test families do not enter this catalog.

Known queries comprise the new 6,400-image preparation bank; added-font queries are 462 different-text crops. Their text/condition distributions differ, so their scores should not be compared as equal-difficulty groups. All retrieval queries compete against **107 families**.

| Frozen feature index | Known top-1 / top-5 | Added top-1 / top-5 |
| --- | ---: | ---: |
| One reference per family | 48.09% / 76.80% | 56.71% / 79.65% |
| Four references, one per length band | 46.03% / 75.39% | 54.98% / 78.35% |
| Trained-head projection, one reference | 45.16% / 74.59% | 56.49% / 76.41% |
| Trained-head projection, four references | 45.11% / 73.84% | 57.79% / 80.52% |

The trained-head projection factors the existing classifier's weight matrix with SVD. It preserves bias-free logit cosine geometry in at most 64 dimensions; it does not add a pixel heuristic or use new supervision. It failed to improve known-family retrieval overall. [retrieval.json](retrieval.json) contains the full slices and hashes.

One 64-dimensional int8 reference plus its scale costs **68 bytes per family**: 7,276 bytes for this 107-family catalog, or 680,000 bytes at 10,000 families. Four references would cost 2.72 MB at 10,000 families. These figures exclude encoder, labels, metadata and runtime, and say nothing about recognition quality at that scale. Adding the seven families leaves the encoder hash unchanged.

## Matched training result

Both arms start from the same checkpoint and train for **12,000 updates**. Each batch has 32 distinct families, each rendered in two different training strings; the strings are shared across the families to supply same-letter negatives. Renderer/window choices are identical for the same seed. The control uses family cross-entropy; the other adds `0.1 × supervised contrastive loss` on normalized features, temperature 0.15. The encoder/head size is unchanged.

The harmonic mean of known/added development top-5 selects checkpoints, with the starting model eligible. **Both runs selected step zero.** Their exported artifacts are byte-for-byte identical to the preserved baseline. Continuing classification or adding this contrastive objective did not improve this retrieval recipe. Reports retain the complete learning curves: [control](metric-control.json), [contrastive](metric-contrastive.json). Updates took about 224 and 189 seconds respectively; timing includes evaluation and is not a controlled hardware-throughput comparison.

This does not establish that embeddings cannot work. It rules out promoting these particular continuations, the length-banded references and the frozen trained-head projection. The current classifier remains the stronger known-family baseline, and the known-font retrieval gate is unmet.

## Next bounded experiment

Before adding hundreds of families, compare the current encoder with one wider reference model and a training-only glyph head on verified single-character examples. Keep model size and update budgets explicit; test whether the bottleneck is capacity or character-content variation. Train/evaluate complete-glyph groups without reducing their raster height as aggressively as whole-word fitting. Keep the current model eligible. Use new development data for these choices: the newly opened verification set is now historical.

Expand real weight/style coverage from the successful ten-family pilot, then evaluate independently sourced, font-verified screenshots and catalog-specific rejection. A full corpus of poorly separated vectors would increase errors; it would not solve this failure.

The [pinned Google Fonts inventory](google-corpus.json) records 3,886 TTF files under `ofl`, `apache`, and `ufl` across 2,056 directories. These are not deduplicated canonical families or a claim of complete script coverage. Aliases, related versions, genuine faces/axes, licenses and supported scripts still need resolution before a 500-family split or full-corpus training.

```sh
node scripts/python.mjs -m train.retrieval
node scripts/python.mjs -m train.metric control
node scripts/python.mjs -m train.metric contrastive
```
