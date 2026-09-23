# Encoder and reference refinement

This iteration addresses failed recognition of the demo's Lora and Montserrat samples. It changes the learned embedding and the catalog references; it does not restrict the search to the sample's known identity. Every Google query still searches all 2,004 eligible families.

**The first candidate is held back.** Fresh phrase top-1/top-5 improves from 26.03%/44.91% to 38.72%/63.27%, but the reported Lora example falls from rank 2 to 11. Montserrat moves from 29 to 16. These regressions fail the practical requirement despite better aggregate metrics. The deployed encoder remains unchanged while case-diverse training is investigated. The first phrase benchmark is now opened diagnostic evidence for that next iteration.

## Selection protocol

The original encoder averaged reference images across scripts into one vector per family. Frozen-weight development comparisons tested that mean, separate script means, and four/eight spherical clusters. Separate scripts won. This preserves distinctions that disappear when Latin, Greek and Cyrillic are averaged together.

The refinement starts from the original classification checkpoint. It runs 12,000 additional updates on the same 1,403 training families, pairing different text and renderers. The objective is the original classification cross-entropy plus 0.5 times the symmetric episodic retrieval loss. The classifier remains training-only. AdamW starts at 0.0002 and decays to 0.00001. The architecture remains 349,920 parameters and a 128-dimensional embedding.

Checkpoints are selected on the existing clean development subset every 2,000 updates. Step 10,000 wins. Full development selection then compares the original and refined int8 encoders, script means, and extra lowercase/uppercase reference means. The extra reference strings are `hamburgefontsiv` and `packmyboxwithfivedozenliquorjugs`, rendered at 32 and 56 px. Unsupported strings are excluded by the font's recorded glyph coverage. They are references, never extra optimizer examples.

The full selection criterion is the harmonic mean of known/unseen macro family top-five accuracy, using 143,280 development queries and 1,704 candidates. The [frozen selection](encoder-quality-selection.json) records all candidates. The [quality report](encoder-quality.json) records subsequent confirmation results. Original final queries are now historical regression checks, not an untouched test. No reported demo example or fresh phrase query is used to select weights or reference recipes.

## Confirmation

The fresh phrase benchmark was frozen before refinement results: 1,356 clean Chromium crops from 113 demo fonts, at 32 and 56 px. Texts are `Amber clouds drift`, `Hidden paths unfold`, `Crisp winter light`, `Baked figs and honey`, `aRneGQop`, and `ri`. It tests text and scale variation under the demo renderer, not independent photographs. Reported `Quiet rivers flow` cases are kept as explicit regressions and are separate from this benchmark.

The model is still experimental. Similarity is not calibrated certainty. Very short crops, shared outlines, unavailable faces and real-image degradation remain limitations. Face metadata does not establish weight/style prediction accuracy.

## Reproduce

The case-diverse continuation adds 87,936 Pillow/Chromium training crops from 1,374 eligible training families. Half its batches use these lowercase/title-case/uppercase strings and half retain the original script-balanced data. It completes 20,000 updates in 1,361 seconds. Checkpoint selection uses the original clean development subset plus the opened first phrase bank, restricted to training/development families. Lora remains held out from optimization.

The [second confirmation report](encoder-case-quality.json) uses another 1,356 crops, with new strings at 24/48 px. Their exact strings are disjoint from the original data, additional training data and references. Top-1/top-5 improves from **23.60%/40.56% to 37.76%/66.89%**. However, Lora ranks 9 and Montserrat 26 on the reported example, so this checkpoint is also held back. The first and second phrase percentages must not be compared directly: their text and size distributions differ.

Additional diagnostics fit a training-only affine head and compare averaging reference recipes. They improve some aggregate scores but do not resolve the named failures. Their scripts and local reports are retained under `train/encoder_metric.py`, `train/encoder_reference_means.py` and `.data/detection-quality/`.

After the original encoder experiment and datasets exist:

```sh
node scripts/python.mjs -m train.encoder_references compare
node scripts/encoder-quality.mjs
node scripts/python.mjs -m train.encoder_refine
node scripts/python.mjs -m train.encoder_word_refs
node scripts/python.mjs -m train.encoder_quality select
node scripts/python.mjs -m train.encoder_word_refs --phase final
node scripts/python.mjs -m train.encoder_quality confirm
```

The refinement and selection commands refuse to overwrite completed experiments. Embedding caches bind encoder and dataset hashes. Confirmation writes the selected Google catalog to `.data/detection-quality/google-fonts.json`; publishing requires copying the selected encoder/checkpoint and that catalog into `models/encoder/`, then recompiling image-reference catalogs with the new encoder. Existing catalogs from the previous encoder must not be mixed with it.

The reference comparison also retains clustered-prototype results in `.data/detection-quality/reference-comparison.json`. Checkpoint history, prepared reference pixels and frozen query tensors stay under `.data/`. No font binaries or third-party preview captures are added by this iteration.

For the case-diverse continuation:

```sh
node scripts/python.mjs -m train.encoder_words plan
node scripts/python.mjs -m train.encoder_words render
node scripts/encoder-words-browser.mjs
node scripts/python.mjs -m train.encoder_words pack
node scripts/encoder-quality.mjs --next
node scripts/python.mjs -m train.encoder_refine --case-training --steps 20000
node scripts/python.mjs -m train.encoder_case_quality
```

The first phrase generator is preserved in commit `c7789b2`; extending it for the second bank changes its code hash. Existing pixel snapshots retain their original provenance. Reproducing a historical confirmation requires the recorded generator version, not silently changing its pins.
