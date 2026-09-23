# Encoder and reference refinement

This iteration addresses failed recognition of the demo's Lora and Montserrat samples. It changes the learned embedding and the catalog references; it does not restrict the search to the sample's known identity. Every Google query still searches all 2,004 eligible families.

**The demo now uses the compact refined encoder with script-separated references**, selected by the recovery comparison below. The [fresh confirmation](encoder-recovery-quality.json) improves top-1/top-five from 21.24%/37.61% to 24.41%/45.72%. This remains weak recognition. Lora's reported sample ranks first with a narrow margin; Montserrat still ranks twelfth. [Model, Google catalog and required modules](encoder-recovery-runtime.json) total 1,723,636 bytes, below 10 MB. All source catalogs were regenerated with the selected encoder.

The initial script-plus-words candidate was held back. Its first phrase top-1/top-5 improves from 26.03%/44.91% to 38.72%/63.27%, but the reported Lora example falls from rank 2 to 11. Montserrat moves from 29 to 16. Those regressions fail the practical requirement despite better aggregate metrics. The first phrase benchmark became opened diagnostic evidence for subsequent iterations. Results from different phrase banks must not be compared as if they used identical queries.

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

## Text dependence and capacity

The [text-dependence diagnostic](encoder-text-dependence.json) reuses opened development images as references. It searches the same 1,672 Latin families with one Pillow reference each and Chromium queries at 56 px. With the case-trained encoder, held-out-family top-1 is 75.0% for matching text and 43.0% for different text. The original encoder scores 70.3% and 37.8%. The two strings also differ in length and crop geometry, so this isolates neither glyph identity nor OCR benefit. It establishes that good matching of the same specimen does not establish text-independent recognition.

The capacity experiment doubles convolution channels to 64/128/192/256/256 while retaining 128 output dimensions and the same preparation. It has 1,363,264 parameters. Widening duplicates existing filters and divides their incoming weights to preserve the initial function; a small training-only perturbation breaks symmetry. Tests check the preserved projections, batch padding, repeated inputs, gradients and int8 export. The run uses the same case-diverse data and checkpoint criterion, for 12,000 updates. It is a bounded capacity probe, not an equal-update comparison with the 20,000-step case run.

The third phrase bank is frozen before capacity results: `Silver branches sway`, `Warm stone arches`, `Painted clay vessels`, `Fresh thyme and sage`, `wBNgkfea`, `es`, at 26/52 px. Its strings are disjoint from all earlier training, reference and phrase banks. It is confirmation data, not checkpoint-selection data. The capacity run selects step 12,000 and completes in 1,773.6 seconds. Its encoder is 1,856,928 bytes. Script-only references put Lora/Montserrat at ranks 2/20; adding word references gives 9/23. It fails the recovery gate and is held back. [CPU/Metal parity and payload](encoder-large-runtime.json) are verified independently of deployment.

```sh
node scripts/encoder-quality.mjs --large
node scripts/python.mjs -m train.encoder_refine --case-training --large --steps 12000
node scripts/python.mjs -m train.encoder_case_quality --large
node scripts/python.mjs -m train.encoder_text_diagnostic --large
node checks/encoder.mjs .data/encoder/large-refine/encoder.json .data/detection-quality/large-google-fonts.json bench/encoder-large-runtime.json
```

Commit `3d553b9` preserves the second-bank generator and case-training implementation before adding the capacity experiment. Historical snapshots keep those hashes. The runtime check accepts explicit candidate paths so it can verify CPU/Metal parity and payload size before deployment.

## Recovery selection

Before opening the third phrase results, compare the three frozen refined encoders with script-only and script-plus-word references. All eight reported demo examples belong to training/development families. They now become explicit development-selection cases: every rank must improve or tie, and Lora/Montserrat must improve strictly over the deployed baseline. This is an improvement gate, not a claim that all eight become first-place matches. Full-development harmonic known/unseen macro top-five must also improve. Prefer the smallest encoder within one absolute percentage point of the best eligible development score.

This stage changes the earlier protocol deliberately: reported examples are no longer independent confirmation. No final-family query or third phrase result selects the candidate; final-family reference images do supply catalog distractors. The chosen encoder, reference method, all candidate scores and source hashes are frozen in `bench/encoder-recovery-selection.json` before the third phrase confirmation. The script refuses to overwrite that selection.

```sh
node scripts/python.mjs -m train.encoder_recovery
node scripts/python.mjs -m train.encoder_case_quality --recovery
```

The [frozen recovery choice](encoder-recovery-selection.json) is `refined-scripts`. On the historical final queries, unseen-family top-one/top-five rises from 26.78%/47.31% to 39.59%/63.06%. The reported cases are development evidence, not independent accuracy claims:

| Sample | Previous rank | Current rank |
|---|---:|---:|
| Lora | 2 | 1 |
| Montserrat | 29 | 12 |
| Inter | 26 | 9 |
| Roboto | 42 | 29 |
| Poppins | 213 | 17 |
| Merriweather | 8 | 5 |
| Playfair Display | 1 | 1 |
| Geist Mono | 6 | 2 |

The next quality experiment should address the measured text dependence: compare training with glyph identities against the current objective at a fixed budget, with whole-glyph boundaries and known labels during synthetic training. A training-only glyph head can be tested before committing to runtime OCR or a larger download. Its benefit remains unproven. Require improvements on fresh text and Montserrat/Roboto confusions; do not mistake the new Lora ranking for reliable general recognition.
