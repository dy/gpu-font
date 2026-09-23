# 100-font recognition experiment

The selected model has been evaluated once on the previously unused final test and deployed in the demo. It is useful on clean longer text; short-fragment reliability is not solved.

The browser UI now has a searchable grid of real font previews, an Open image icon beside Image, a clickable empty state, movable/resizable crop handles, and a source-resolution slider. Preview canvases still display exactly the inference tensors. Crop handles support dragging, click-then-click placement, and keyboard movement. Home/Escape selects the full image.

## Training contract

100 distinct named regular families from the local Google Fonts collection, including the original ten. Another 13 families are excluded from training: seven for rejection validation, six for the final unknown-font test. This is a local selection, not a claim that these are the 100 most popular fonts. See [font provenance](fonts-100.json).

- 254 training strings: all 62 ASCII letters/digits, 64 random two/three-character strings, 64 words, and 64 phrases. Both Pillow and Chromium render every training string for every known family, in clean and augmented views: 101,600 source training images.
- Validation: 22 strings × 100 families × six conditions = 13,200 known-font images, plus 924 unknown-font images. The user-reported `ri`/`riv` cases are development examples.
- Final test: 36 strings × 100 families × six conditions = 21,600 known-font images, plus 1,296 unknown-font images. The final test is separate from training/model selection.
- Multi-character strings are disjoint across splits. Single characters deliberately share the alphabet, with different rendering plans. This tests new renderings and combinations, not unseen character recognition.
- Conditions: clean, a cut final glyph, 45% source resolution, ±6° rotation, a faint displaced shadow, and large plain margins. Length bands describe the source text before the edge cut. Training also includes blur and JPEG compression.
- Shared JavaScript normalization produces 237,844 variable-size windows, packed as 1,076 MiB of grayscale bytes. Each window is at most 128 × 48. No OCR or hand-written font descriptor is involved.

## Model selection, fixed before final evaluation

Start from the existing ten-font CNN's features, expand its classifier to 100 outputs, and train all weights for 40,000 steps of 64 crops. Balance families and renderers. Select checkpoints using known-font validation, never final test.

Compare two continuations from exactly that checkpoint: 20,000 steps at the same learning-rate schedule and seed, with and without one additional 64-channel convolution. The extra layer expands receptive field from 29 to 45 pixels, at the low-resolution end of the network. Keep it only for at least two absolute validation accuracy points over the smaller continuation. Preserve both reports and artifacts. This comparison separates added capacity/context from additional optimizer steps; it does not separately identify which effect of the added layer helped.

Fold batch normalization, quantize weights to int8 with float32 arithmetic, and verify exported validation accuracy loses less than one point. Fit temperature using known-font validation negative log likelihood. Select an uncertainty threshold using validation only, targeting at least 95% correct among accepted known-font crops and at most 5% false acceptance of validation unknown fonts. Report coverage; rejecting everything cannot establish recognition quality. Both temperature and threshold are used unchanged on the final test.

The first 40,000-step run has 52,884 parameters and took 604.9 seconds on Apple M4 Max MPS. Its validation top-1 is 63.23% overall; 95.7% on clean longer text, but only 43.0% on single letters across conditions. The threshold meeting the two rejection targets covers only 8.1% of known queries. Those figures are evidence of remaining limitations, not a successful reliability gate. [Baseline report](hundred-baseline.json).

## Final results

| Source characters | Clean top-1 | All views top-1 | All views top-3 | Images, all views |
| --- | ---: | ---: | ---: | ---: |
| 1 | 54.0% | 43.2% | 65.8% | 3,600 |
| 2-3 | 59.2% | 53.4% | 74.1% | 6,000 |
| 4-7 | 90.1% | 82.6% | 93.1% | 6,000 |
| 8+ | 97.8% | 87.3% | 94.3% | 6,000 |

Overall: **69.23% top-1**, **83.60% top-3**, on 21,600 synthetic known-font images. Every family has the same number of images. Clean and margin-only conditions have identical pixels after preparation, so their equality is an invariance check, not independent evidence. Rotation: 52.33% top-1; 45%-resolution images: 60.67%. [Full per-font and condition results](hundred-test.json).

At the unchanged validation threshold, **2,030/21,600 (9.40%)** known-font crops are accepted; **2,026/2,030 (99.80%)** are correct. Four confident errors remain. **0/1,296** test images from the six held-out unknown families are accepted. These are repeated synthetic strings from a small family set, not 1,296 independent real-world fonts; the low observed false-accept rate does not establish universal rejection. Low coverage means the reliability gate is **not met**. The demo still shows ranked candidates for uncertain crops, without presenting the old raw scores as confidence.

The added convolution wins the equal-budget comparison: 66.62% int8 validation top-1 versus 63.78% for the smaller continuation (**+2.84 points**). It has **89,812** folded parameters. The model and four core modules total **91.23 KiB Brotli**, including **85.13 KiB** of compressed model JSON; font preview files and demo UI are separate. [Selection](hundred-selection.json), [size accounting](hundred-size.json).

Training/checkpoint-validation time: 604.9 s initial run, 312.4 s context continuation, 278.9 s control continuation; **19.94 minutes** total on Apple M4 Max MPS, excluding rendering, packing, quantization/calibration, browser tests and the final evaluation. One seed was tested. The selected model's own lineage used 15.29 minutes. A 151×151 validation grid combining softmax and logit energy offered no coverage gain; it is excluded from inference. [Energy method reference](https://arxiv.org/abs/2010.03759).

Model SHA-256: `66b98a49d1bebe9b2e56bc896f1fbe9b8e76ece52e74608aeae58ef3dd4cefbb`.

## Reproduce the selected model

After the setup in the [README](../README.md) and the ten-font import in [ten.md](ten.md). Training uses Apple MPS; set `training.device` in `bench/hundred.json` to `cpu` elsewhere.

```sh
node scripts/python.mjs -m train.hundred_data import --source ~/projects/fontr/data/fonts_collected/google
npm run train:ten
node scripts/hundred.mjs data
node scripts/python.mjs -m train.hundred train
# Preserve the initial model.json, best.pt, calibration.json and training report
# if repeating the matched smaller-model control.
node scripts/python.mjs -m train.hundred train --resume --context --steps 20000
node scripts/hundred.mjs evaluate
npm run demo:build
npm run test:demo
```

The control is a 20,000-step `--resume` continuation without `--context` from the same initial 100-font checkpoint. It is **not** continued from the context model. Archive files are under ignored `.data/hundred/baseline`, `context`, and `control`; their reports are retained in `bench/`.

Final test is now open and historical: reserve new test strings/source images before further model selection. The trainer source hashes record the actual runs. Later resume-initialization guards and a final calibration-hash check leave selected weights unchanged. A generator cold-start determinism fix was verified to reproduce every existing plan, source pixel and source record before updating its preparation provenance.

## Browser verification

The deployed model agrees with independent PyTorch logits to within 0.000031 on minimum, maximum, odd-sized and real font inputs. The actual Metal WebGPU check exercises five convolution stages and all 100 output classes, concurrent A → A → B → A reuse, input ownership, invalid-input recovery and device loss. Warm three-window inference plus ranking measures 3.3 ms median / 3.6 ms p95 on Chromium 151 / Apple M4 Max; preparation, asset loading and initialization are excluded. CPU median is 167.2 ms. [Browser report](demo.json).

The default “Quiet rivers flow” smoke specimen is correct for 92/100 families. Across the original ten, `ri` and `ri` plus a two-pixel slice of `v` are each correct for only 2/10 fonts; all of these fragments are marked uncertain. `rivers` is correct for 9/10. These targeted diagnostics expose a real remaining failure, rather than demonstrate short-fragment success. In particular, Lora `ri` currently ranks Libre Baskerville first; longer Lora text is recognized correctly.

Validation includes 36 Node tests and 36 Python tests, plus the full browser lifecycle at 320/375/414/768/1440 px. Exact preview pixels and exported input arrays match at 100%, 50% and 10% source resolution; returning to 100% restores the original arrays. The crop can move/resize using handles, keyboard or click-then-click placement. Cancellation, foreign pointers, live-update coalescing, replacement races, blank/corrupt/transparent input and font-load recovery are checked. Screenshots are under ignored `.data/demo-checks/`.

## Next measured experiment

Rotation and downscaling are the largest controllable failures. Inspect the actual windows before spending more training budget: a rotated long line can have a much taller ink box, reducing the effective letter detail in a window. Compare text-line deskew/height estimation with the current preparation on a fresh validation set. OCR transcription is not required for that experiment. Each training string currently has one base size/DPR plan per renderer. Expand each glyph and common short-string context across multiple raster sizes and combinations, rather than assuming that 101,600 correlated source images cover those variations. Independently labeled screenshots and broader short-string renderings are needed before claiming arbitrary-image reliability; do not expand the catalog again to hide the existing error rate.

## Interpretation

Training on single letters and random strings encourages the same font label across different content. It does not prove that the network has learned a disentangled notion of font style. Text recognition is a separate task: an OCR front end may help isolate text, but cannot recover information missing from a low-resolution `ri`, or distinguish identical outlines. Automatic text-line selection/deskew needs its own measured cost/benefit experiment.

A crop with only added plain background must have identical normalized pixels; `extra plain background changes source coordinates but not the model pixels` verifies this. Additional marks, other lines, shadows or partial neighbors can change the detected ink bounds. More training cannot restore stroke detail already lost during normalization/downsampling.

No independent real-screenshot accuracy, font weight/italic recognition, or universal open-set rejection is established here. Fonts can share glyph designs. There is no promise that every possible combination identifies a unique family.

Synthetic plus real-data domain adaptation is part of [DeepFont](https://arxiv.org/abs/1507.03196); its results are not directly comparable to this much smaller closed-set task. [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) addresses probability calibration; it is not by itself an unknown-font detector.

## Review follow-up

Fixed a resume regression: the 100-font trainer previously initialized its best score to −1, so a worse continuation could replace the starting checkpoint. It now evaluates and retains the initialized starting weights as step zero. The ten-font trainer already compared against its starting score. Both selected continuations in this report improved over their starting model; the deployed artifact and measured results are unchanged.

Direct regression evidence in `tests/test_hundred.py`:

- `test_worse_one_step_resume_preserves_starting_weights` and `test_worse_one_step_context_resume_preserves_initialized_weights`: a two-class checkpoint with Pillow/Chromium training windows, two known validation samples and one unknown sample; perform a real optimizer step, supply a worse validation result, then verify every restored quantized tensor and the exported artifact exactly match the initial four- or five-convolution model. Step zero remains selected.
- `test_invalid_steps_reject_before_resume_model_access`: overrides 0, −1, false and 1.5, plus configured 0/−1 defaults, all reject before reading a resume artifact.
- `test_empty_prediction_and_missing_calibration_groups_fail_before_inference`: zero selected windows fail before calling the model; all-known and all-unknown calibration sets each fail explicitly instead of producing NaN statistics or an unsupported rejection threshold.
- `test_manifest_rejects_unknown_training_leakage_and_packed_boundaries`: five valid 1×1 windows decode exactly to `[0, 64, 128, 192, 255]`; empty, four-byte and six-byte payloads reject. Fractional width, boolean source/height and null window records also reject. The same type-validation gap was fixed and directly tested in the ten-font loader's `test_manifest_rejects_leakage_stale_files_and_invalid_packed_boundaries`.

Full suite: 36 Node + 36 Python tests pass. The real ten- and 100-font manifests also load under the stricter validation. No inference code, UI or trained artifact changed in this review; they remain byte-identical to the last WebGPU/browser-verified build. The training-only resume check adds one initial validation pass, and does not affect inference cost or bundle size. Temporary probes were removed.
