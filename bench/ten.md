# Ten-font neural classifier — 2026-09-22

The deployed int8 model identifies the correct family first on **633/640 clean test images (98.91%)**, and **3,574/3,840 across six conditions (93.07%)**. Top-3 across all conditions is **98.96%**. This is a word-disjoint synthetic Chromium test of ten pinned regular faces, not accuracy on arbitrary screenshots.

The live demo smoke check identifies **9/10** samples for “Quiet rivers flow”: **Open Sans is incorrectly ranked below Source Sans 3**. This concrete failure remains recorded in [ten-demo.json](ten-demo.json); no phrase-specific override or training example was added to hide it. Similar sans-serif faces, partial crops and rotation still need improvement.

## Final test

The final test was kept out of training and checkpoint selection until the model was selected. Model hash: `e10a2ecd979a3dc51189a9c5b6a5069411dccfa90c0bf6ee5fccf23f34e71fe9`. [Full metrics/confusion matrix](ten-test.json), [training and validation](ten-training.json).

| Condition | Correct / images | Top-1 | Top-3 |
| --- | --- | --- | --- |
| Clean | 633/640 | 98.91% | 100.00% |
| Half-width crop | 578/640 | 90.31% | 98.12% |
| Rescaled to 60% | 585/640 | 91.41% | 98.28% |
| Rotation ±6° | 554/640 | 86.56% | 97.66% |
| Gaussian blur, radius 0.45 | 614/640 | 95.94% | 99.84% |
| JPEG quality 65 | 610/640 | 95.31% | 99.84% |

Each family has 64 distinct held-out texts, with six derivatives per text. Counts are balanced, so macro and overall accuracy coincide. Transformations are evaluated individually, not as arbitrary combinations. Text sizes were 20/28/36/48/64 CSS px, DPR 1 or 2; the small condition resizes rendered pixels to 60%. Cases include light text on dark backgrounds.

| Family | Clean | All six conditions |
| --- | --- | --- |
| Inter | 63/64 (98.4%) | 329/384 (85.7%) |
| Roboto | 63/64 (98.4%) | 353/384 (91.9%) |
| Open Sans | 64/64 (100.0%) | 326/384 (84.9%) |
| Source Sans 3 | 63/64 (98.4%) | 343/384 (89.3%) |
| Montserrat | 64/64 (100.0%) | 369/384 (96.1%) |
| Poppins | 61/64 (95.3%) | 360/384 (93.8%) |
| Nunito | 63/64 (98.4%) | 354/384 (92.2%) |
| Lora | 64/64 (100.0%) | 377/384 (98.2%) |
| Merriweather | 64/64 (100.0%) | 381/384 (99.2%) |
| Playfair Display | 64/64 (100.0%) | 382/384 (99.5%) |

A [phrase-cluster bootstrap](ten-uncertainty.json), keeping all fonts and derivatives together, gives a 95% interval of 91.2%–94.7% for all conditions and 97.8%–99.7% for clean text. This expresses sampling uncertainty within this generator; it cannot estimate unmeasured real-world performance.

## What was trained

- Ten-way family classification, cross-entropy with 0.03 label smoothing, AdamW and cosine learning-rate decay. This is a classifier, not a descriptor, OCR model, semantic font-parameter estimator or extensible embedding.
- Four 3×3 convolution stages: 1→16→32→48→64 channels, strides 1/2/2/2 and dilations 1/1/2/2. The last filters see a 29×29 receptive field instead of 17×17, without adding weights. ReLU follows each stage; global average pooling and a 64→10 head produce class logits.
- BatchNorm is folded into convolutions for deployment. The deployed model has **47,034 parameters**, per-output-channel symmetric int8 weights, float32 scales/biases and float32 inference arithmetic. Training includes 320 additional BatchNorm affine parameters.
- Variable input geometry is preserved. Batch padding is masked after every convolution; the tests prove that an individually evaluated tight tensor matches its padded-batch output. No padding is added to the inference tensor solely to fit a fixed display rectangle.
- Up to three aspect-preserving windows, each at most 128×48, are extracted from the selected crop. Blank borders are removed from the tensors. Softmax probabilities are averaged across windows. These probabilities are not calibrated confidence, and the model cannot reliably reject unknown families.

Preparation, byte quantization and source geometry are shared by the data generator and browser through [input.mjs](../src/input.mjs). The preview displays every pixel of these exact tensors; the result JSON exports the same arrays. This does not detect or name individual letters. It preserves the original font pixels instead of OCR/re-rendering them.

## Data and selection

[Configuration](ten.json), [pinned text pools](ten-texts.json), [font provenance](fonts.json).

The final corpus has **30,720 training source images**, 3,840 validation images and 3,840 test images; 102,848 prepared windows occupy 503.1 MiB as packed bytes. Training uses 512 strings × ten fonts × two renderers × three views (clean, a deterministic mixed augmentation, and rotation). Pillow and real Chromium Canvas are sampled equally, with uniform family sampling. Validation has 32 strings per font in both renderers; the final test has 64 strings per font in Chromium.

Underlying words, not just complete strings, are disjoint across train/validation/test. All image derivatives stay in their source split. Font files, text pools, preparation code, generator, tensor bytes and exported model are hash-checked. The test rendering uses the same family of generator as training; it is a generalization test over new words/rendering conditions, not an independent image distribution.

Validation drove these experiments, all with 47,034 deployed parameters:

| Run | Training time | Validation top-1 | Chromium validation |
| --- | --- | --- | --- |
| Initial small corpus, CPU | 521 s | 85.18% int8 | 80.00% int8 |
| Equal renderer sampling, continuation | 473 s | 87.24% float | 84.79% float |
| Expanded browser coverage + clean/rotated views, MPS continuation | 92 s | 90.47% float | 89.06% float |
| Wider context, fresh MPS training (selected) | 173 s | 90.83% int8 | 90.16% int8 |

[Initial](ten-initial.json), [balanced](ten-balanced.json), and [expanded/local-context](ten-local.json) reports retain the intermediate measurements. They are not controlled single-variable architecture ablations: initialization, data or training schedules differ. The selected model starts from scratch; reproducing it does not require the earlier runs. About 21 minutes were spent in these recorded training/checkpoint-evaluation loops combined, excluding data generation and post-export checks. Early runs used CPU because sandboxed PyTorch could not access Metal; the final run uses the native GPU.

The final run takes 12,000 batches of 48 windows at learning rate 0.002, seed 20260922, and selects on validation only. Hardware: Apple M4 Max, macOS 26.5, PyTorch 2.11.0. The 173 s figure includes periodic validation but excludes data rendering/preparation, post-training quantization checks and the final test. It is one seed on one machine.

## Browser cost and verification

[Size measurements](ten-size.json): model JSON **49,416 bytes Brotli**; model plus four unminified preprocessing/inference modules **55,522 bytes (54.22 KiB) Brotli** when compressed separately. This excludes the demo UI and optional TTF previews and is not a published/minified npm package measurement. Preview fonts are loaded for presentation; inference uses only the image and neural weights.

On Chromium 151.0.7922.34 / Metal, 20 warm measurements of the three displayed Lora windows give GPU median **3.1 ms**, p95 **3.7 ms**; JavaScript CPU median **136.1 ms**, p95 **138.2 ms**. These include inference plus ranking, excluding image preparation, GPU initialization and font loading. They are not phone or end-to-end latency guarantees.

[Browser verification](ten-demo.json) checks actual WebGPU against independently decoded PyTorch logits (maximum error 0.00002289), plus CPU, repeated/concurrent calls, input ownership, minimum/odd/maximum dimensions, invalid-input recovery and device loss. UI checks cover byte-exact adaptive input previews, all ten samples, searchable popover keyboard/failure behavior, upload/paste/drop, blank/corrupt/transparent images, crop races and responsive widths 320/375/414/768/1440. Unit tests also verify gradient updates, BatchNorm folding, int8 round trips and packed-data boundaries/split leakage.

## gpu-* comparison and next training focus

The [gpu-* research](../research.md#gpu--comparison-for-the-ten-font-classifier-2026-09-22) compares the actual target peers. They train offline in PyTorch, apply deterministic task preparation, quantize weights and ship small browser runtimes. Our 47,034 parameters are roughly 14% above gpu-lexer's 41,321 deployed weights and 21% above gpu-time's 38,745 parameters. Our current 55,522-byte compressed core is about twice gpu-lexer's reported 28,305-byte minified package and 22% above gpu-time's 45,561-byte package. The packaging and tasks differ; equal parameter counts would not imply equal image-processing cost.

OCR is not a prerequisite. Tight text windows plus a small supervised classifier now provide measurable recognition at a modest local training cost. A separate letter recognizer would add its own labeling, training, runtime and error propagation; its savings have not been demonstrated. If future failures come mainly from locating text, evaluate a separate line detector before a full OCR pipeline.

Next: collect verified screenshots for these ten faces, split by source before tuning, and target similar-sans/rotation/partial-crop failures with new development examples. Keep this test as a frozen historical benchmark; future model selection must use development data and reserve a new final test. Bold, italic, variable axes, mixed fonts, large rotations, textured backgrounds and absent-family rejection remain outside the measured scope.

## Reproduce the selected model

After the setup in the [README](../README.md). The import pins existing local font files and creates regular weight-400 instances; it does not modify `fontr`, which is unnecessary afterwards.

```sh
npm run data:import -- --source ~/projects/fontr/data/fonts_collected/google
npm run train:ten
node scripts/ten.mjs evaluate
npm run demo:build
npm run demo
# In another terminal:
npm test
npm run test:demo
```

`bench/ten.json` selects MPS; explicitly set `training.device` to `cpu` if Metal is unavailable. `node scripts/demo-build.mjs --validation` is available for development previews before unlocking a final test; its UI labels those metrics as validation. The normal build requires the matching final-test report.

## Post-training review

The review corrected `--steps 0` silently selecting the default run and negative overrides reaching an old checkpoint. Training now requires a positive integer. The training report retains the source hash from the actual training run; this later argument-validation fix does not alter trained weights or inference. The deployed model, preparation hashes, test metrics, browser report and size report were rechecked against the same artifact.

Direct regression evidence:

- `test_invalid_step_counts_reject_before_model_or_checkpoint_access`: explicit 0, −1, false and 1.5 overrides, plus omitted overrides with configured 0/−1, raise before constructing a model or reading a checkpoint.
- `test_one_step_trains_and_replaces_an_old_checkpoint`: a one-step override against a 12,000-step default trains on two families/two renderers, replaces an unusable previous checkpoint, changes classifier bias, and exports a matching model/report with step 1 and two validation images.
- `smallest two-tone image preserves its grayscale ramp through A → blank → A`: a 2×1 black/white RGBA image produces a monotonic grayscale row repeated vertically; A → A → 1×1 blank → A retains exact pixels and tight edges.
- `packed weights preserve signed values and reject either side of the final byte`: a one-class artifact decodes −128/127 at scale 0.5 to −64/63.5; empty, one-byte-short and one-byte-long final tensors reject. Each error is followed by a valid decode, exact logits `[1]`, and class probability 1; an earlier decoded model remains valid.

The falsey step-override pattern occurs only in the ten-font trainer. Earlier fixed-configuration experiments have no `--steps` override or old-checkpoint reuse path. Temporary test directories are removed by their context managers. This review changed trainer validation and tests, not the browser hot path; the existing latency and size measurements remain applicable.
