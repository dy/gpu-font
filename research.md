# Phase 0 research

Reviewed 2026-09-21. Sources below are primary repositories, training code, and project documentation. Reported upstream accuracy and speed are not measurements of gpu-font.

2026-09-22 follow-up: [scaling within 10 MB](bench/scaling.md) rechecks the gpu-* model cards, calculates classifier/catalog storage, reproduces the clipped Geist Mono example, and compares letter detection, OCR, deskewing, skeletons and Fourier normalization. It defines the next controlled preparation and neural-retrieval experiments. The original 100 KiB target below is historical; the current deployment ceiling is 10 MB.

## How the gpu-* projects train

| Project | Supervision and training | Browser implementation | Relevant lesson |
| --- | --- | --- | --- |
| [gpu-lexer](https://github.com/vercel-labs/gpu-lexer/tree/07f3e56c69f3141c1bcdb9a732aac14889ca5c74) | Shiki labels source code offline. JavaScript prepares corpora; [PyTorch trains the network](https://github.com/vercel-labs/gpu-lexer/blob/07f3e56c69f3141c1bcdb9a732aac14889ca5c74/packages/training/torch/train.py), including quantization-aware and distillation paths. | Custom WGSL and JavaScript; 41,321 deployed int6 weights. | Reuse a reliable label source; separate the training environment from the delivered runtime. |
| [gpu-time](https://github.com/arikchakma/gpu-time/tree/c7fea0662c158a22e2aa404e27542f23131003c3) | Generated schedules and phrases, plus harvested text with teacher/verification rules. [PyTorch with AdamW](https://github.com/arikchakma/gpu-time/blob/c7fea0662c158a22e2aa404e27542f23131003c3/packages/training/torch/train.py); role, boundary, and optional structured/distillation losses. | TypeScript CPU path or WGSL; calendar resolution remains ordinary code. | Predict the uncertain part; compute deterministic rules normally. Split phrase families, not merely random strings. |
| [gpu-query](https://github.com/safzanpirani/gpu-query/tree/41739a28ecc930ad9c550ec347c602dd6a3308cc) | Synthetic queries with known schemas/filters. [PyTorch with AdamW](https://github.com/safzanpirani/gpu-query/blob/41739a28ecc930ad9c550ec347c602dd6a3308cc/spike/train.py); role cross-entropy and boundary loss. | TypeScript or WGSL using int6 weights; deterministic schema matching and filter compilation. | Test unseen schemas explicitly. Its generated-corpus results do not establish real-user accuracy. |
| [neural-flexbox](https://github.com/aaronvanston/neural-flexbox/tree/d417801be3a97f6fde01b419369eff2ffc6a1e27) | [Node/Playwright renders layouts](https://github.com/aaronvanston/neural-flexbox/blob/d417801be3a97f6fde01b419369eff2ffc6a1e27/packages/training/src/generate.mjs) and reads browser geometry. [PyTorch with AdamW and smooth-L1 regression](https://github.com/aaronvanston/neural-flexbox/blob/d417801be3a97f6fde01b419369eff2ffc6a1e27/packages/training/src/experiment.py); later int8-aware training/calibration. | Small model, browser runtime, quantized export. | Browser-generated ground truth and offline Python training work together. |
| [tinySarf](https://github.com/AhmedAbdel-Aal/tinySarf/tree/faaa6a8fea6315b026bb5fa0bd731dbf31ff9e6b) | CAMELMORPH/CAMeL Tools teacher data; [PyTorch/AdamW](https://github.com/AhmedAbdel-Aal/tinySarf/blob/faaa6a8fea6315b026bb5fa0bd731dbf31ff9e6b/packages/training/src/tinysarf_training/train.py). Its [model card](https://github.com/AhmedAbdel-Aal/tinySarf/blob/faaa6a8fea6315b026bb5fa0bd731dbf31ff9e6b/MODEL_CARD.md) records CPU training on an Apple M2, then a separate root-model experiment. | JavaScript reference and WGSL with int8 weights; additional decoding rules. | Teacher agreement, component accuracy, and complete-task accuracy are different. Independent human gold remains missing in this experimental release. |
| gpu-cron | Training implementation unverified. Searches did not locate a confirmed source repository, and [the reported demo](https://gpu-cron.vercel.app) could not be retrieved in this session. | Unverified here. | Do not use the earlier conversation's parameter count or inferred training stack as evidence. |

All five inspectable projects use PyTorch for training. Their npm-style commands can invoke Python underneath. None demonstrates that gpu-font needs PyTorch, that all training should happen in JavaScript, or that a similarly small font model will work.

The strongest reusable pattern is: generate trustworthy examples → train offline → evaluate held-out task behavior → quantize/export → implement only the required inference operations. GPU acceleration is workload-dependent; gpu-query and gpu-time explicitly retain CPU execution for small requests. See their pinned README/architecture documents.

## Preparing screenshots

Three different operations should stay distinct:

1. **Detect text regions:** find words/lines in a screenshot, potentially with different fonts, backgrounds, and orientations.
2. **Normalize a selected crop:** composite transparency, remove color, choose polarity, trim margins conservatively, resize uniformly, and pad.
3. **Identify the family:** compare the surviving visual features with a catalog.

Existing building blocks:

| Tool | Available capability | Fit for this project |
| --- | --- | --- |
| [Tesseract.js](https://github.com/naptha/tesseract.js) | Browser/Node wrapper around Tesseract WASM; OCR, structured layout output, and retrievable rotated/grayscale/binary images. Non-text outputs must be enabled explicitly. | Useful optional crop/OCR baseline. Reuse detected coordinates to crop original pixels; its OCR preprocessing is not automatically appropriate for font identity. |
| [PaddleOCR text detection](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html) | Dedicated text-region detection, without requiring recognition as the same step. The documented PP-OCRv5 mobile detector alone is 4.7 MB at this review. | Possible offline detector baseline or optional component. It exceeded the original 100 KiB proposal; under the newer 10 MB ceiling, count detection and runtime costs alongside recognition before choosing it. |
| [OpenCV.js](https://docs.opencv.org/4.13.0/d7/dd0/tutorial_js_thresholding.html) | Global/adaptive thresholding and Otsu threshold selection. | Reuse as an experimental reference if necessary; thresholding alone does not detect text or identify fonts. |
| Canvas + typed arrays | Image decode/readback plus small numeric transforms that we implement and test. | Starting point for the core crop normalizer; no OCR framework is needed for a caller-supplied line crop. |

Recommendation: preserve grayscale antialiasing in the model input. Binary thresholding collapses different intensity values to the same black/white value; the inference is that this may erase useful differences in thin strokes and edges. Compare grayscale against binary preprocessing experimentally rather than treating a visually cleaner binary image as better training data. A threshold can locate margins while the model still receives original grayscale samples.

Do not OCR and re-render the words: that replaces the evidence with another font. Do not resize every character independently: it changes width relationships and spacing. Begin with word/line images; connected-component extraction and character recognition are not prerequisites for the embedding experiment.

An arbitrary screenshot may contain several families. An eventual detector should return separate regions, not concatenate unrelated characters into one supposed font sample. Keep user-selected cropping available when detection is uncertain.

## JavaScript, PyTorch, and JZ

Use JavaScript for the first data/normalization experiment and eventual public API. Training need not be installed to establish input quality and evaluation fixtures.

[TensorFlow.js supports training](https://www.tensorflow.org/js/guide/train_models), including automatic gradients and custom optimization. It is a viable JS training candidate. Backend availability, supported gradients, tensor disposal, throughput, and weight-export parity must be tested on the actual machine. Writing JavaScript does not imply the arithmetic is JavaScript: [Node can bind to native TensorFlow](https://www.tensorflow.org/js/guide/platform_environment).

Selected for the first learned baseline: PyTorch offline, with a plain tensor/data boundary to JavaScript. The local gradient/update/export probe and 400-step pilot now run; see [results](bench/report.md). This follows the inspected implementations and avoids writing gradient/optimizer machinery. It is not an irreversible architecture choice, but there is no reason to maintain two full trainers for this pilot.

[JZ](https://jz.js.org/) compiles JavaScript to WASM. The matching local checkout at `/Users/div/projects/jz` reports version 0.9.2; the public site returned version 0.8.1 during this review, so those artifacts must not be assumed identical. The local README documents typed arrays, SIMD, explicit memory reset, and copying across the JS/WASM boundary unless using shared JZ buffers. It excludes DOM and Node platform APIs. These make numeric kernels a suitable experiment, but do not establish that an entire ML framework can compile unchanged.

JZ is a candidate for grayscale/resize loops, dot products, and eventually CPU convolution. It compiles the operations supplied to it; a training implementation still needs gradients, convolution backward passes, optimizers, batching, and checkpointing. Keep browser decoding/Canvas in JavaScript, compile coarse numeric operations, and measure allocation/copying as part of the cost. WASM CPU execution and WebGPU execution are separate backends.

### Local JZ smoke result

The included [probe](bench/jz-probe.mjs) compiles the same [grayscale kernel](bench/gray.mjs) that runs in JavaScript. Run it with:

```sh
node bench/jz-probe.mjs /absolute/path/to/jz
node --test
```

Observed with Node 25.9.0 and the local JZ 0.9.2 working tree based on commit `c5af69befdca3bbf59f1cae0b0fd8e446b0bb957`:

- 4,359 pixels checked across patterned RGBA, explicit opaque/transparent cases, all 256 alpha values on black, repeated single-pixel calls, and a clamped subarray; also test an empty array.
- Maximum absolute JS/WASM output difference: 0.
- Generated WASM in the latest local run: 1,199 bytes, including rejection of incomplete pixels. This uses the mutable local JZ working tree, so it is not a pinned-version size guarantee.
- The JZ checkout contains local changes; this is evidence about that working tree, not a released-version guarantee. No JZ files were modified.

The review found two host-boundary limits: direct `Uint8ClampedArray` input is rejected by this JZ checkout, and compiled type checks do not reliably distinguish marshaled array kinds. The probe validates types in JavaScript and passes clamped inputs through a zero-copy `Uint8Array` view preserving byte offset/length. It copies decoded output before resetting WASM memory in `finally`, including when compiled code throws. Numeric kernel parity does not imply interchangeable host input semantics.

[Seven regression tests](tests/gray.test.mjs) cover empty/single-pixel inputs; null/missing input; lengths 1–3 and 5–7; independent expected RGB/grayscale values; every alpha value; clamped subarray offsets/immutability; and A → A → B → empty → A with independent retained outputs. The integration probe additionally rejects arrays, strings, objects, float32/uint16 inputs before marshaling, calls a valid pixel after every malformed-length exception, and checks retained outputs after memory resets. These are probe tests, not recognition tests.

This is compilation and numerical compatibility evidence. It is not a speed benchmark, full normalizer, font-recognition experiment, or total shipped size. The interop layer, model, catalog, and surrounding code are excluded from the WASM byte count.

## What `fontr` contributes

The local `../fontr` experiment contains 217 collected Google Font files and useful style/topology work. Its inspected `src/mvp/train.py` fits style vectors and glyph topology for isolated glyphs; `src/mvp/style.py` defines 12 semantic/free axes. It does not establish a ready arbitrary-image font-family or parameter predictor. Glyph skeleton reconstruction is not required for family retrieval.

The pilot imports 20 local font files and records their actual embedded licenses and checksums. It explicitly instantiates variable fonts at weight 400: Montserrat defaults to 100, Nunito/Nunito Sans/Source Sans 3 to 200, and Merriweather to 300 in this collection. Leaving defaults untouched would confound the regular-family experiment. No `fontr` source or environment was modified. Its embedded notices do not substitute for recovering pinned upstream sources before distributing the larger corpus.

The 12-axis representation may be useful later for labeled weight/width/slant supervision. A learned retrieval vector is not automatically interpretable as those semantic parameters; that requires a separate supervised task and evaluation.

## First deliverable completed

The synthetic portion of [bench/protocol.md](bench/protocol.md) now runs: known-font crops → shared grayscale preparation → inspected fixtures → descriptor retrieval → one small learned baseline. [The report](bench/report.md) records an overall learned-model regression and the next experiments. Independent real screenshots remain outstanding. A generic screenshot detector, custom training engine, or compiler optimization is not a prerequisite for improving crop-level retrieval.


## gpu-* comparison for the ten-font classifier (2026-09-22)

The relevant comparison is the small gpu-* task models, not commercial font-identification services. Current upstream artifacts were checked again; counts below come from their own model cards, not measurements made on this machine.

| Package | Learned payload | Training and division of work |
| --- | --- | --- |
| [gpu-lexer](https://github.com/vercel-labs/gpu-lexer/blob/main/MODEL_CARD.md) | 41,321 deployed int6 weights; 27.64 KiB Brotli for the reported minified package | Offline PyTorch, Shiki teacher labels. Mechanical JavaScript splitting/features precede learned token roles. Millions of labeled parts; repository/package-disjoint evaluation. |
| [gpu-time](https://github.com/arikchakma/gpu-time/blob/main/MODEL_CARD.md) | 38,745 parameters; 22,501 bytes Brotli weights, 45,561 bytes Brotli package | Offline PyTorch, generated and harvested labeled text, quantized export. Ordinary TypeScript resolves calendar semantics; the network predicts roles/slots. |
| [gpu-query](https://github.com/safzanpirani/gpu-query) | 29,597 parameters; reported 40 KiB int6 weights; total package size not independently measured here | [PyTorch training](https://github.com/safzanpirani/gpu-query/blob/main/spike/train.py) on synthetic schema/query examples. Deterministic features, fuzzy field matching and a compiler surround the role model. |

Our 47,034-parameter classifier is about 14% above gpu-lexer's reachable weight count and 21% above gpu-time's parameter count. This is a rough capacity comparison: parameter definitions and tasks differ. Image convolutions touch many spatial positions, so similar parameter counts do not imply similar GPU work, latency or training cost. Our int8 JSON is also less compressed than their int6 payloads.

The useful shared pattern is a narrow task, trustworthy automatically generated labels, cheap input preparation, offline training, a separate held-out evaluation and a very small inference implementation. Their small deployed networks do not imply tiny training sets. JavaScript or JZ training is not necessary to ship a JavaScript package.

For gpu-font, tight crop preparation is the cheap deterministic stage. Identifying letters is an additional learned task, not an equivalent free parser. It may help control glyph content later, but first compare total accuracy and cost with the direct text-window classifier. Never OCR and re-render the words: that replaces the font evidence being measured.
