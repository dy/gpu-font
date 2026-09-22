# First experiment contract

Status: the synthetic pilot and first bounded training run are complete; see [results](report.md) and [reproduction commands](../README.md). The encoder does not beat the descriptor overall. Independent real screenshots and preparation-policy comparisons remain open. Release accuracy/size targets remain in [todo.md](../todo.md).

## Input and output

The first supported input is a caller-selected crop containing one horizontal line of upright, regular Latin text in one family. Start with roughly 6–30 visible characters, 12–48 CSS px, device pixel ratios 1 and 2, and a substantially uniform background. Include tiny, short, blurred, and low-contrast examples as stress cases whose failures are reported separately.

The primary output is ranked canonical family IDs, not exact font files, weights, or variable-axis coordinates. Keep aliases and source versions in metadata. Return similarity scores with defined semantics; do not label them confidence probabilities.

Automatic region detection is a separate optional stage. Its eventual output is a list of regions retaining original-image coordinates. Different fonts on a page receive separate recognition queries. No OCR or automatic detector is required to start.

## Immediate implementation slice

1. Pin 20 permitted font families: include at least five visually similar pairs and several distinct serif, sans, mono, and display examples. Record source revision, file hash, license, style, and glyph coverage. These 20 families belong to later training/development pools, never to the final unseen-family test.
2. Render three different strings in every family, at 14 and 28 CSS px, pixel ratios 1 and 2, and light/dark polarity: 480 controlled crops. Verify loaded fonts and glyph coverage. Add at least 20 independently captured, source-verified development screenshots before evaluating screenshot transfer.
3. Keep the first string for prototype generation and the other two for development queries. Once training begins, generate a separate text pool; never train on the query crops. The tiny pilot has no untouched final-test claim.
4. Implement `prepare` on decoded RGBA pixels. Preserve the same numeric function for offline preparation and browser inference, with browser decoding and Canvas access outside the kernel.
5. Save original/normalized pairs and an inspectable contact sheet. Compare grayscale, binary thresholding, and resize choices. Inspect thin strokes, counters, dots, punctuation, spacing, and similar-family distinctions. A visually neat image is not an accuracy result.
6. Run a cheap descriptor/prototype retrieval baseline on different-text queries. Report family top-1/top-5, near-family errors, polarity consistency, and processing time. This exposes preprocessing and evaluation errors before model training.
7. Run a bounded pipeline smoke trainer after checking labels, crops, and evaluation. Then diagnose failures and expand to the planned 50-family smoke experiment; the fixed 500-family experiment follows only after its sources and splits are ready. The first 20-family training pilot used one seed and 400 steps, not a release-quality run.

Proposed internal boundary, not a published API:

```js
prepare({ width, height, data }, { rect })
// { width, height, pixels: Float32Array, rect }
```

`rect` is an explicit region in source coordinates and defaults to the whole supplied crop. Start with float values in [0, 1], white background and dark foreground. Do not require character identities or character segmentation. A detector can supply rectangles later without changing normalization.

Preparation requirements:

- Composite transparency against the declared background; handle dark-background inputs explicitly rather than assuming transparency means white ink.
- Derive grayscale while retaining intermediate intensities and antialiasing. Use conservative contrast/polarity handling; no aggressive sharpening, erosion, dilation, or per-character stretching by default.
- Preserve original glyph aspect ratios. Test fitting/padding into 128 × 32 against height-preserving windows for long lines; choose the policy from the pilot before freezing the model input contract.
- If a threshold is used to find foreground bounds, retain unthresholded grayscale inside those bounds. Preserve detached dots, diacritics, punctuation, ascenders, and descenders.
- Reject invalid dimensions, buffer-length mismatches, invalid rectangles, and empty input. Give blank/insufficient-contrast crops an explicit non-match path. Do not claim a contrast check can reliably detect all non-text images.
- Test deterministic output, input immutability, transparency, inverted polarity, edge-touching glyphs, empty crops, and resize/padding alignment. Compare all future compiled implementations against the JavaScript reference.

The existing `gray.mjs` is only a JZ compatibility probe. It does not satisfy these normalization requirements or handle arbitrary screenshots.

## Stack decision

Use plain JavaScript/typed arrays for preparation and Node/browser tooling for the initial fixtures. Pin an exact browser automation version when installing the renderer. The full experiment still needs a separate rendering/capture path to measure renderer transfer; one-browser pilot fixtures alone cannot establish it.

PyTorch 2.11.0 is the chosen pilot trainer after a local forward/backward/update and JSON export probe. Dependencies are in `requirements.txt`; Playwright is pinned in `package-lock.json`. The full 400-step pilot also passed finite/nonzero-gradient checks and export parity. No second trainer is being maintained.

Keep exported tensors, preprocessing version, and catalog metadata independent of the trainer. The browser package must not depend on the training framework. JZ compilation can be an optional build step for measured numeric bottlenecks; keep the source runnable as JavaScript.

## Evaluation and work limits

The later 500-family experiment retains the planned 350/50/100 training/development/final unseen-family split, grouped by related designs, plus separate absent-family rejection sets. All crops inspected in this pilot are development data. Preserve original/source-image grouping when creating augmentation variants.

Before a full training run, lock architecture sizes, seeds, training-step limit, and measured time/cost estimate in its configuration. The plan allows up to three architecture sizes and two seeds each; no cloud compute is required for this phase. The fixed first pilot took 9.7 CPU seconds for 400 steps; [its quality is measured](report.md). Compiler speedups remain unmeasured.

Phase 0 remains open until:

- [x] The initial font selection and source manifest exist. Local file hashes are pinned; missing upstream revisions are explicitly null.
- [ ] The preparation policy has been inspected on actual font pixels; remaining choices above are resolved.
- [x] One trainer passes the local gradient/export probe and its dependencies are pinned.
- [x] Exact rendering/training/evaluation commands exist; `npm test` covers JS preparation/retrieval and Python data/training/export invariants, beyond the grayscale probe.
- [ ] Named laptop/phone and browser versions replace the release-performance placeholders.
- [ ] Accuracy gates, split manifests, and a bounded training budget are frozen before the first full run.

The immediate next experiment is error diagnosis and a height-preserving-window comparison, followed by expansion to 50 verified families and real development screenshots. Automatic full-screenshot text detection can follow once crop-level recognition is useful.
