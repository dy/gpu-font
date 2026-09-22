# Google Fonts corpus candidate

2,004 eligible text families from Google Fonts commit `e44c4b011a820c2cbe2fd2cfa8052037d7edb571`. The complete source inventory contains 2,055 family directories; [the manifest](../../bench/corpus.json) records all 51 exclusions. Each family uses its available normal face/axis settings, not every weight or style.

- `model.json`: int8 inference weights, ordered family labels, catalog, source revision, preparation pins and observed rendering ambiguities; 1,475,833 bytes.
- `best.pt`: selected unfused checkpoint. Load `state` into `Classifier(2004, context=True, wide=True, dilations=[1,1,2,2,1])`.
- `initial.pt`: selected initial-stage checkpoint, retained to reproduce the corrected 20,000-update continuation.

Use `prepareLine(image, { deskew: true, sampler: 'windows' })` with the exact source versions pinned in `model.json`. Pass the returned windows to `createNetworkGPU(readNetwork(artifact))`, then aggregate using `rankWindows`. The model loader preserves the preparation method but does not preprocess images itself. Display the actual window pixels and their source polygons when inspecting input.

`catalog[].trainingFace` is source metadata, **not inferred weight/style**. Scores are closed-set classifier outputs; the old 100-family confidence calibration does not apply. There is no calibrated unknown-font rejection for this candidate.

Held-out Chromium results are **28.4% exact family / 48.7% top-five**, across short, rotated and reduced-resolution synthetic crops. This artifact establishes broad coverage and compact browser execution; it is not a reliable arbitrary-screenshot recognizer. The deployed website remains separately owned and requires coordinated preparation/catalog integration. See [results, limitations and reproduction](../../bench/corpus.md).
