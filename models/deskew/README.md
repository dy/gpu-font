# Deskew candidate — 100 regular families

This is the second-seed deskew candidate selected on the fresh preparation development bank. It is preserved separately from the demo. See [the experiment report](../../bench/preparation.md).

Use `prepareLine(image, { deskew: true, sampler: 'windows' })` from the exact `src/line.mjs` version identified by `model.json` → `preparation.lineSha256`. The returned arrays are the inference inputs; source polygons map those arrays back to the selected image. A no-rotation decision preserves the existing sampler's tensors exactly.

`best.pt` contains the unfused training checkpoint; load its `state` into `Classifier(100, context=True, dilations=[1,1,2,2,1])`. `calibration.json` was fitted on development queries and unknown-validation families only. The raw classifier loader does not choose preprocessing on the caller's behalf.

The normalizer and artifact must be integrated together. Do not point the current demo's unchanged `prepareInput` call at this candidate. Short/degraded text and absent-family rejection still fail the release gates.
