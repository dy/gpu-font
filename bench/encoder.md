# Google Fonts shared-encoder experiment

The product requirement is a frozen encoder plus a caller-provided visual catalog. Adding a font must add reference vectors without changing the encoder. This experiment trains fresh weights on the pinned Google Fonts corpus; it does not continue the classifier that already saw all 2,004 families.

## Frozen protocol

- [Family split](encoder-split.json): 1,403 training, 301 development and 300 final families. Related names, shared upstream repositories, identical binaries and observed identical render banks are connected before assignment. There are 1,099 groups. These are conservative leakage controls, not a complete historical genealogy.
- Script coverage is 161 in training, 31 in development and 26 in final families. Rare scripts concentrated in one lineage cannot appear in all splits. Noto's 228 families remain together in training. Report script slices; do not claim held-out recognition for all 162 corpus scripts.
- All convolution and projection weights start randomly, seed 20260922. No historical checkpoint, classification head or previous corpus gradient is reused.
- Identical 32/64/96/128/128 convolution channels and a learned 128-dimensional projection in both arms. Embeddings are L2-normalized for cosine retrieval. The baseline has an additional training-only classification head; it is discarded before catalog evaluation.
- Per covered alphabet: 32 fresh training strings, eight reference strings, eight development-query strings and eight final-query strings. Lengths are 2 (3 for small alphabets), 4, 8 and 12. All historical strings are excluded, and roles are disjoint after case folding. These are synthetic strings, not a natural-language screenshot benchmark.
- Pillow/RAQM and Chromium render the same selected normal face/axes. The browser receives a verified static font instance so optical size and other variable defaults cannot change the selected face. Originals stay unchanged. Both renderers provide reference and query images; training includes clean and augmented examples, queries include clean/rotated/small conditions.
- Preparation uses the existing grayscale, tight, aspect-preserving deskew/windows code. Preserve all three windows where emitted, average normalized window vectors per source, then normalize. No OCR or handcrafted descriptor is used.
- Each episode selects up to 48 training families and two different text strings per family, with opposite renderers. Prefer the same script and shared text across competing fonts. Sparse scripts fill the remaining candidates with other families. Randomize candidate order. Hard negatives come only from the current training-family reference vectors, refreshed at checkpoint selection.
- Episodic objective: symmetric cross-entropy over the two views' cosine matrix, temperature 0.1. The comparison arm uses global training-family cross-entropy with 0.02 label smoothing on the same episode construction. Observed per-script render aliases share target mass instead of receiving contradictory hard labels.
- First comparison: **20,000 optimizer updates per arm**, AdamW, learning rate 0.001 down to 0.00003, weight decay 0.0001, gradient norm cap 10. Both arms use the same seed and fresh initialization. This is a bounded first comparison, not a claim that 20,000 updates are optimal.
- Checkpoints at step 0 and every 4,000 updates. Select by the harmonic mean of macro family top-5 for development families and 80 fixed training-family controls, on clean 8+ character queries in both renderers. Search all 1,704 non-final families at every checkpoint. No final-family pixels are used during optimization or selection.
- Full development evaluation compares one versus four references per face with the same underlying reference images. The four-reference case groups by source text length. Measure strict identity top-1/top-5 by family, split, renderer, condition, script and length; quantify weight/vector int8 loss separately.
- Freeze the selected encoder and reference recipe before preparing/indexing final families. Evaluate all 2,004 candidates together; confirm adding/removing/reordering entries preserves encoder bytes. Catalogs bind both encoder and preparation hashes. A similarity score is not calibrated certainty.

Quality remains subject to the gates in [todo.md](../todo.md). A useful result requires unseen-family retrieval, independent screenshots and rejection measurements; passing the size limit alone is insufficient. Genuine weight/style variation and a training-only glyph objective remain subsequent measured comparisons.

## Reproduction

```sh
node scripts/python.mjs -m train.encoder_data split
node scripts/python.mjs -m train.encoder_data plan
node scripts/python.mjs -m train.encoder_data faces --workers 6
node scripts/python.mjs -m train.encoder_data render --workers 6
node scripts/encoder-browser.mjs development
node scripts/python.mjs -m train.encoder_data pack
node scripts/python.mjs -m train.encoder episodic --steps 20000
node scripts/python.mjs -m train.encoder classification --steps 20000
npm test
```

The two renderers can run concurrently after normal face instances are ready. Training requires MPS access on the local Mac. The `.data/encoder/` archive is separate from the deployed demo. Each arm writes `progress.json`, an exact optimizer/scheduler/random-state `last.pt` for `--resume`, a selected `best.pt`, and `encoder.json`. Resume rejects changed training code, data, split, method or total step budget. Reports are written to `bench/encoder-episodic.json` and `bench/encoder-classification.json` only after the arm finishes.

This file records the protocol before optimization. Results and final catalog commands will be added after measurement; no recognition gain is claimed yet.
