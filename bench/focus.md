# Preserve letter detail before expanding training

The controlled experiment supports training on local text windows. At the same model size and training budget, clean Chromium Inter recall increased from **28.1% to 59.4%**, while false positives fell from **9.8% to 8.2%**. Neither model passes the proposed 90% recall / 5% false-positive gate. This is useful evidence about input policy, not a production-quality font recognizer.

## What the preview shows

The demo's model-input preview displays the actual 128 × 32 grayscale tensor, enlarged for visibility. Its rectangle remains fixed because the current model takes that shape. The selected image is contrast-normalized, trimmed, uniformly scaled and padded inside it; letters are not independently stretched. The browser check compares every displayed pixel with the exported inference tensor after import, cropping and image replacement.

The fixed shape is valid. Fitting an arbitrarily long line into it is the problem: median prepared training ink height was **15.7 pixels** for the whole-line policy and **28 pixels** for the local-window policy. A fixed rectangle can contain much more useful information without increasing network size.

The whole-line and local-window tensors are drawn by `scripts/focus_preview.py` into `bench/focus-inputs.png`, kept locally (images are not committed).

The figure uses checksummed raw source pixels and prepared tensors, with matching training-report hashes. It does not reconstruct the input from recognized characters.

## Controlled comparison

Run after importing the pilot fonts and installing the existing Python/Playwright dependencies:

```sh
npm run train:focus
node scripts/python.mjs -m scripts.focus_preview
```

`node scripts/focus.mjs data` regenerates fixtures only; `node scripts/focus.mjs train` reuses validated fixtures. Configuration: [focus.json](focus.json). Results: [whole line](focus-line.json), [local window](focus-patch.json). Generated data and exported experimental models stay in `.data/focus/`; the demo weights are unchanged.

- **Task:** Inter regular versus 13 other training families. Six further families are query-only negatives. This is binary verification, not 20-family retrieval or text detection.
- **Text:** 512 unique training strings, 64 calibration strings, 64 development strings. Each consists of one to four random 3–8-letter words, with lowercase/titlecase/uppercase variations. Splits are case-insensitively disjoint before rendering. Text, size and polarity plans match across positive and negative families. These are randomized letter sequences, not a natural-language corpus.
- **Data per policy:** 7,168 clean training crops; 896 clean calibration crops; 8,960 development crops per renderer (20 families × 64 strings × seven conditions). Development uses Pillow and Chromium. Distorted Chromium examples start from Canvas rendering and are subsequently transformed in Pillow.
- **Only the input policy differs:** whole line versus one centered window at most 124/28 times the ink height. Both then pass through the same JavaScript normalizer at 128 × 32 with two-pixel padding. A narrow source retains all its ink. The window may cut letters and does not know their identities. Window selection happens before the stress-test distortion; this is not an automatic detector/deskewer evaluation.
- **Training:** same 25,186-parameter CNN and binary head, seed 31415, 2,400 Adam steps, 24 matched positive/negative pairs per batch, learning rate 0.001, four CPU threads. No pretraining or larger input. Approximately **42 seconds per model**, excluding rendering, preparation and evaluation. Model export/reload logits match exactly.
- **Training-data storage:** the prepared fixture cache is 812 MiB; selecting one policy requires 406 MiB of input storage in RAM, excluding model/optimizer/intermediate tensors. The trainer shares that selected array with PyTorch and avoids a redundant second 406 MiB copy. These offline fixtures are not part of the browser download.
- **Threshold:** independently calibrated per model on the 64 calibration strings at ≤5% false positives. Each calibration contains 64 Inter and 832 negative examples; both thresholds allow 41/832 false positives. Development thresholds are not retuned.

## Results

Each entry is **Inter recall / other-font false-positive rate**. Each development row has 64 positives and 1,216 negatives. Rows share underlying text and are correlated.

| Condition | Whole line | Local window |
| --- | ---: | ---: |
| Clean Pillow | 23.4% / 5.8% | 75.0% / 6.2% |
| Clean Chromium | 28.1% / 9.8% | 59.4% / 8.2% |
| Half-size Chromium | 28.1% / 7.3% | 43.8% / 6.3% |
| Short Chromium crop | 3.1% / 1.2% | 45.3% / 9.7% |
| Chromium, −8° | 1.6% / 0.8% | 4.7% / 0.4% |
| Chromium, +8° | 0.0% / 0.1% | 4.7% / 0.7% |
| Blurred Chromium | 29.7% / 9.3% | 62.5% / 9.5% |
| JPEG Chromium | 28.1% / 8.9% | 71.9% / 8.6% |

Patch training recall is 99.8%, calibration recall 75.0%, and clean browser recall 59.4%. This is substantial overfitting/transfer loss despite learning the training task. The clean-only model's rotation failure is expected and now measured separately. Higher recall in another row does not by itself mean a better model: compare false positives too.

These are one-seed development results on synthetic inputs. The new experiment changes corpus and training schedule relative to [the earlier experiment](robustness.md), so its percentages cannot establish a historical accuracy improvement. Only the two arms here form a controlled comparison. No real screenshot, natural-word, arbitrary-background, mixed-font, italic, font-axis, unknown-family rejection or final-test claim is established. The six query-only families are negatives for Inter, not demonstrated recognizable new families.

Regression coverage in [test_focus.py](../tests/test_focus.py) checks crop geometry at zero, default and maximum valid padding using exact source pixels, along with blank/minimal inputs, malformed geometry, disjoint text pools, tensor boundaries/checksums and preview/report freshness. Both renderer paths derive the window aspect from the configured padding; the recorded default remains two pixels. A full regeneration/retraining after this fix reproduced both model hashes, all numerical metrics, source images and prepared tensors exactly. Training took 36.4/37.8 seconds on the repeat versus approximately 42 seconds originally; this timing variation is not an isolated speed benchmark.

## Next training budget

1. Keep 128 × 32 local windows. Generate fresh window positions and clean/short/scale views during training rather than repeating a fixed raster for every string. Include real words as well as random character sequences, preserving underlying-text splits. The train/development gap is the next target; merely training longer on these same 512 images per family is not the priority.
2. Add a small, separately split Chromium training/calibration corpus to measure renderer transfer. Preserve a new browser evaluation corpus and verified real screenshots; the current development set has already informed this decision.
3. Run one matched-budget comparison: clean windows versus a curriculum retaining clean samples while introducing crop/scale, then mild rotation, then blur/JPEG. Report each condition and calibration false positives separately. Budget a few thousand steps before deciding whether a larger reference CNN is justified.
4. Test at most three overlapping windows and average predictions when one crop contains enough text. Recalibrate at image level and count all three passes in latency. Do not treat correlated windows as independent confidence evidence or merge different fonts.
5. Require the one-font gate before adding a second target or replacing the demo model. If a larger reference succeeds while the tiny model fails, compress the successful model afterward.

## Are letters or OCR needed?

There are three distinct operations: finding a text region, reading its characters, and identifying its font. Our current crop supplies the region; local windows preserve font evidence without reading the text. CNN patch classification with aggregation is an established approach ([Tensmeyer et al., 2017](https://arxiv.org/abs/1708.03669)). This experiment supports trying that inexpensive route first.

Character-level font identification is also possible ([Font-ProtoNet, 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w34/Goel_Font-ProtoNet_Prototypical_Network-Based_Font_Identification_of_Document_Images_in_Low_CVPRW_2020_paper.html)). But letter segmentation adds its own failures—joined glyphs, detached dots and accents, clipped letters—and a lone `I` or `l` often lacks distinguishing evidence. Recognizing letters can help align comparisons, but OCR should remain an optional experiment if patch training stalls. It must preserve the original pixels rather than replace them with re-rendered text.

For a full screenshot, an existing text-region detector can eventually propose word/line boxes. Training that detector from scratch would be a separate task and is not necessary to improve supplied-crop recognition now.
