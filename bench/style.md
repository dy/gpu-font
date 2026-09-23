# Style encoder

The requirements: the original font in the top five from 5–10 characters; any catalog JSON works with the same encoder; uppercase matches lowercase, one script matches another where the font has both, hand drawings find style matches; matches carry their weight and style. The shared encoder before this work ranked the Roboto demo sample 29th, with Grandstander and Azeret Mono in its top five, Roboto's 937th and 770th nearest designs by their letters.

## Ground truth for style

`train/style_teacher.py` renders the same glyphs in every face and compares them glyph by glyph: 62 Latin letters and digits, 64 Cyrillic, 48 Greek, 200 common Hanzi, and the most widely covered characters of 18 more scripts with at least ten families. Each script scales by a reference glyph (Latin `H`, Cyrillic `Н`, Hanzi `国`) and shares a baseline, so proportions stay; the distance is the mean normalized squared difference of lightly blurred glyphs both faces contain. Faces are every static file and each hundred along variable weight axes: 8,132 faces of 2,004 families.

Renderer noise sets the resolution. Re-rendering one face with a third-pixel shift and its hinting removed moves it by a median 0.0088 (95th percentile 0.0148, 300 faces). Faces closer than the median are **twins**, pairwise and never chained: 483 of 1,973 Latin-capable families have one. Lora and Parastoo are 0.0012 apart; Roboto, Vazirmatn and Heebo 0.0035. Weights 100 apart in one file differ by a median 0.0157, so this resolution cannot separate adjacent weights; weight is scored with a ±100 tolerance.

The distance recovers near-duplicates and derivatives exactly. Beyond the nearest twenty designs it stops tracking perceived style (Bodoni Moda ranks 171st for Playfair Display), so broad style classes come from Google's own taxonomy: METADATA categories and the weighted `tags/all/families.csv` classes at the pinned commit.

## Training views

`train/style_data.py` generates every view from a face, a script and a seed; nothing is stored and the text never repeats. Strings of 1–24 characters use each face's common characters (frequency-weighted Latin, 3,000 common Hanzi, 2,000 Hangul syllables, the face's own letters for rare scripts) in lower, upper, title and mixed case, sizes 14–72 px, occasional letter-spacing, and screenshot damage: dark mode, low contrast, rotation, rescaling, blur, JPEG, noise and clipped edges. Pixels pass through the browser's own `prepareLine` in a persistent Node child (`scripts/style-prepare.mjs`), byte-identical to the batch preparation. Twelve workers produce about 1,400 views per second. Stored Chromium renders of the training families (116,608) add the browser rasterizer to every batch.

Sketch views (the fine-tune) thin each rendering to one-pixel centre lines (Zhang–Suen, every stroke kept), redraw them with a round pen proportional to the text size, then add a smooth wobble and a slight tilt.

## Objective

Each batch holds 32 faces and, for each, one of its sixteen nearest designs, two views apiece. The loss has three parts over proxies for every training face:

- **identity**: margin softmax (scale 30, margin 0.15); a face's twins share its target;
- **geometry**: each view's similarity profile over all faces (temperature 0.05) matches the letter-by-letter profile of its face (temperature 0.02), within the script the distance covers;
- **heads**: weight, italic, script, Google category and fine class, read linearly from the normalized embedding and shipped with it.

## Frozen benchmark

`train/style_bench.py` froze the evaluation before any style result: 9,409 Chromium canvas renders of static instances (so optical sizing cannot change a face), each random 5–10 characters: 300 families seen in training, the 301 development families that select checkpoints, and the 300 final families held out as a swapped catalog. Default, bold and italic faces; lower, upper, title and mixed case; one other-script query per face where the family has one; 1,426 degraded (75% size, JPEG 70) and 1,405 dark-mode copies; and a Hanzi slice of 406 queries. A separate synthetic slice of 2,214 sketches applies the training sketch transform, with fixed seeds, to clean Latin development and test renders at 32 px and above: it bounds real drawings from above.

Every query searches per-face references of all families: lowercase and uppercase Latin lines, plus two lines per other script, averaged per face. A result counts when the true family, or a twin of it, is in the top five.

## Results

`train/style.py` warm-starts from earlier checkpoints and trains 1,403 families (5,827 faces). The development families select checkpoints and the 300 test families never enter training or selection. The deployed model is the 1.36M-parameter encoder (the runtime's existing large architecture) after 30,000 steps, 2.5 hours on an M4 Max. The same objective on the 350K-parameter encoder reached 62.8% development top-five after 10,000 steps, where the large one had 67.8%: with unlimited text, capacity pays, unlike the earlier capacity probe on 32 fixed strings per alphabet.

Catalog references rendered by Chromium match browser crops far better than FreeType renders of the same lines. Chosen on development families only ([comparison](style-references.json)):

| Development families | FreeType references | Chromium references | Both averaged |
|---|---:|---:|---:|
| Top-5 (twins count) | 75.0% | **83.5%** | 82.5% |
| Top-1 | 49.5% | **62.1%** | 60.9% |
| Weight error of the matched face | 21 | **9** | 12 |
| Signed weight bias | +16 | **−2** | +5 |

Final report on the frozen benchmark, every model against the same Chromium per-face references of all 2,004 families ([report](style-quality.json)):

| | Previous encoder | Interim (case-diverse) | Style, 350K | **Style, 1.36M (deployed)** |
|---|---:|---:|---:|---:|
| Held-out families, top-5 | 60.5% | 65.7% | 72.0% | **84.3%** |
| Held-out families, top-1 | 37.3% | 42.3% | 49.5% | **62.5%** |
| Held-out, 8–10 characters, top-5 | | | | **87.8%** |
| Held-out, 5–7 characters, top-5 | | | | **81.2%** |
| Held-out, clean Latin, top-5 | 64.9% | 71.2% | 75.2% | **87.9%** |
| All roles, uppercase / lowercase, top-5 | 60.7% / 54.3% | 65.6% / 63.0% | 73.5% / 70.7% | **85.6% / 83.8%** |
| All roles, degraded, top-5 | 41.5% | 49.1% | 59.1% | **75.0%** |
| All roles, Hanzi, top-5 | 9.0% | 8.5% | 34.3% | **49.2%** |
| All roles, other scripts, top-5 | 34.6% | 35.0% | 51.8% | **64.0%** |
| Top-5 answers in the true Google category | 72.5% | 73.8% | 75.6% | **78.3%** |
| Top-5 answers that can draw the query's script | 94.2% | 94.1% | 97.5% | **98.6%** |
| Weight error of the matched face | 9 | 10 | 10 | **8** |

Strict top-five (twins not credited) is 84.0% on held-out families; seen families contain more near-duplicates (strict 69.2%, twins counted 82.8%). Dark-mode copies score exactly as their light originals: the preparation inverts polarity to identical pixels. The eight reported demo samples all rank in the top five, seven first; Roboto ranks first, followed by its Latin twins Heebo and Vazirmatn, Gothic A1 and Atkinson Hyperlegible Next.

Typed verdicts, from the heads: italic 97.5%, script 97%. The weight head is weak (82 on the 100–900 scale), so weight comes from the matched face instead (8).

## By requirement

`train.style breakdown` splits the held-out test families by the requirements ([report](style-breakdown.json)), with the same script-aware search and twin credit.

Decorative styles are the easy ones: brush 96.7% top-five (91 queries), Didone 93.4%, techno 94.4%, formal script 92.5%, display 90.4%, handwriting 88.6%. Plain text faces are the hard ones, because many of them look alike: Garalde old-style serifs 74.5%, humanist sans 76.3%, modern serifs 79.5%, sans serif overall 79.4%. So a collected catalog of brush or script faces needs coverage, not training.

Length matters as expected: 81.2% at 5–7 characters, 87.7% at 8–10. Scripts do not: Latin 85.6%, other scripts 69.8%, Chinese 48.1% (only 52 queries).

The shipped catalog holds every case and script of a face, which hides how little style crosses between letterforms. With one kind of reference removed for every family:

| Held-out query | References kept | Top-5 | With every reference |
|---|---|---:|---:|
| Capitals | lowercase only | 46.9% | 85.0% |
| Lowercase | capitals only | 52.3% | 84.1% |
| Other scripts | Latin only | 42.0% | 69.8% |
| Chinese | Latin only | 19.2% | 48.1% |

A catalog built from one case or from Latin alone, like most collected previews, finds capitals, other scripts and Chinese far less often. Style transfer across case and script is the next thing to train, and the benchmark to train it against now exists.

## Catalog

The Google Fonts catalog lists every face: 8,132 entries with 23,234 Chromium references (lowercase, uppercase and each script), a style name (Thin to Black, Italic), script coverage, and, on default faces, per-script twins: families whose letters in that script are closer than 95% of one face's re-renders (0.0148). Encoder (1,980,636 bytes, with heads), catalog (6,404,833) and required modules total 8,423,440 bytes, 4,235,749 with Brotli ([runtime check](style-runtime.json)); CPU and Metal agree with PyTorch within 3.8e-6, and a warm 128×48 window takes 1.3 ms on Metal. Chromium parses and decodes the catalog in 44 ms (39 ms warm) and ranks it in 6 ms. Search skips fonts that cannot draw the script the heads detect; a catalog with none of them ranks every font by style instead.

The page shows one row per design. Families that are twins in the crop's script fold into one row, named by the member whose name begins the others (IBM Plex Sans KR, Arabic and the Thai-derived Anuphan under IBM Plex Sans), scored by its best member and listing the rest; the saved result keeps every family. Twins are judged per script because shared Latin letters say nothing about the rest: Noto Sans Arabic and Noto Kufi Arabic are Latin twins but different Arabic designs. The display threshold is looser than the benchmark's median twins on purpose: a fold hides nothing, since folded names stay listed, while accuracy credits only the stricter pairs.

## Limits and negative results

- **Hand drawings do not work.** On the synthetic sketch slice the deployed model puts the drawn font in the top five 5% of the time, and 39% of its answers share the drawing's category (chance is about 28%). Two fine-tunes did not help: training sketch views on identity (8% of views) left development accuracy 1.7 points lower and sketch results unchanged; training them on a broad style target (25%) cost 4.8 points and lowered sketch category agreement to 33%. Both runs were stopped; the second is kept in `.data/style/rejected-sketch-style-target`, the first was overwritten. A real drawing set, and a category-level target, come first.
- **Other rasterizers are unmeasured.** Queries and references share Chromium on macOS. Safari, Firefox, Windows ClearType and phone screenshots may differ; the averaged catalog is one point lower here and hedges that risk.
- **Synthetic queries.** No real screenshots, photographed text, colored backgrounds or letter-spacing beyond training augmentation are measured.
- **Gates.** Held-out families at 8–10 characters: 87.8% top-five, above the 80% target. Seen families: 87.3% at 8–10 characters, below the 95% target. Other scripts (64%) and Hanzi (49%) stay well below Latin instead of within five points; 98.6% of answers can draw the script, not all; the italic head is 97.5% (the matched face's posture is right 99.9%). The two letter-distance style targets are not met, and the metrics themselves are unreliable beyond the nearest twenty designs.
- **Filtering by the predicted posture was rejected**: restricting candidates to upright or italic faces lowered development top-five from 61.7% to 59.7%, because 3% of posture predictions are wrong.

## Reproduce

```sh
node scripts/python.mjs -m train.style_teacher                  # faces, letter-by-letter distances, twins
node scripts/python.mjs -m train.style_bench plan               # frozen benchmark
node scripts/python.mjs -m train.style_bench instances --workers 6
node scripts/style-bench-browser.mjs
node scripts/python.mjs -m train.style_bench pack
node scripts/python.mjs -m train.style_bench sketches
node scripts/python.mjs -m train.style_catalog                  # FreeType references
node scripts/python.mjs -m train.style_bench instances --references --workers 6
node scripts/style-references-browser.mjs                       # Chromium references
node scripts/python.mjs -m train.style train --run evaluation-large --large --lr 3e-4 --warm .data/encoder/large-refine/best.pt --steps 30000 --seed 20260923
node scripts/python.mjs -m train.style compare --run evaluation-large
node scripts/python.mjs -m train.style final --run evaluation-large,evaluation-small --source browser
node scripts/python.mjs -m train.style export --run evaluation-large
node scripts/python.mjs -m train.style catalog --run evaluation-large --source browser
node scripts/python.mjs -m train.style deploy --run evaluation-large
node scripts/python.mjs -m train.style breakdown --run evaluation-large    # by length, style, case and script
node scripts/python.mjs -m scripts.preview_swap                             # catalog swap on collected previews
node checks/encoder.mjs models/encoder/encoder.json models/encoder/google-fonts.json bench/style-runtime.json
```

Preview catalogs are recompiled with `scripts.preview_catalog` after any encoder change; the compiler indexes faces whose captures are complete and reports the rest, so a live collection no longer stops the build.
