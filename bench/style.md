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

Its dependency pins have moved only when preparation output was verified identical on every stored render: once for `scripts/corpus-prepare.mjs` (input validation), then for `src/prepare.mjs`, `src/input.mjs` and `src/line.mjs` (crop edges, below). Each move is a `repins` entry in the manifest, with its evidence.

A model trained on every family cannot be read on held-out families, so the same planner builds a catalog benchmark (`--name catalog`, its own text seed) over all 2,004 families: a validation role of 6,852 queries (default face, Latin lower and title, a degraded copy, one native line) that selects checkpoints of whole-catalog runs, and a test role of 20,851 queries in the frozen recipe, read once per released model. `train.style final`, `breakdown` and `deploy` measure a run on the benchmark its training roles allow.

Every query searches per-face references of all families: lowercase and uppercase Latin lines, plus two lines per other script, averaged per face. A result counts when the true family, or a twin of it, is in the top five.

## Results

`train/style.py` warm-starts from earlier checkpoints and trains 1,403 families (5,827 faces). The development families select checkpoints and the 300 test families never enter training or selection. The first style model was the 1.36M-parameter encoder after 30,000 steps, 2.5 hours on an M4 Max. The same objective on the 350K-parameter encoder reached 62.8% development top-five after 10,000 steps, where the large one had 67.8%: with unlimited text, capacity pays, unlike the earlier capacity probe on 32 fixed strings per alphabet. The deployed model takes that further: 10,000 more steps, then a 1.5× wider encoder of 3.04M parameters for 20,000 ([second step](#second-step-longer-then-wider)).

Catalog references rendered by Chromium match browser crops far better than FreeType renders of the same lines. Chosen on development families only ([comparison](style-references.json)):

| Development families | FreeType references | Chromium references | Both averaged |
|---|---:|---:|---:|
| Top-5 (twins count) | 75.0% | **83.5%** | 82.5% |
| Top-1 | 49.5% | **62.1%** | 60.9% |
| Weight error of the matched face | 21 | **9** | 12 |
| Signed weight bias | +16 | **−2** | +5 |

Final report on the frozen benchmark, every model against the same Chromium per-face references of all 2,004 families ([report](style-quality.json)):

| | Previous encoder | Interim (case-diverse) | Style, 350K | Style, 1.36M | **Style, 3.04M (deployed)** |
|---|---:|---:|---:|---:|---:|
| Held-out families, top-5 | 60.4% | 65.7% | 72.0% | 84.3% | **87.3%** |
| Held-out families, top-1 | 37.3% | 42.3% | 49.5% | 62.5% | **68.6%** |
| Held-out, 8–10 characters, top-5 | | | | 87.7% | **89.4%** |
| Held-out, 5–7 characters, top-5 | | | | 81.2% | **85.3%** |
| Held-out, clean Latin, top-5 | 64.8% | 71.3% | 75.2% | 87.9% | **90.5%** |
| All roles, uppercase / lowercase, top-5 | 60.7% / 54.2% | 65.6% / 63.0% | 73.5% / 70.7% | 85.6% / 83.8% | **89.1% / 87.7%** |
| All roles, degraded, top-5 | 41.5% | 49.1% | 59.1% | 75.0% | **80.0%** |
| All roles, Hanzi, top-5 | 9.3% | 8.5% | 34.3% | 49.2% | **55.7%** |
| All roles, other scripts, top-5 | 34.6% | 35.2% | 51.8% | 64.0% | **70.4%** |
| Top-5 answers in the true Google category | 72.5% | 73.8% | 75.6% | 78.3% | **79.3%** |
| Top-5 answers that can draw the query's script | 94.2% | 94.1% | 97.5% | 98.6% | **99.1%** |
| Weight error of the matched face, held-out | 8 | 9 | 10 | 8 | **5** |

Strict top-five (twins not credited) is 86.9% on held-out families, top-one 66.5%; seen families contain more near-duplicates (strict 72.9%, twins counted 87.1%). Dark-mode copies score exactly as their light originals: the preparation inverts polarity to identical pixels. The eight reported demo samples all rank first.

Typed verdicts, from the heads: italic 97.6%, script 97.2%. The weight head is weak (82 on the 100–900 scale), so weight comes from the matched face instead (5).

## Second step: longer, then wider

The first model's errors were not near-duplicates. On development families only 5.8% of wrong first answers sit within 0.03 of the truth by letter distance (three times the renderer noise); the median wrong answer is 0.082 away. The model underfit: 41% of training views were right, families it trained on scored no better than unseen ones, and development accuracy was still rising when the schedule ended. Retrieval was already right: keeping reference lines apart, or scoring windows apart, gained nothing over averaging them.

Two continuations of 10,000 steps from that model, same learning rate, compared on development families (top-5, and top-1 for all queries):

| | All | Capitals among lowercase references | Other scripts among Latin references | Degraded | 5–7 characters |
|---|---:|---:|---:|---:|---:|
| Start | 83.5% / 62.1% | 45.4% | 40.9% | 74.4% | 79.5% |
| View-to-view loss, each face's two views in opposite cases | 84.0% / 62.2% | **53.2%** | 40.9% | 75.2% | 80.4% |
| Plain continuation | **84.5% / 63.3%** | 46.2% | **44.2%** | **76.2%** | **81.9%** |

The view loss taught case but cost everything else, so it stays off (`--pairs`). `widen` then duplicates each channel and divides its outgoing weights, so the 3.04M-parameter encoder starts with the exact function of the plain continuation (outputs within 3.8e-6), and trains 20,000 steps on a fresh data seed (4.5 hours on a busy M4 Max). Development top-five rose from 84.5% to 88.0% and top-one from 63.3% to 68.5%, still climbing at the end; training views went from 41% to 45% right. The held-out test, run once, is in the table above.

## Third step: references at three sizes

The whole-catalog model read once on the catalog test with 48 px references, 20,851 crops, 91.7% top-five (twins counted), splits by text size:

| Size | 16 px | 20 px | 24 px | 32 px | 40 px | 48 px | 64 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-5 | 82.3% | 89.8% | 91.9% | 94.6% | 94.2% | 95.0% | 94.2% |

From 32 px up the model is at 95%; the gap is small text, and a wider encoder sees the same eight-pixel x-height. The other side of the match is the reference: every catalog vector came from lines drawn at 48 px, and a 16 px crop upscaled to the model's window looks unlike a 48 px line shrunk to it. The same lines drawn at 24 and 16 px too, all averaged into the face's vector (`src/references.mjs` SIZES, shared by the page's My fonts and the repository's renderer), on catalog validation with the exported model:

| References | All | First | 16 px | 20 px | 24 px | 32 px | 64 px | Degraded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 px | 90.4% | 72.0% | 77.4% | 87.9% | 90.8% | 95.4% | 93.3% | 85.8% |
| 48 + 24 | 91.4% | 74.8% | 79.0% | 90.1% | 92.3% | 95.8% | 93.3% | 86.6% |
| **48 + 24 + 16** | **92.5%** | 74.3% | **83.7%** | 91.8% | 93.6% | 95.1% | 93.2% | 88.5% |
| 24 + 16 | 91.6% | 72.8% | 83.7% | 91.2% | 92.2% | 94.4% | 91.6% | 88.0% |

Two points of top-five with no training and no download cost: the catalog still holds one vector per face and case. Large sizes need the 48 px lines and small ones the 16 px lines; 32 px added (92.4%) or 32 and 20 px added (92.4%, first 75.3%) gain nothing more for a proportionally slower build. Building a catalog draws three times as many lines, in the page as in the repository.

## By requirement

`train.style breakdown` splits the held-out test families by the requirements ([report](style-breakdown.json)), with the same script-aware search and twin credit.

Decorative styles are the easy ones: brush 96.7% top-five (91 queries), Didone 95.1%, formal script 93.3%, techno 92.5%, handwriting 92.5%, display 92.4%. Plain text faces are the hard ones, because many of them look alike: humanist sans 81.1%, Garalde old-style serifs 81.6%, sans serif overall 82.6%, modern serifs 84.9%. So a collected catalog of brush or script faces needs coverage, not training.

Length matters as expected: 85.3% at 5–7 characters, 89.4% at 8–10. Scripts do not: Latin 88.4%, other scripts 75.1%, Chinese 57.7% (only 52 queries).

The shipped catalog holds every case and script of a face, which hides how little style crosses between letterforms. With one kind of reference removed for every family:

| Held-out query | References kept | Top-5 | With every reference |
|---|---|---:|---:|
| Capitals | lowercase only | 54.0% | 86.6% |
| Lowercase | capitals only | 56.5% | 87.3% |
| Other scripts | Latin only | 43.6% | 75.1% |
| Chinese | Latin only | 15.4% | 57.7% |

A catalog built from one case or from Latin alone, like most collected previews, finds capitals, other scripts and Chinese far less often. The wider encoder raised the case rows by 4–7 points, not the script rows. Style transfer across case and script is the next thing to train.

## Catalog

The Google Fonts catalog lists every face: 8,132 entries with 23,234 Chromium references (lowercase, uppercase and each script), a style name (Thin to Black, Italic), script coverage, and, on default faces, per-script twins: families whose letters in that script are closer than 95% of one face's re-renders (0.0148). Encoder (4,232,607 bytes, with heads), catalog (6,404,810) and required modules total 10,675,579 bytes, 5,348,002 with Brotli ([runtime check](style-runtime.json)); CPU and Metal agree with PyTorch within 3.8e-6, and a warm 128×48 window takes 2.6 ms on Metal. The wider encoder costs 2.25 MB more, 1.1 MB with Brotli. Chromium parses and decodes the catalog in 44 ms (39 ms warm) and ranks it in 6 ms. Search skips fonts that cannot draw the script the heads detect; a catalog with none of them ranks every font by style instead.

The page shows one row per family. Kinds of one family that are twins in the crop's script, one name holding the other's first word, fold into one row, named by the member whose name sits inside the others (IBM Plex Sans KR and Arabic under IBM Plex Sans), scored by its best member and listing the rest; a different font that borrows the letters, like Parastoo with Lora's Latin or Anuphan with IBM Plex Sans's, keeps its own row; the saved result keeps every family. Twins are judged per script because shared Latin letters say nothing about the rest: Noto Sans Arabic and Noto Kufi Arabic are Latin twins but different Arabic designs. The display threshold is looser than the benchmark's median twins on purpose: a fold hides nothing, since folded names stay listed, while accuracy credits only the stricter pairs.

## Crop edges

Two things at a crop's edge changed the answer; preparation (`src/prepare.mjs`, `src/input.mjs`, `src/line.mjs`) now removes both.

- **A border caught in the crop.** A UI line along an edge counted as text: it shrank the letters in the model's input and, when faint, set their contrast, so Josefin Sans Light read as ExtraLight. Preparation now ignores a line that lies along an edge (at most two pixels in), is one to three pixels thick, inks 90% of its length, has background between it and the text (other borders aside) and runs past the text at both ends. Drawn onto 941 benchmark renders with 4 and 8 pixel margins, a black, gray or faint line on any side, on the edge or one pixel in, now prepares exactly as the render without it every time, and a box one pixel inside the crop 99.1–100% of the time; before, never. Letters never qualify: on 74,224 renders cropped exactly to their ink, clean and inverted, the rule changes nothing. A Devanagari headline fails the background test, since its stems hang from it; a tight crop's l or h fails the run-past test.
- **A crop that touches the text.** Preparation keeps one pixel of background around the text, which a touching crop does not have, so its letters came out 4–10% larger than in any training view. Such a crop is now framed in two pixels of its own background first. On catalog validation cropped to the ink, the whole-catalog model (`catalog-30k`, not yet shipped) went from 88.48% to 90.00% twin top-five and from 4.4 to 3.6 mean weight error (CSS weight units), against 90.83% and 3.5 with margins. What remains is the degradation applied after cropping and the background estimate: the median of the crop's outer ring leaves the paper when text covers half of that ring, as in 5% of crops cut exactly to the ink.

Every stored benchmark render prepares byte-for-byte as before (bench, catalog and sketch packs), so each manifest's pins moved with the evidence in its `repins`.

## Quantization

`catalog-30k` on catalog validation, weights or catalog vectors rounded per row to b bits (twin top-five / twin top-one):

| Bits | Encoder weights | Catalog vectors |
|---|---:|---:|
| float | 90.78 / 72.02 | 90.78 / 72.02 |
| 8 (shipped) | 90.54 / 72.01 | 90.76 / 71.89 |
| 6 | 90.38 / 72.02 | 90.85 / 71.89 |
| 5 | 87.93 / 68.62 | |
| 4 | 76.43 / 53.24 | 90.78 / 71.89 |

Weights hold at 6 bits and break at 5. Catalog vectors lose nothing measurable at 4 bits, and they are 3.3 MB of the Google catalog's 3.5 MB gzipped download.

So the exporters now write 6-bit weights (`train.ten_model.export(bits=6)`, a `bits` field per layer) and 4-bit catalog vectors (`int4-base64`, 64 bytes a row), and both readers still take the 8-bit files. Reported accuracy now comes from the exported encoder, which the page runs, and the catalog is built with it: the float checkpoint scored 0.24 points above the shipped 8-bit model on this set.

## Limits and negative results

- **Hand drawings do not work.** On the synthetic sketch slice the deployed model puts the drawn font in the top five 5% of the time, and 39% of its answers share the drawing's category (chance is about 28%). Two fine-tunes did not help: training sketch views on identity (8% of views) left development accuracy 1.7 points lower and sketch results unchanged; training them on a broad style target (25%) cost 4.8 points and lowered sketch category agreement to 33%. Both runs were stopped; the second is kept in `.data/style/rejected-sketch-style-target`, the first was overwritten. A real drawing set, and a category-level target, come first.
- **Other rasterizers are unmeasured.** Queries and references share Chromium on macOS. Safari, Firefox, Windows ClearType and phone screenshots may differ; the averaged catalog is one point lower here and hedges that risk.
- **Synthetic queries.** No real screenshots, photographed text, colored backgrounds or letter-spacing beyond training augmentation are measured.
- **Gates.** Held-out families at 8–10 characters: 89.4% top-five, above the 80% target. Seen families: 91.7% at 8–10 characters, below the 95% target. Other scripts (70%) and Hanzi (56%) stay well below Latin instead of within five points; 99.1% of answers can draw the script, not all; the italic head is 97.6% (the matched face's posture is right 99.4%). The two letter-distance style targets are not met, and the metrics themselves are unreliable beyond the nearest twenty designs.
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
node scripts/python.mjs -m train.style train --run evaluation-large --architecture font-conv64-128-192-256-256-v4 --lr 3e-4 --warm .data/encoder/large-refine/best.pt --steps 30000 --seed 20260923
node scripts/python.mjs -m train.style compare --run evaluation-large
node scripts/python.mjs -m train.style train --run plain-10k --architecture font-conv64-128-192-256-256-v4 --lr 1.5e-4 --warm .data/style/evaluation-large/best.pt --steps 10000 --seed 20260923
node scripts/python.mjs -m train.style train --run pairs-10k --architecture font-conv64-128-192-256-256-v4 --lr 1.5e-4 --warm .data/style/evaluation-large/best.pt --steps 10000 --seed 20260923 --pairs 1   # rejected
node scripts/python.mjs -m train.style train --run wider-20k --architecture font-conv96-192-288-384-384-v5 --lr 2e-4 --warm .data/style/plain-10k/best.pt --steps 20000 --seed 20260924
node scripts/python.mjs -m train.style export --run wider-20k               # the exported encoder is what final, breakdown and the catalog measure
node scripts/python.mjs -m train.style catalog --run wider-20k --source browser
node scripts/python.mjs -m train.style final --run wider-20k,evaluation-large,evaluation-small --source browser
node scripts/python.mjs -m train.style breakdown --run wider-20k            # by length, style, case and script
node scripts/python.mjs -m train.style deploy --run wider-20k
node scripts/python.mjs -m scripts.preview_swap                             # catalog swap on collected previews
node scripts/python.mjs -m train.style_bench plan --name catalog            # catalog benchmark: every family, fresh text
node scripts/python.mjs -m train.style_bench instances --name catalog --workers 6
node scripts/python.mjs -m train.style_bench render --name catalog
node scripts/python.mjs -m train.style_bench pack --name catalog
node scripts/python.mjs -m train.style train --run wider-plain-10k --architecture font-conv96-192-288-384-384-v5 --lr 1.5e-4 --warm .data/style/wider-20k/best.pt --steps 10000 --seed 20260925
node scripts/python.mjs -m train.style train --run catalog-30k --architecture font-conv96-192-288-384-384-v5 --roles train,development,test --select catalog:validation --lr 1.5e-4 --warm .data/style/wider-plain-10k/best.pt --steps 30000
node scripts/python.mjs -m train.style export --run catalog-30k             # 6-bit weights
node scripts/python.mjs -m train.style catalog --run catalog-30k --source browser   # 4-bit vectors, references at three sizes
node scripts/python.mjs -m train.style final --run catalog-30k --source browser     # read on the catalog test: the run trained on every family
node scripts/python.mjs -m train.style breakdown --run catalog-30k
node scripts/python.mjs -m train.style deploy --run catalog-30k
node scripts/catalog-files.mjs                                              # every other catalog, with the shipped encoder
node checks/encoder.mjs models/encoder/encoder.json models/encoder/google-fonts.json bench/style-runtime.json
node scripts/demo-build.mjs
```

Preview catalogs are recompiled with `scripts.preview_catalog` after any encoder change; the compiler indexes faces whose captures are complete and reports the rest, so a live collection no longer stops the build.
