# Style encoder

The requirements: the original font in the top five from 5–10 characters; any catalog JSON works with the same encoder; uppercase matches lowercase, one script matches another where the font has both, hand drawings find style matches; matches carry their weight and style. The shared encoder before this work ranked the Roboto demo sample 29th, with Grandstander and Azeret Mono in its top five, Roboto's 937th and 770th nearest designs by their letters.

## Ground truth for style

`train/style_teacher.py` renders the same glyphs in every face and compares them glyph by glyph: 62 Latin letters and digits, 64 Cyrillic, 48 Greek, 200 common Hanzi, and the most widely covered characters of 18 more scripts with at least ten families. Each script scales by a reference glyph (Latin `H`, Cyrillic `Н`, Hanzi `国`) and shares a baseline, so proportions stay; the distance is the mean normalized squared difference of lightly blurred glyphs both faces contain. Faces are every static file and each hundred along variable weight axes: 8,132 faces of 2,004 families.

Renderer noise sets the resolution. Re-rendering one face with a third-pixel shift and its hinting removed moves it by a median 0.0088 (95th percentile 0.0148, 300 faces). Faces closer than the median are **twins**, pairwise and never chained: 483 of 1,973 Latin-capable families have one. Lora and Parastoo are 0.0012 apart; Roboto, Vazirmatn and Heebo 0.0035. Weights 100 apart in one file differ by a median 0.0157, so this resolution cannot separate adjacent weights; weight is scored with a ±100 tolerance.

Since the student's release the teacher takes every collected font file as well as Google's: `train/style_fonts.py` lists Google's families from bench/corpus.json and the open ones from bench/open-fonts.json with their letters, an open face names its store so the renderers find its file, and open faces come after Google's in faces.json so every stored benchmark index keeps its face. An open family's role is a fixed rule, not a drawn split: one in ten by the hash of its id is `held-out` (`open_role`), a role no training run names, so a model trained on the rest still has an open read (`train.style_open plan --held-out`), whatever is collected later. Open families have no Google taxonomy, and the category and tag losses already skip a family without labels. They brought 26 scripts the Google set lacked (48 in all), so a warm start carries a checkpoint's script and tag rows over by label and lets new labels start fresh (`adapt_heads`).

The distance recovers near-duplicates and derivatives exactly. Beyond the nearest twenty designs it stops tracking perceived style (Bodoni Moda ranks 171st for Playfair Display), so broad style classes come from Google's own taxonomy: METADATA categories and the weighted `tags/all/families.csv` classes at the pinned commit.

## Training views

`train/style_data.py` generates every view from a face, a script and a seed; nothing is stored and the text never repeats. Strings of 1–24 characters use each face's common characters (frequency-weighted Latin, 3,000 common Hanzi, 2,000 Hangul syllables, the face's own letters for rare scripts) in lower, upper, title and mixed case, sizes 14–72 px, occasional letter-spacing, and screenshot damage: dark mode, low contrast, rotation, rescaling, blur, JPEG, noise and clipped edges. Pixels pass through the browser's own `prepareLine` in a persistent Node child (`scripts/style-prepare.mjs`), byte-identical to the batch preparation. Twelve workers produce about 1,400 views per second. Stored Chromium renders of the training families (116,608) add the browser rasterizer to every batch.

Since the student's release, one flat render in seven wears a display effect (outline, drop shadow, extrusion), the forum pictures' failure mode, and the 2,980 WhatFontIs development photographs of Google families join the stored Chromium views as training views, each in the face of its photographed weight and holding a fixed eighth of each step's 32 stored views (pooled with 210,000 renders they would be seen four times in a run); the 3,795 final photographs stay the held-out read.

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

| | Previous encoder | Interim (case-diverse) | Style, 350K | Style, 1.36M | **Style, 3.04M (wider-20k)** |
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

The shipped model trains on every family (`catalog-30k`: 10,000 more steps of the wider encoder, then 30,000 on all 2,004 families with checkpoints selected on catalog validation), so it is read on the catalog test instead: 20,851 crops in the frozen recipe, on text never used in training or selection, against references at three sizes. The same test for the two earlier encoders, and the frozen benchmark's held-out figure of the model before it, for scale:

| Twins counted | Previous encoder, catalog test | Interim, catalog test | wider-20k, frozen held-out | **catalog-30k, catalog test (shipped, 6-bit)** |
|---|---:|---:|---:|---:|
| Top-5 | 61.1% | 67.2% | 87.3% | **92.8%** |
| Top-1 | 36.3% | 41.4% | 68.6% | **73.5%** |
| 8–10 characters, top-5 | | | 89.4% | **95.1%** |
| 5–7 characters, top-5 | | | 85.3% | **90.8%** |
| Clean / degraded / dark, top-5 | | | | **93.5% / 88.0% / 94.7%** |
| Title / capitals / lowercase, top-5 | | | | **95.5% / 94.7% / 92.8%** |
| Latin / other scripts / Hanzi, top-5 | | | 88.4% / 75.1% / 57.7% | **94.1% / 80.6% / 65.0%** |
| Weight error of the matched face | | | 5 | **6** |

Strict top-five is 84.1%, top-one 63.5%. The test was read twice for this model: once with 48 px references (91.7% / 71.4%, the size table in the third step) before the reference sizes were chosen on validation, then as shipped.

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

The whole-catalog model on the catalog test with 48 px references, 20,851 crops, 91.7% top-five (twins counted), splits by text size:

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

Two points of top-five with no training and no download cost: the catalog still holds one vector per face and case. On the catalog test the shipped model went from 91.7% to 92.8% top-five and 71.4% to 73.5% first, 16 px crops from 82.3% to 87.7%. Large sizes need the 48 px lines and small ones the 16 px lines; 32 px added (92.4%) or 32 and 20 px added (92.4%, first 75.3%) gain nothing more for a proportionally slower build. Building a catalog draws three times as many lines, in the page as in the repository.

## Fourth step: 64 numbers

The catalog's vectors are 3.3 MB of its 3.5 MB gzipped download, so halving the embedding halves the download. Font styles are not clusters: 95% of the reference variance lies in 39 of the 128 numbers, and the leading 64 principal directions of the unit reference-line embeddings (every face, case, script and size) carry 99.1% of it. `train.style project` folds those directions into the head, so the encoder puts out 64 numbers whose unit vector is the projected direction of the old embedding, and refits the typed heads on the projected lines (`--dimensions 64`; the page, the catalogs, the style links and the readers take the width from the model since). The fold alone costs a point, since the basis is fit on lines that are mostly Latin and the network never trained for 64 outputs; 5,000 steps of the usual training from the folded model (`catalog-30k-64r`, learning rate 5e-5, selected on catalog validation) give it back with interest. Catalog validation, exported models, references at three sizes:

| Exported model | Numbers | Top-5 | First | 16 px | Degraded | Other scripts | Photos, top-5 / first |
|---|---:|---:|---:|---:|---:|---:|---:|
| catalog-30k (shipped) | 128 | 92.47% | 74.31% | 83.5% | 88.4% | 83.7% | 54.9% / 32.5% |
| folded | 64 | 91.42% | 73.29% | 82.0% | 87.7% | 80.6% | 54.4% / 31.8% |
| **folded, then 5,000 steps** | **64** | **92.62%** | **75.06%** | 83.9% | 89.1% | 83.1% | 54.9% / 32.1% |

The 64-number model is the teacher of the separable student below and the fallback release if the student misses its gate.

## Fifth step: a separable student

The last two blocks hold 77% of the encoder's weights and see a 6×16 grid. Rebuilt with depthwise-separable convolutions past the first block (`font-sep96-192-288-384-384-v7`: a 3×3 filter per channel, then a 1×1 mix), the same widths hold 373k weights instead of 3.04M. Nothing of the dense model transfers into that shape (`widen` refuses it), so the student learns the 64-number teacher's embedding: at each step the teacher reads the clean render of a view and the student, reading the damaged or photographed one (40% photographs), follows that direction (cosine distance, weight 10) on top of the usual identity, geometry and head losses, from the teacher's heads and proxies. The control is the same run without the teacher: same architecture, 10,000 steps, seed and views. Catalog validation and the development photographs, twins counted:

| 10,000 steps from scratch | Top-5 | First | Photos, top-5 / first |
|---|---:|---:|---:|
| Direct, no teacher | 75.6% | 50.5% | 37.0% / 16.9% |
| **Distilled from catalog-30k-64r** | **86.8%** | **64.7%** | **56.7% / 31.5%** |
| The teacher | 92.6% | 75.1% | 54.9% / 32.1% |

The teacher is worth 11 points on validation and 20 on photographs at equal steps, and the student already passes its teacher on photographed text. It trailed the teacher by 5.8 points on validation after 10,000 steps against the teacher's 105,000, so it continued for 20,000 more from its best checkpoint at half the learning rate (`student-30k`), rising to 90.4% by step 17,500 and flat after. Then its weights were clustered to 4 bits: each layer's values, scaled per output row, k-means'd into 16 shared values (Deep Compression), the indices fixed and the codebooks, scales, biases and heads retrained for 3,000 steps with the teacher (`--cluster 4`); the file then holds each layer's codebook and every weight's index. Exported models, validation and development photographs:

| Exported model | Weights | Top-5 | First | 16 px | Degraded | Other scripts | Photos, top-5 / first | Encoder, gzipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher, 6-bit (catalog-30k-64r) | 3.04M | 92.6% | 75.1% | 83.9% | 89.1% | 83.1% | 54.9% / 32.2% | 2.15 MB |
| Student, 6-bit (student-30k) | 373k | 90.3% | 70.1% | 80.1% | 85.9% | 81.1% | 64.7% / 38.7% | 0.35 MB |
| Student, 4-bit clustered (student-30k-q4) | 373k | 89.2% | 68.0% | 78.6% | 84.6% | 79.5% | 63.0% / 37.2% | 0.27 MB |

The student misses the one-point gate on rendered text by 2.3 points and beats the teacher by ten on photographs, at a sixth of the download; clustering to 4 bits saves 80 KB more for 1.1 points, which is not worth it at this size, since the file is mostly the heads, scales and biases by then. The 6-bit student is the shipped model; its release read is under [By requirement](#by-requirement) and [Catalog](#catalog). A wider student (128 to 512 channels, about 0.7M weights) is the next single change for the render gap.


## Sixth step: every collected font, effects and photographs

`open-30k` continues the shipped student (its weights, proxies re-imprinted, the script head widened by label for the scripts the open files brought) on the enlarged data: Google's 8,132 faces and 8,623 faces of 4,112 open families (Debian, Font Library, Fontshare, Uncut, Velvetyne, Use & Modify and the smaller sources, one family in ten held out), the teacher rebuilt over all 16,755 faces, one flat render in seven wearing a display effect, and the 2,980 WhatFontIs development photographs as an eighth of each step's stored views. Same teacher (`catalog-30k-64r`), learning rate 5e-4, 30,000 steps, `--photo 0.4`, checkpoint selection on catalog validation as before:

| Step | Validation top-5 | First | Photographs top-5 | First |
|---|---:|---:|---:|---:|
| 0 | 90.4% | 70.3% | 65.5% | 39.9% |
| 2,500 | 88.9% | 67.9% | 67.8% | 42.7% |
| 5,000 | 89.5% | 68.8% | 67.3% | 40.6% |
| 10,000 | 90.3% | 70.6% | 70.1% | 44.8% |
| 15,000 | 90.5% | 71.4% | 70.4% | 44.8% |
| 20,000 | 91.3% | 71.8% | 73.5% | 49.5% |
| 25,000 | 91.5% | 72.3% | 74.4% | 50.8% |
| 30,000 | 91.7% | 72.5% | 75.0% | 50.6% |

The warm start dips at 2,500 while the new faces take their proxies, is back by 10,000, and then climbs past the student on both counts; photographs gain 9.5 points on top five and 10.7 on first place, renders 1.3 and 2.2. Its release read is under [By requirement](#by-requirement) and [Catalog](#catalog).

## Seventh step: GitHub's long tail

`open2-30k` continues `open-30k` on everything collected since: 7,902 open families (GitHub's long tail and Luuse after Debian, Font Library and the foundries), the teacher rebuilt over 26,974 faces of 9,906 families, 24,938 of them training (11,981 with a twin, one open family in ten held out), 217,664 stored Chromium views and the 2,980 development photographs. The open files draw 158 scripts, so the script head grows from 48 rows to 158, carried by label. Same teacher, learning rate 5e-4, 30,000 steps, `--photo 0.4`, selection on catalog validation. The run fits 32 GB: the teacher's distances are taken in blocks from a memory-mapped glyph file (peak 8 GB where the whole matrix needed 32), each step is two half-batches with the gradients summed, and the process tree peaked at 21 GB with 4.6 GB on the device.

| Step | Validation top-5 | First | Photographs top-5 | First |
|---|---:|---:|---:|---:|
| 0 | 91.7% | 72.5% | 74.9% | 50.7% |
| 2,500 | 90.5% | 70.2% | 72.0% | 47.3% |
| 5,000 | 90.7% | 71.2% | 72.9% | 48.7% |
| 10,000 | 91.4% | 72.9% | 71.4% | 45.0% |
| 15,000 | 92.0% | 73.5% | 72.6% | 47.4% |
| 20,000 | 92.0% | 73.9% | 75.3% | 51.3% |
| 25,000 | 92.6% | 74.4% | 75.7% | 52.1% |
| 30,000 | 92.8% | 74.3% | 75.9% | 51.9% |

The same dip and recovery as the sixth step, a point higher throughout. Its release read, the exported model on the catalog test: 92.12% top five and 73.14% first against 91.92% and 72.09% for `open-30k`, read again under the rebuilt teacher (91.92% and 72.08% in its own report: the rebuilt teacher moved nothing). The gain is where the new material is: other scripts 84.4% top five from 81.7%, Cyrillic 90.1% from 88.8%, Hanzi 65.1% from 63.9%, Latin level at 92.9%; the photographs the model trains on gain a point. Twins are read pairwise from the teacher's matrix at each step now, so the setup holds no list of them.

The model file is smaller with three times the scripts: 335 KB gzipped against 369 KB. The heads' numbers, meant at six decimals, were written at seventeen, the digits of a float32 read as a double, and so were every layer's scales and biases; the exporter now writes head weights at four decimals (a logit moves under 0.0004 on a unit embedding, a predicted weight under 0.12 units; on a million random embeddings the script choice changes three times) and scales and biases as the shortest text that reads back to the same float32, which loses nothing. The capture compiler writes each row's scale at six digits as the Google catalog does: 1.8 MB of text out of DaFont's file.

## Eighth step: photographed surfaces and web screens

`photo-30k` continues `open2-30k` for photographs. Half the streamed views are rendered as WhatFontIs-Bench composites its pictures, with the ranges its development labels give: ink on a crop of one of 383 CC0 Poly Haven textures (`scripts/textures.mjs`, none of the 43 the benchmark used) that shades the letters too, contrast from a quarter up, light letters on dark now and then, a cast shadow or glare, worn print or a display effect, uneven light, mild blur and sensor noise, and a camera a little off the normal, at times well off it (`train.style_data.surface`). Beside the renders, the captures of the web: 18,506 lines cut from the homepages of the development sites, nine sites in ten by the hash of their origin, each labelled by the font Chromium drew it with (`train.style_screens`), and the 2,980 development photographs, a quarter of each step's stored views. Same teacher, learning rate 5e-4, 30,000 steps, `--photo 0.5`, selection on catalog validation; the process tree peaked at 20 GB.

| Step | Validation top-5 | First | Photographs top-5 | First |
|---|---:|---:|---:|---:|
| 0 | 92.8% | 74.3% | 75.9% | 51.9% |
| 2,500 | 91.6% | 72.0% | 73.0% | 49.1% |
| 5,000 | 91.6% | 73.1% | 73.3% | 48.2% |
| 10,000 | 91.9% | 73.1% | 72.4% | 47.1% |
| 15,000 | 92.4% | 73.8% | 73.7% | 50.3% |
| 20,000 | 92.5% | 74.4% | 75.2% | 51.8% |
| 25,000 | 92.8% | 75.1% | 74.2% | 51.1% |
| 30,000 | 93.0% | 75.1% | 74.4% | 50.7% |

The harder views cost the trained-on photographs three points for half the run, made back by the end. The exported model reads 92.17% top five and 72.06% first on the catalog test against 92.13% and 73.12% for the seventh step: a point of first place on rendered text. On photographs it never trained on it gains that back and more: the 840 final WhatFontIs images of held-out Google families, read against the Google references (`train.style photos`), 73.2% top five and 51.4% first against 70.2% and 48.1%; on the page's read across every catalog, 8,915 images with twin credit, 46.1% against 45.2% top five and 62.5% against 60.9% top twenty, the 5,935 final images 41.0% against 40.4%; on the 151 forum crops 46.4% against 44.4% top five and 22.5% against 21.2% first. It ships for the pictures people bring, and the rendered first place is the price ([photos](photos.md)).

Two reads changed with it, for both models alike. A row of a family the teacher finds indistinguishable from the true one now counts as right in every read (`twins.json`, written by `train.style twins`; `rankOf` in scripts/photos.mjs): by the teacher's distance, 87 of GitHub's 413 wrong first answers on the development photographs were clones of the truth, Debian's 29 of 91, Font Library's 14 of 144, and the credit is worth eight points of top five on those photographs. And the file-built catalogs carry each default face's twins across every catalog, so the page folds a family's copies in GitHub or Debian under one row; Google's catalog keeps its own twins only, since the page folds a pair when either side names the other and the default download need carry no ids it cannot use.

## Other rasterizers, and 12 px

Queries and references both come from Chromium on macOS. `scripts/style-bench-engines.mjs` renders the catalog validation queries through Playwright's WebKit (Safari's engine) and Firefox, and through Chromium at 12 px, packed as benchmarks of their own; the 64-number model against the same Chromium references:

| Validation queries rendered by | Top-5 | First | Clean | Degraded | 16 px | 24 px | 48 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chromium 151 | 92.6% | 75.1% | 94.1% | 89.1% | 83.9% | 93.0% | 95.4% |
| WebKit 26.5 | 91.8% | 74.6% | 93.6% | 87.5% | 82.4% | 93.3% | 95.0% |
| Firefox 153 | 92.2% | 74.7% | 94.0% | 88.0% | 83.5% | 93.2% | 94.7% |
| Chromium 151, every query at 12 px | 56.7% | 36.1% | 72.6% | 18.3% | | | |

The 64-number student (`student-30k`), same queries and references:

| Validation queries rendered by | Top-5 | First | Clean | Degraded | 16 px | 24 px | 48 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chromium 151 | 90.3% | 70.1% | 92.1% | 85.9% | 80.1% | 92.5% | 94.2% |
| WebKit 26.5 | 89.8% | 69.6% | 91.6% | 85.4% | 78.1% | 92.5% | 93.6% |
| Firefox 153 | 90.5% | 70.0% | 92.0% | 86.8% | 81.1% | 92.4% | 94.1% |
| Chromium 151, every query at 12 px | 52.4% | 30.0% | 66.0% | 19.6% |  |  |  |

The sixth step (`open-30k`):

| Validation queries rendered by | Top-5 | First | Clean | Degraded | 16 px | 24 px | 48 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chromium 151 | 91.6% | 71.8% | 93.4% | 87.4% | 81.4% | 93.7% | 94.3% |
| WebKit 26.5 | 91.3% | 72.4% | 93.1% | 86.7% | 81.5% | 93.9% | 93.6% |
| Firefox 153 | 91.8% | 72.2% | 93.3% | 88.0% | 81.5% | 94.1% | 94.8% |
| Chromium 151, every query at 12 px | 54.0% | 31.5% | 67.1% | 22.3% |  |  |  |

The seventh step (`open2-30k`):

| Validation queries rendered by | Top-5 | First | Clean | Degraded | 16 px | 24 px | 48 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chromium 151 | 92.0% | 73.4% | 94.1% | 87.0% | 80.9% | 94.3% | 95.5% |
| WebKit 26.5 | 91.7% | 73.5% | 93.7% | 86.7% | 80.3% | 93.8% | 94.8% |
| Firefox 153 | 92.4% | 73.3% | 94.2% | 88.1% | 82.0% | 94.4% | 95.5% |
| Chromium 151, every query at 12 px | 54.6% | 32.6% | 68.2% | 21.8% |  |  |  |

The shipped model (`photo-30k`, [reads](style-reads.json)):

| Validation queries rendered by | Top-5 | First | Clean | Degraded | 16 px | 24 px | 48 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chromium 151 | 92.1% | 73.9% | 93.9% | 87.7% | 82.8% | 94.2% | 95.1% |
| WebKit 26.5 | 91.8% | 73.6% | 93.9% | 86.6% | 81.7% | 94.0% | 94.9% |
| Firefox 153 | 91.9% | 74.0% | 93.9% | 87.2% | 81.8% | 94.2% | 94.8% |
| Chromium 151, every query at 12 px | 55.4% | 33.0% | 68.6% | 23.5% |  |  |  |

Another rasterizer costs under a point. Twelve-pixel text is the floor: six-pixel x-heights keep 73% of clean crops in the top five, and next to nothing once such a crop is also shrunk and JPEG-compressed. A pixel ratio of 2 changes nothing here: the canvas is drawn at device pixels, so 16 CSS px at ratio 2 is the 32 px row. Windows ClearType and phones remain unmeasured.

## Other catalogs

The catalogs besides Google Fonts hold families the model never trained on, so they are the held-out read. `train.style_open` renders the default face of every such family the file-built catalogs index, in lower and title case at the catalog benchmark's sizes with one degraded copy of the first, through the frozen Chromium renderer; each query searches every shipped catalog at once, the page's All, and counts only the exact family, since twins are not known across catalogs ([report](style-open.json)). Since the sixth step trains on nine open families in ten, the read takes the held-out tenth (`--held-out`): 110,752 families in the search, 2,160 queries of 720 families never trained on. Google Fonts validation queries take the same search for comparison:

| Latin queries | Queries | Top-5 | First |
|---|---:|---:|---:|
| Held-out open families, all | 2,160 | 74.2% | 49.8% |
| Held-out open families, clean | 1,440 | 77.3% | 52.6% |
| Held-out open families, degraded | 720 | 67.9% | 44.3% |
| Google Fonts validation, same search | 5,919 | 70.8% | 48.1% |

The shipped student, which trained on no open family, read 77.3% and 54.8% on all 3,563 open families against 77.4% and 55.1% for Google's across the four file-built catalogs (5,783 families), and 73.8% and 50.2% against 73.1% and 49.8% once DaFont and Adobe Fonts shipped (45,542 families): the sixth step 72.3% against 71.7% at 55,428 families, and the seventh 72.2% against 70.6% at 110,543: each widening of the search costs both seen and unseen families alike, and at 110,000 families a family the model never saw is found more often than one it trained on, by three points with the eighth step.

By source, the larger ones and the ones that stand out:

| Source | Queries | Top-5 | First |
|---|---:|---:|---:|
| Debian | 591 | 74.3% | 49.9% |
| Font Library | 327 | 73.7% | 52.3% |
| GitHub | 1,098 | 73.8% | 47.7% |
| Fontshare | 24 | 83.3% | 70.8% |
| Velvetyne | 15 | 100.0% | 86.7% |
| Uncut | 27 | 81.5% | 77.8% |
| Use & Modify | 15 | 100.0% | 66.7% |
| Luuse | 15 | 80.0% | 66.7% |
| Fontsource | 12 | 66.7% | 33.3% |

The exact-family rule gives no credit to a clone: TeX Gyre and URW base 35 are the Times, Helvetica, Palatino and Bookman designs that Liberation, Arimo, Tinos and their kin in the other catalogs also draw, so the right name is one of several right answers the rule counts as wrong. The teacher now spans every collected face, so twins across catalogs are computable; wiring them into this read is next.

## By requirement

`train.style breakdown` splits the catalog benchmark's test queries, every family on text never trained on, by the requirements ([report](style-breakdown.json)), with the same script-aware search and twin credit; the figures are the shipped model's (`photo-30k`).

Decorative styles are the easy ones: slab geometric 100.0% (153 queries), theme tuscan 100.0%, theme woodtype 99.3%, theme blobby 99.3%, slab humanist 98.5%, theme wacky 98.2%; display 96.8% and handwriting 95.7% as categories. Plain text faces are the hard ones, because many of them look alike: sans geometric 86.7%, sans humanist 88.7%, serif transitional 88.7%, sans neo-grotesque 90.5%, sans serif overall 89.3%. So a collected catalog of brush or script faces needs coverage, not training.

Length matters as expected: 89.9% at 5–7 characters, 94.8% at 8–10. Other scripts trail Latin because of the script filter, not the encoder: Latin 93.0%, other scripts 84.6% on their own letters (the open fonts brought 158 scripts into the head; on rare ones the detection is still the weak link, see todo.md). Chinese is 63.6% (418 queries).

The shipped catalog holds every case and script of a face, which hides how little style crosses between letterforms. With one kind of reference removed for every family:

| Query | References kept | Top-5 | With every reference |
|---|---|---:|---:|
| Capitals | lowercase only | 55.9% | 92.6% |
| Lowercase | capitals only | 66.4% | 92.1% |
| Other scripts | Latin only | 62.7% | 84.6% |
| Chinese | Latin only | 38.1% | 63.1% |

A catalog built from one case or from Latin alone, like most collected previews, finds capitals, other scripts and Chinese far less often. Every cross row rose one to three points with the sixth step's data, and other scripts another two with the seventh's. Style transfer across case and script is still the next thing to train.

## Catalog

The Google Fonts catalog lists every face: 8,132 entries with 26,459 vectors (lowercase, uppercase and each script), each the average of its lines drawn at 16, 24 and 48 px by Chromium (249,162 lines in all), a style name (Thin to Black, Italic), script coverage, and, on default faces, per-script twins: families whose letters in that script are closer than the median distance between two renders of one face (0.0083 since the teacher was rebuilt over every collected face); Google's faces list Google twins only, the file-built catalogs' theirs across every catalog. Encoder (511,867 bytes, 6-bit weights with heads for 158 scripts), catalog (3,592,862, 4-bit vectors of 64 numbers) and required modules total 4,160,576 bytes, 1.24 MB with gzip and 1.05 MB with Brotli ([runtime check](style-runtime.json)); CPU and Metal agree with PyTorch within 4.1e-07, and a whole match takes 15 to 60 ms on Metal by the machine's load and 1.1 s on the CPU in Chromium, and 1 to 1.7 s in Node by the crop's size. DaFont's catalog adds 5.7 MB with gzip (96,212 faces, one a family, 30 MB raw: names, links and ids are most of it), Adobe Fonts 0.84 MB (21,816 rows for 4,788 faces), GitHub 0.95 MB (10,873 faces of 3,565 families, its twins across catalogs 1.4 MB of the raw file), Debian 0.35 MB, Font Library 0.32 MB, WhatFontIs 0.03 MB (226 faces, three rows each) and Other 0.17 MB (575 families); the page's All is about 9.6 MB, a third of a megabyte of it the twins the page folds by, and a photograph matched against its 187,078 rows takes 102 ms on Metal.

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

The shipped model's rows, read the same way on catalog validation ([reads](style-reads.json)): float rows 92.1 / 73.9, the shipped 4-bit rows 92.1 / 73.5. The seventh step lost two points of first place to 4 bits (92.0 / 73.4 against 91.5 / 71.1), so both readers now take 6-bit rows (`int6-base64`, `train.style catalog --bits 6`) for a model that needs them; the eighth loses four tenths and ships at 4. Product quantization, the rows cut into subvectors each replaced by the nearest of 256 k-means centroids, is not a way down from there: 16 subvectors of 4 numbers (16 bytes a row, half the 4-bit row) read 90.2 / 68.7, and 8 subvectors of 8 numbers (8 bytes) 81.7 / 52.6.

Below 6 bits, rounding per row breaks but clustering does not: with 16 shared values per layer and the codebooks retrained (`train.ten_model.cluster_values`, `Clustered`; the fifth step), the separable student keeps all but 1.1 points at 4 bits.

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
node scripts/python.mjs -m train.style_photos                                # WhatFontIs photographs, tracked at every check
node scripts/style-bench-engines.mjs webkit && node scripts/python.mjs -m train.style_bench pack --name catalog-webkit   # also firefox, and chromium --size 12
node scripts/python.mjs -m train.style project --run catalog-30k --dimensions 64
node scripts/python.mjs -m train.style train --run catalog-30k-64r --architecture font-conv96-192-288-384-384-v5 --roles train,development,test --select catalog:validation --lr 5e-5 --warm .data/style/catalog-30k-64/best.pt --steps 5000 --check 1250
node scripts/python.mjs -m train.style train --run student-10k --architecture font-sep96-192-288-384-384-v7 --teacher catalog-30k-64r --warm none --roles train,development,test --select catalog:validation --lr 1e-3 --steps 10000 --photo 0.4
node scripts/python.mjs -m train.style train --run direct-10k --architecture font-sep96-192-288-384-384-v7 --warm none --dimensions 64 --seed 2290464039 --roles train,development,test --select catalog:validation --lr 1e-3 --steps 10000 --photo 0.4   # the control
node scripts/python.mjs -m train.style train --run student-30k --architecture font-sep96-192-288-384-384-v7 --teacher catalog-30k-64r --warm .data/style/student-10k/best.pt --roles train,development,test --select catalog:validation --lr 5e-4 --steps 20000 --photo 0.4
node scripts/python.mjs -m train.style train --run student-30k-q4 --architecture font-sep96-192-288-384-384-v7 --cluster 4 --teacher catalog-30k-64r --warm .data/style/student-30k/best.pt --roles train,development,test --select catalog:validation --lr 2e-4 --steps 3000 --check 1000 --photo 0.4
node scripts/python.mjs -m train.style final --run student-30k --source browser     # the release read: catalog test role, then breakdown and deploy
node scripts/python.mjs -m train.style breakdown --run student-30k
node scripts/python.mjs -m train.style deploy --run student-30k
node scripts/catalog-files.mjs                                              # every other catalog, with the shipped encoder
node checks/encoder.mjs models/encoder/encoder.json models/encoder/google-fonts.json bench/style-runtime.json
node scripts/demo-build.mjs
node scripts/python.mjs -m train.style_open plan                            # other catalogs: open families never trained on, read against every shipped catalog
node scripts/python.mjs -m train.style_bench render --name open && node scripts/python.mjs -m train.style_bench pack --name open
node scripts/python.mjs -m train.style_open evaluate --run student-30k
node scripts/python.mjs -m train.style_reads --run student-30k                # other rasterizers, 12 px, 4-bit and product-quantized rows
node scripts/python.mjs -m train.style_teacher build --workers 10             # the teacher over Google and every open face (bench/open-fonts.json)
node scripts/python.mjs -m train.style train --run open-30k --architecture font-sep96-192-288-384-384-v7 --teacher catalog-30k-64r --warm .data/style/student-30k/best.pt --roles train,development,test --select catalog:validation --lr 5e-4 --steps 30000 --check 2500 --photo 0.4
node scripts/python.mjs -m train.style export --run open-30k && node scripts/python.mjs -m train.style catalog --run open-30k --source browser
node scripts/python.mjs -m train.style final --run open-30k --source browser && node scripts/python.mjs -m train.style breakdown --run open-30k && node scripts/python.mjs -m train.style deploy --run open-30k
node scripts/python.mjs -m train.style_open plan --held-out                  # the open read on the tenth never trained on
```

Preview catalogs are recompiled with `scripts.preview_catalog` after any encoder change; the compiler indexes faces whose captures are complete and reports the rest, so a live collection no longer stops the build.
