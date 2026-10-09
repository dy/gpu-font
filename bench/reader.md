# The line reader and the crop's own words

A crop's embedding is compared with catalog rows made from fixed reference lines. Set the shortlist's faces in the crop's own text instead and the comparison is letter for letter. [photos.md](photos.md) measured what that is worth with the text known: eighteen points of first place. This report is about getting the text without asking for it: a line reader small enough to ship beside the encoder, and what the two together do on the final photographs.

## What it is

`models/reader/reader.json` reads one line of Latin text: space, digits and the 52 letters; an accented letter reads as its base letter. It takes the strip `prepareLine`'s `line` sampler gives (`src/line.mjs`): the whole line, deskewed as a crop is, 48 pixels tall and as wide as the line is, up to 1,024. Architecture `read-24-48-96-160-v1` (`train/read.py`):

| Stage | Layers | Output |
|---|---|---|
| Three stride-2 blocks | a dense 3×3 to 24 channels, then depthwise 3×3 + pointwise to 48 and to 96 | 96 × 6 rows × W/8 columns |
| Fold | the six rows become channels | 576 × 1 × W/8 |
| Mix and context | pointwise to 160, then three 1×3 depthwise taps along the line at dilations 1, 2 and 4, each with a pointwise | 160 × 1 × W/8 |
| Head | pointwise to 64 classes (blank + 63) | one logit per class per column |

Connectionist temporal classification over the columns, eight pixels each; greedy decoding collapses repeats and drops blanks. 187,264 weights, rounded per output row to 6 bits as the encoder's are: 222 KB on disk, 156 KB gzipped. The browser runs it on the CPU (`src/reader.mjs`), 73 ms a line at the median; a strip of 600 pixels is about twenty million multiply-adds.

## How it was trained

On the style stream's own views, nothing collected for it: words of 1–24 characters drawn by `train.style_data` in the 23,111 Latin faces of the training families, 15% clean, the rest damaged as the encoder's views are, half of those on photographed surfaces (`--photo 0.5`). Labels come free with the render. 20,000 steps of 48 strips, AdamW with a one-cycle schedule peaking at 2e-3, weight decay 1e-4, six data workers: 26 minutes on an M4 Max, the process tree peaking at 9.4 GB.

Faces of the held-out open families and the test Google families never trained it. 768 views of those 3,128 faces, half clean and half damaged, are its development set:

| Views of unseen fonts | Character error | Whole line right |
|---|---:|---:|
| Clean renders | 11.9% | 70.3% |
| Damaged, half on surfaces | 29.3% | 46.7% |

The error fell steadily to 15,000 steps and flattened after. A larger reader would read better; this one is sized for the download.

## On the photographs

400 of the final WhatFontIs-Bench photographs (bench/sametext.json): Google families the encoder never trained on, 42 of them, drawn by seed 1 from the 840 with a text label, each at the set's crop box. The same crop and embedding ranked three ways over every shipped catalog, 110,953 families, with twin credit as every read gives it:

| Ranking by | First | Top five | Top twenty | By family, first | By family, top five |
|---|---:|---:|---:|---:|---:|
| The catalog's rows | 26.0% | 49.5% | 73.2% | 25.8% | 48.6% |
| Rows, plus the first 30 set in the label's text | 44.8% | 66.2% | 76.5% | 44.5% | 65.0% |
| Rows, plus the first 30 set in the text the reader read | 43.8% | 65.8% | 76.5% | 43.6% | 64.5% |

The reader read these crops with a character error of 11.0%, the whole word right 68.8% of the time. What its text buys is within a point of what the true text buys: a misread letter or two still sets the faces in nearly the right word, and the catalog score is still counted. Eighteen points of first place and sixteen of the top five, on the shipped encoder, from 156 KB more. The eighth step's encoder read these 400 at 53.0% top five by its rows and 69.8% with the reader's text: the ninth gave up three and a half points on Google families' photographs for its gains everywhere else ([photos.md](photos.md)), and the stage's lift is the same on both.

The same-text stage (`src/sametext.mjs`) draws the first 30 rows' faces in the text at the three reference sizes where the page can load the font, prepares and embeds each as it does a crop, averages the sizes, and adds the cosine with the crop's embedding to the catalog score; a face it cannot draw counts its catalog score twice, so it neither gains nor loses against a face whose text reads exactly as its references do. Here only Google Fonts drew (the measurement loads faces from Google Fonts by name); on the page every face with a loadable file draws. Both same-text rankings together took 239 ms a photograph at the median, the catalog match 112 ms and the read 73 ms, in Chromium on Apple Metal.

## On the page

After a match the page reads the crop's line, shows the words read in every preview so each candidate can be seen in the picture's own text, and with WebGPU sorts the shortlist by the catalog and same-text scores averaged; the note under the matches says which text was used. Editing any preview re-ranks with the typed text, so a misread costs one correction. Without WebGPU the words still show, and the catalog's order stands: the thirty faces at three sizes would take the CPU half a minute.

## Limits

- Latin letters, digits and space. Punctuation is not read, because the stream draws none; Cyrillic and Greek wait for the same recipe.
- Case is where it errs most on display faces: capitals read as lowercase shapes when a face draws them alike.
- A crop that holds a whole scene rather than a line reads poorly; the measurement above is at the set's crop boxes, as the page's crop tool gives them.
- A line wider than about twenty times its height is squeezed to 1,024 pixels rather than cut.

## Reproduce

```sh
node scripts/python.mjs -m train.read train --run reader-20k --steps 20000 --check 1000 --workers 6 --photo 0.5
node scripts/python.mjs -m train.read export --run reader-20k --bits 6     # .data/read/reader-20k/reader.json → models/reader/reader.json
node scripts/sametext.mjs --sample 400 --seed 1                           # with `npm run demo` running; writes bench/sametext.json
```
