# Photos and real requests

The [frozen benchmark](style.md#frozen-benchmark) reads Chromium renders. The results below are the shipped student (`student-30k`, SHA-256 `f57d7254…`); the failure analyses under them were made with the earlier encoder (`wider-20k`) and describe what goes wrong, with that model's numbers where they give any. People bring photos, screenshots and logos. Three sets measure that, each read as the page reads it (`scripts/photos.mjs`): every shipped catalog searched as one (the page's All), the crop at full resolution, rows folded as the page shows them. A result is right when the true family, or a family folded into its row, is among the first five rows; any weight counts.

| Set | What it holds | gpu-font can answer | Licence | Crop |
|---|---|---|---|---|
| [WhatFontIs-Bench v1.0](https://github.com/whatfontis/WhatFontIs-Bench) | 11,995 photos of one word, 7–12 letters, printed or painted on real surfaces, walls and objects; 600 fonts, 20 images each | 6,775 images of 339 fonts: 191 Google Fonts families (9 of them Adobe Fonts entries that are free families: Lato, Lora, Raleway…), 148 fonts of DaFont, Adobe Fonts and the file-built catalogs | CC BY 4.0, backgrounds CC0 1.0 | the set's own `crop_box` |
| DaFont forum requests, from [Max Halford's LLM benchmark](https://github.com/MaxHalford/llm-font-recognition) | 4,887 "what font is this?" posts, 1,573 with an answer the forum confirmed | 612 answers (38.9%), 341 of them DaFont's own fonts; 151 cropped, 459 not yet | pictures belong to their posters: links only, kept in `.data/` | by hand, [dafont-crops.json](dafont-crops.json) |
| Finder set, [issue #1](https://github.com/dy/gpu-font/issues/1) | 44 crops: 10 Google families, 10 commercial classics, 10 hard cases, 6 scripts, 4 of one to three letters, 4 probe images | 30 scored crops; classics judged as substitutes | CC BY 4.0 photos and renders of OFL fonts | fixed PNG per crop |

## Results

WhatFontIs-Bench ([report](whatfontis.json)). Final: fonts that never entered training or checkpoint selection (the style split's test role, and every catalog besides Google Fonts). Development: the rest.

| | Images | Top-1 | Top-5 | Top-20 |
|---|---:|---:|---:|---:|
| gpu-font, final | 3,795 | 22.1% | 42.4% | 59.0% |
| gpu-font, development | 2,980 | 22.1% | 45.8% | 65.5% |
| WhatFontIs API, all images, among 1.2 million fonts (published by the set's authors) | 11,995 | 83.7% | 93.3% | 96.5% |

The rows are not the same images: WhatFontIs publishes only totals, and it indexes all 600 fonts, since the set was built from its catalogue. gpu-font searches 45,542 families, 339 of the set's fonts among them, 148 of those from DaFont and Adobe Fonts as one preview each with weight and style unknown.

With the four file-built catalogs alone (5,783 families, 198 of the set's fonts, 3,960 images), the same read gave 51.2% top-5 and 68.3% top-20 on the 980 final images and 54.2% and 71.1% on development: the 40,000 families shipped since, each on one preview, cost about eight points on the images both searches share.

By kind (gpu-font top-5, all 6,775): printed objects 52.3%, textures 45.1%, scenes 36.4%; easy 52.5%, medium 49.7%, hard 36.7%. Families seen in training score as unseen ones (45.7% against 45.1% for the style split's test families; 41.7% for the 148 fonts of the other catalogs), so knowing the font is not what is missing. Median time 74 ms a photo, p95 145 ms, WebGPU on an M4 Max.

DaFont requests ([report](dafont.json)): 28.5% top-1, 47.7% top-5 on the 151 cropped requests, searched across all six catalogs (34.4% and 55.0% with the four file-built catalogs, before DaFont's own fonts were indexed). Each chatbot answered a different share, so gpu-font is scored beside it on exactly the requests it answered:

| On the same requests | Requests | Chatbot top-5 | gpu-font top-5 |
|---|---:|---:|---:|
| Gemini 2.5 Flash (preview, 2025) | 65 | 6.2% | 43.1% |
| GPT-4o-mini | 37 | 18.9% | 40.5% |

On every answered request, including answers gpu-font does not index, the chatbots reach 1.5% (753) and 2.2% (360); gpu-font's ceiling there is the 38.9% it indexes. The chatbots saw whole pictures; gpu-font saw a hand crop.

Finder set ([answers](finders.json)), gpu-font top-5: Google families 3 of 10, hard cases 5 of 10, scripts 6 of 6, one to three letters 1 of 4. Probes: the matched face has the query's style (upright, italic); the 300 probe matched a DaFont face, whose weight is unknown, and the 700 probe an 800 face. Classics await a judgement of each look-alike. [finders.json](finders.json) keeps a place for each of the eight finders the page compares, with how it may be run: WhatFontIs through its API, Lens and gpu-font from a script, the other five by hand, as their terms or form require.

## Where it fails

**The photo, not the font** ([controls](photo-controls.json), read again with the student). 199 development photos read three ways:

| The same word in the same face | Top-1 | Top-5 |
|---|---:|---:|
| Clean Chromium render, 64 px | 55.3% | 79.4% |
| Photo, binarized (Otsu) | 28.6% | 54.3% |
| Photo | 25.1% | 51.3% |

With the earlier encoder a threshold recovered half of a 45-point gap between photo and render (45.2% against 90.0%), so texture, coloured ink and uneven light passing through preparation into the windows cost most of it. The student, trained on photographed surfaces (`--photo 0.4`), reads the photo at 51.3% and a threshold adds 3 points of the 28 that remain (59.3% and 3.5 of 27 with the four file-built catalogs): the surface is mostly learned away, and what is left is the photograph's geometry and blur, which no threshold restores. Thresholding still costs clean renders (72.9% top-5 from 79.4%) and moves the forum crops, mostly flat graphics, from 47.7% to 46.4%.

**No answer on legible photos, fixed.** Preparation called 173 of the 3,880 photos (4.5%) low contrast and returned nothing: 159 of them scenes, whose text the set keeps at least 70 grey levels from its surroundings (50 on the hard level). Every one was a polarity error. The median of the crop's outer ring sets the paper, and a ring darker than middle grey means light text; dark letters on a darkish wall were read as light ones, and no ink cleared the cutoff (contrast 0.04–0.08 against 0.35–0.45 the other way). Preparation now reads the other side when the first guess finds no ink above the cutoff and the other side's does. Only crops that failed before can change: the frozen benchmark's 9,409 and 27,703 samples all prepared, and its 9,409 renders and their inversions prepare byte for byte as before. With the same model, all 173 photos now get an answer, 74 with the right family in the top five (33 first): top-5 44.5% overall (1,726 of 3,880), 45.2% final, 44.3% development, scenes 33.8%. The rule has no fitted parameter; the rejected photos it was read on came from both parts.

**Effects read as fonts.** 12 of the 70 forum misses put a font whose design is the effect first: Jawbreaker OL2, Technique OL, Euphoric 3D, Xtrusion, Corpulent Caps Shadow, Rock 3D, Londrina Outline, Big Shoulders Inline, Rubik Glitch Pop. Training never shows a plain face with an outline, a shadow or an extrusion.

**Letterless fonts rank.** 5 forum misses put jsMath cmex10 (math delimiters at Latin code points), feta26 (music glyphs) or Edu AU VIC WA NT Arrows (letters overlaid with stroke arrows) first. They claim Latin and do not draw ordinary letters; the Google catalog is meant to exclude letterless families, and two of these are in it.

**Coverage.** 61% of confirmed forum answers are in no shipped catalog (90% before DaFont and Adobe Fonts shipped). The most requested: Edwardian Script (8), Benguiat and Compacta (4 each), Aachen, Argue, Blaster, Burgues Script and Copperplate Gothic (3 each); [dafont.json](dafont.json) lists all 78 asked for more than once. One thread can hold several answers for several texts; the scrape keeps one, so crops follow the thread's words for that one.

## Rejection

gpu-font always answers. A rule that says "unknown" instead needs a signal that separates crops of fonts a catalog holds from crops of fonts it does not. Measured on the 64-number model with 2,980 development photographs of Google families (present) against 3,000 photographs of the set's 406 fonts in no shipped catalog (absent), scored as the page searches; `train.style export` calibrates this for every model and ships a threshold only when present coverage exceeds absent acceptance by 0.3 or more (`train/style.py` REJECTION_FLOOR):

| Best-match score at least | Present photographs kept | Absent photographs let through | Clean validation renders kept |
|---|---:|---:|---:|
| 0.60 | 85.2% | 85.9% | 98.9% |
| 0.70 | 62.4% | 59.3% | 93.7% |
| 0.75 | 46.3% | 38.0% | 87.0% |
| 0.80 | 26.4% | 15.9% | 74.6% |
| 0.85 | 8.8% | 3.2% | 51.4% |

The distributions all but coincide. Nor do the margin to the next distinct design (best cut separates by 0.05), agreement between the crop's windows (0.04), the script or category confidence (0.08, 0.04), or a logistic combination of all of them, trained on one half and read on the other: AUC 0.555, and keeping 95% of present photographs lets 96% of absent ones through. An absent font's nearest catalog design scores as high as a present font's own: at 3,786 families, most fonts have a close relative in some catalog, and a low score means a far design, not an absent one. So no threshold ships (`matcher.threshold` is null), the page shows no "unknown", and the claim stays what the score expresses: the closest design in the catalog, with its similarity.

## Development and final

Tuning against these sets may use the development part only: the 151 DaFont requests (their failures are read above) and WhatFontIs images of training and development families. The final read, once per released model: WhatFontIs images of held-out families, forum posts newer than the scrape, and the finder set. A model trained on every family (`catalog-30k`) has seen the final families too, so for it the final part is held out from tuning only, not from training.

## Next

- Train on photographed surfaces: CC0 textures and scenes other than the 153 WhatFontIs backgrounds, uneven light, coloured ink, cast shadow, glare, print wear; and on plain faces wearing effects: outline, drop shadow, extrusion, perspective. Select checkpoints on the development part; the frozen benchmark must not drop.
- Drop letterless families from every catalog.
- Scrape forum posts newer than the scrape (September 2025) for a fresh final read.

## Reproduce

With `npm run demo` running and WebGPU in Chromium:

```sh
node scripts/whatfontis.mjs      # downloads the set (1.9 GB) into .data/whatfontis, writes bench/whatfontis.json
node scripts/dafont.mjs          # downloads the scrape and the 153 pictures, writes bench/dafont.json
node scripts/finders.mjs         # writes the 44 crops to .data/finders, gpu-font's answers to bench/finders.json
node scripts/photo-controls.mjs  # writes bench/photo-controls.json
node scripts/python.mjs -m train.style_photos   # packs the Google-family pictures for train.style: photos-development, tracked at every checkpoint, and photos-final
```

`.data/dafont/gpu-font-guesses.json` holds gpu-font's answers in the LLM benchmark's own format.
