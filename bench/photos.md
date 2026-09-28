# Photos and real requests

The [frozen benchmark](style.md#frozen-benchmark) reads Chromium renders. The results below are the shipped model (`photo-30k`, SHA-256 `0ac8707e…`), which trained on the development photographs and the development sites' screens, so only the final part is a read; the failure analyses under them were made with the earlier encoder (`wider-20k`) and describe what goes wrong, with that model's numbers where they give any. People bring photos, screenshots and logos. Three sets measure that, each read as the page reads it (`scripts/photos.mjs`): every shipped catalog searched as one (the page's All), the crop at full resolution, rows folded as the page shows them. A result is right when the true family, or a family folded into its row, is among the first five rows; any weight counts.

| Set | What it holds | gpu-font can answer | Licence | Crop |
|---|---|---|---|---|
| [WhatFontIs-Bench v1.0](https://github.com/whatfontis/WhatFontIs-Bench) | 11,995 photos of one word, 7–12 letters, printed or painted on real surfaces, walls and objects; 600 fonts, 20 images each | 11,935 images of 597 fonts: 191 Google Fonts families, 116 of Adobe Fonts (69 found by the store words in their titles), 61 of DaFont, 3 of GitHub, and the 226 no other catalog held, from WhatFontIs's own renders; 3 fonts are gone from every source | CC BY 4.0, backgrounds CC0 1.0 | the set's own `crop_box` |
| DaFont forum requests, from [Max Halford's LLM benchmark](https://github.com/MaxHalford/llm-font-recognition) | 4,887 "what font is this?" posts, 1,573 with an answer the forum confirmed | 1,017 answers (64.7%), 775 of them DaFont's own fonts; 151 cropped, the rest not yet | pictures belong to their posters: links only, kept in `.data/` | by hand, [dafont-crops.json](dafont-crops.json) |
| Finder set, [issue #1](https://github.com/dy/gpu-font/issues/1) | 44 crops: 10 Google families, 10 commercial classics, 10 hard cases, 6 scripts, 4 of one to three letters, 4 probe images | 30 scored crops; classics judged as substitutes | CC BY 4.0 photos and renders of OFL fonts | fixed PNG per crop |
| Web screens, `scripts/web_screens.mjs` | lines cut from the homepages of the Chrome UX Report's top sites where robots.txt allows, each labelled by the font file Chromium drew it with; 29,301 crops from 5,694 sites at the read | 19,382 crops of 397 families; 472 with no usable font name, 9,447 in no catalog | other sites' designs: local only | the collector's line crop |
| Wikimedia Commons, typefaces by name, `scripts/commons_typefaces.py` | photographs of signs, packaging and print filed under a typeface's category; 2,648 photos of 102 typefaces | 1,160 photos of 48 typefaces with a line of text found | each photo's own free licence | the tallest line of text found (`scripts/text_boxes.py`) |

## Results

WhatFontIs-Bench ([report](whatfontis.json)). Final: fonts that never entered training or checkpoint selection (the style split's test role, and every catalog besides Google Fonts). Development: the rest.

| | Images | Top-1 | Top-5 | Top-20 |
|---|---:|---:|---:|---:|
| gpu-font, final | 8,955 | 20.2% | 41.4% | 59.0% |
| gpu-font, development (trained on since the sixth step) | 2,980 | 29.7% | 56.2% | 75.0% |
| gpu-font, all | 11,935 | 22.6% | 45.1% | 63.0% |
| WhatFontIs API, all images, among 1.2 million fonts (published by the set's authors) | 11,995 | 83.7% | 93.3% | 96.5% |

The rows are not the same images: WhatFontIs publishes only totals, and it indexes all 600 fonts, since the set was built from its catalogue. gpu-font searches 110,752 families, 597 of the set's 600 fonts among them, 406 of those outside Google Fonts, most as one to three preview lines each with weight and style unknown. A row of a family the teacher cannot tell from the true one counts as right, as on the rendered benchmark.

By the catalog that holds the font: Google Fonts 55.3% top five (3,820 images of 191 fonts), Adobe Fonts 49.4% (2,315 of 116), the WhatFontIs slice 39.1% (4,520 of 226, each font known by three lines of WhatFontIs's own render), GitHub 36.7% (60 of 3), DaFont 27.8% (1,220 of 61, each font one preview). What a catalog knows of a font decides the read as much as the model does.

Like for like, on the 8,915 images seven catalogs index and with twin credit for both, the eighth step reads 46.1% top five against the seventh's 45.2%, 62.5% against 60.9% top twenty, and 41.0% against 40.4% on the 5,935 final images; twin credit itself is worth eight points of top five on the development photographs (55.0% from 47.1% for the seventh step). Earlier: the sixth step read 44.2% final among 55,428 families with 41,113 of DaFont's; the 50,000 single-preview faces shipped since cost about five points on the images every search shares, and DaFont whole (96,212 faces against the 49,449 of its first eight parts) three hits of 3,233. The wrong first answers do not come from the single-preview rows: on the development photographs DaFont supplies 30% of them for 52% of the rows searched, Adobe Fonts and the file-built catalogs one and a half to two times their share, a fifth to a third of those clones the twin credit now counts.

By kind (gpu-font top-5, all 11,935): printed objects 53.3%, textures 47.1%, scenes 36.2%; easy 53.4%, medium 49.4%, hard 38.8%. Google families the model trained on read 55.7%, the style split's test families 52.0%, the 406 fonts of the other catalogs 40.3%. Median time 94 ms a photo, p95 138 ms, WebGPU on an M4 Max.

DaFont requests ([report](dafont.json)): 22.5% top-1, 46.4% top-5 on the 151 cropped requests, searched across all eight catalogs with twin credit (the seventh step by the same rule: 21.2% and 44.4%; without the credit 16.6% and 41.1%). The sixth step read 24.5% and 53.6% with 41,113 DaFont faces, when the answers indexed were 687 of the 1,573 confirmed; they are 1,017 now, and the crops were all made for answers the smaller scope held, so they overstate what the larger catalog cost. Each chatbot answered a different share, so gpu-font is scored beside it on exactly the requests it answered:

| On the same requests | Requests | Chatbot top-5 | gpu-font top-5 |
|---|---:|---:|---:|
| Gemini 2.5 Flash (preview, 2025) | 65 | 6.2% | 43.1% |
| GPT-4o-mini | 37 | 18.9% | 46.0% |

On every answered request, including answers gpu-font does not index, the chatbots reach 1.5% (753) and 2.2% (360); gpu-font's ceiling there is the 64.7% it indexes. The chatbots saw whole pictures; gpu-font saw a hand crop.

Finder set ([answers](finders.json)), gpu-font top-5: 18 of the 30 scored crops, 13 first (the seventh step: 13 and 9). Probes: the matched face has the query's style (upright, italic) and weight (300, 700), the 300 and the upright and italic probes first, the 700 fifth. Classics await a judgement of each look-alike. [finders.json](finders.json) keeps a place for each of the eight finders the page compares, with how it may be run: WhatFontIs through its API, Lens and gpu-font from a script, the other five by hand, as their terms or form require.

## Web screens

Lines from the top sites' homepages, as the browser drew them ([report](screens.json)): 52,590 crops from 10,233 sites at the read, 35,281 of them naming a family a catalog holds. Sites are split by the hash of their origin: nine in ten are development, whose crops train the model since the eighth step; the held-out tenth is the read. The label is the font file, so a page set in Arial or Helvetica is labelled with a font no catalog holds, and the exact-family rule counts the answer wrong even when it names a clone (a twin by the teacher's distance now counts, as in every read): so the installed fonts are a floor, and the web fonts the pages load are the read.

| | Crops | Families | Top-1 | Top-5 | By family, top-1 | By family, top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Held-out sites | 3,503 | 169 | 11.6% | 27.7% | 15.2% | 31.7% |
| Development sites, trained on | 31,778 | 510 | 12.0% | 27.2% | 17.1% | 32.6% |
| Web fonts, all sites | 20,676 | 509 | 18.4% | 40.6% | 17.8% | 33.1% |
| Installed fonts, all sites | 14,605 | 44 | 2.9% | 8.3% | 6.9% | 19.2% |
| Under 14 px | 8,543 | 301 | 7.4% | 19.0% | 8.4% | 24.3% |
| 14–19 px | 18,028 | 428 | 12.1% | 28.3% | 17.2% | 32.0% |
| 20–31 px | 5,887 | 370 | 18.3% | 37.5% | 22.3% | 40.9% |
| 32 px and over | 2,823 | 286 | 11.9% | 24.2% | 17.6% | 32.1% |

By family, each counted once, so Arial's thousands of crops do not speak for the rest. The seventh step read 13.3% top five on the 19,382 crops collected by then, 19.6% on web fonts and 8.0% under 14 px, with no twin credit and no site held out; training on the development sites' lines doubled the read, and the held-out sites read as the development ones do. Half the crops are 14 to 19 px, where the frozen benchmark reads 87% on clean renders of 5 to 10 letters; body text at those sizes on a page, anti-aliased, its words spaced, reads 28.3%. Median 70 ms a crop, p95 90 ms, WebGPU on an M4 Max.

## Commons photographs

Photographs of typefaces in use from Wikimedia Commons ([report](commons.json)), the 1,138 of 48 typefaces whose name a shipped catalog holds and in which a line of text was found; none trained on. gpu-font puts the typeface in the top five for 3.8% of them (43 photos), first for 1.1%, by typeface 1.5%: Futura 31 of 327, Helvetica 0 of 313, Arial 0 of 65, ITC Benguiat 0 of 54; the seventh step read 2.4%.

The read measures its label and its catalogs as much as the model. The category names the photo, not the line: a Futura plate's tallest line reads "Berthold AG Berlin", a street sign's the town's name in another face, and the line read is whatever lettering is largest. The catalog's namesake is often not the typeface: Futura is a font of that name in a GitHub repository, Helvetica and Arial are Adobe Fonts entries indexed from one specimen line each, Gill Sans a DaFont look-alike. And the photographs are signage in perspective, carved, cast and weathered, at every angle. What the set does give is a floor for photographs nobody framed for a benchmark, and a list of what such a photograph needs: a line chosen for its typeface, not its size, references of the real typeface with more than one line of evidence, and perspective corrected before the crop. Until then it is kept as a floor, not a gate.

## Where it fails

**The photo, not the font** ([controls](photo-controls.json), read again with the student). 199 development photos read three ways:

| The same word in the same face | Top-1 | Top-5 |
|---|---:|---:|
| Clean Chromium render, 64 px | 47.7% | 78.9% |
| Photo, binarized (Otsu) | 28.1% | 57.3% |
| Photo | 27.6% | 53.8% |

With the earlier encoder a threshold recovered half of a 45-point gap between photo and render (45.2% against 90.0%), so texture, coloured ink and uneven light passing through preparation into the windows cost most of it. The shipped model, trained on textured surfaces from the bank and on the development photographs themselves, reads these 199 at 53.8%, and a threshold adds 3.5 points (57.3%): the surface is mostly learned away, and the 25 points to the clean render are the photograph's geometry and blur, which no threshold restores. The seventh step read them at 48.2% against 81.9% for the clean render, a gap of 34 points; the eighth step's surfaces closed it to 25, three points of it paid on the clean render. Thresholding still costs clean renders (69.3% top-5 from 78.9%).

**No answer on legible photos, fixed.** Preparation called 173 of the 3,880 photos (4.5%) low contrast and returned nothing: 159 of them scenes, whose text the set keeps at least 70 grey levels from its surroundings (50 on the hard level). Every one was a polarity error. The median of the crop's outer ring sets the paper, and a ring darker than middle grey means light text; dark letters on a darkish wall were read as light ones, and no ink cleared the cutoff (contrast 0.04–0.08 against 0.35–0.45 the other way). Preparation now reads the other side when the first guess finds no ink above the cutoff and the other side's does. Only crops that failed before can change: the frozen benchmark's 9,409 and 27,703 samples all prepared, and its 9,409 renders and their inversions prepare byte for byte as before. With the same model, all 173 photos now get an answer, 74 with the right family in the top five (33 first): top-5 44.5% overall (1,726 of 3,880), 45.2% final, 44.3% development, scenes 33.8%. The rule has no fitted parameter; the rejected photos it was read on came from both parts.

**Effects read as fonts.** 12 of the 70 forum misses put a font whose design is the effect first: Jawbreaker OL2, Technique OL, Euphoric 3D, Xtrusion, Corpulent Caps Shadow, Rock 3D, Londrina Outline, Big Shoulders Inline, Rubik Glitch Pop. Training never shows a plain face with an outline, a shadow or an extrusion.

**Letterless fonts rank.** 5 forum misses put jsMath cmex10 (math delimiters at Latin code points), feta26 (music glyphs) or Edu AU VIC WA NT Arrows (letters overlaid with stroke arrows) first. They claim Latin and do not draw ordinary letters; the Google catalog is meant to exclude letterless families, and two of these are in it.

**Coverage.** 35% of confirmed forum answers are in no shipped catalog (56% with DaFont's first eight parts, 90% before DaFont and Adobe Fonts shipped). The most requested: Edwardian Script (8), Benguiat and Compacta (4 each), Aachen, Burgues Script, Copperplate Gothic, Gazzetta and Primal Sailor (3 each); [dafont.json](dafont.json) lists all 42 asked for more than once. One thread can hold several answers for several texts; the scrape keeps one, so crops follow the thread's words for that one.

## Rejection

gpu-font always answers. A rule that says "unknown" instead needs a signal that separates crops of fonts a catalog holds from crops of fonts it does not. Measured on the 64-number model with 2,980 development photographs of Google families (present) against 3,000 photographs of the set's 406 fonts in no shipped catalog (absent), scored as the page searches; `train.style export` calibrates this for every model and ships a threshold only when present coverage exceeds absent acceptance by 0.3 or more (`train/style.py` REJECTION_FLOOR):

| Best-match score at least | Present photographs kept | Absent photographs let through | Clean validation renders kept |
|---|---:|---:|---:|
| 0.60 | 85.2% | 85.9% | 98.9% |
| 0.70 | 62.4% | 59.3% | 93.7% |
| 0.75 | 46.3% | 38.0% | 87.0% |
| 0.80 | 26.4% | 15.9% | 74.6% |
| 0.85 | 8.8% | 3.2% | 51.4% |

The distributions all but coincide. The seventh step's export calibrates the same way: a threshold of 0.79 would keep 55.6% of present photographs, let 37.3% of absent ones through and keep 86.6% of clean renders, a separation of 0.18, so it ships no threshold either.

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
node scripts/screens.mjs         # the web screens read of .data/screens (scripts/web_screens.mjs collects them), writes bench/screens.json
node scripts/commons.mjs         # the Commons read of .data/photos/commons (scripts/commons_typefaces.py collects them), writes bench/commons.json
node scripts/python.mjs -m train.style_photos   # packs the Google-family pictures for train.style: photos-development, tracked at every checkpoint, and photos-final
node scripts/python.mjs -m train.style photos --run open2-30k,photo-30k   # two runs' exported models on the packed photographs, development and final, against the Google references: a comparison, not the page's read
```

`.data/dafont/gpu-font-guesses.json` holds gpu-font's answers in the LLM benchmark's own format.
