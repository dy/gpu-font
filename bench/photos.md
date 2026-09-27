# Photos and real requests

The [frozen benchmark](style.md#frozen-benchmark) reads Chromium renders. The results below are the shipped model (`open2-30k`, SHA-256 `7993b256…`), which trained on the development photographs, so only the final part is a read; the failure analyses under them were made with the earlier encoder (`wider-20k`) and describe what goes wrong, with that model's numbers where they give any. People bring photos, screenshots and logos. Three sets measure that, each read as the page reads it (`scripts/photos.mjs`): every shipped catalog searched as one (the page's All), the crop at full resolution, rows folded as the page shows them. A result is right when the true family, or a family folded into its row, is among the first five rows; any weight counts.

| Set | What it holds | gpu-font can answer | Licence | Crop |
|---|---|---|---|---|
| [WhatFontIs-Bench v1.0](https://github.com/whatfontis/WhatFontIs-Bench) | 11,995 photos of one word, 7–12 letters, printed or painted on real surfaces, walls and objects; 600 fonts, 20 images each | 7,415 images of 371 fonts: 191 Google Fonts families (9 of them Adobe Fonts entries that are free families: Lato, Lora, Raleway…), 180 fonts of DaFont, Adobe Fonts, GitHub and the file-built catalogs | CC BY 4.0, backgrounds CC0 1.0 | the set's own `crop_box` |
| DaFont forum requests, from [Max Halford's LLM benchmark](https://github.com/MaxHalford/llm-font-recognition) | 4,887 "what font is this?" posts, 1,573 with an answer the forum confirmed | 1,017 answers (64.7%), 775 of them DaFont's own fonts; 151 cropped, the rest not yet | pictures belong to their posters: links only, kept in `.data/` | by hand, [dafont-crops.json](dafont-crops.json) |
| Finder set, [issue #1](https://github.com/dy/gpu-font/issues/1) | 44 crops: 10 Google families, 10 commercial classics, 10 hard cases, 6 scripts, 4 of one to three letters, 4 probe images | 30 scored crops; classics judged as substitutes | CC BY 4.0 photos and renders of OFL fonts | fixed PNG per crop |
| Web screens, `scripts/web_screens.mjs` | lines cut from the homepages of the Chrome UX Report's top sites where robots.txt allows, each labelled by the font file Chromium drew it with; 29,301 crops from 5,694 sites at the read | 19,382 crops of 397 families; 472 with no usable font name, 9,447 in no catalog | other sites' designs: local only | the collector's line crop |
| Wikimedia Commons, typefaces by name, `scripts/commons_typefaces.py` | photographs of signs, packaging and print filed under a typeface's category; 2,648 photos of 102 typefaces | 1,160 photos of 48 typefaces with a line of text found | each photo's own free licence | the tallest line of text found (`scripts/text_boxes.py`) |

## Results

WhatFontIs-Bench ([report](whatfontis.json)). Final: fonts that never entered training or checkpoint selection (the style split's test role, and every catalog besides Google Fonts). Development: the rest.

| | Images | Top-1 | Top-5 | Top-20 |
|---|---:|---:|---:|---:|
| gpu-font, final | 4,435 | 22.1% | 41.2% | 57.1% |
| gpu-font, development (trained on since the sixth step) | 2,980 | 22.6% | 47.1% | 67.9% |
| WhatFontIs API, all images, among 1.2 million fonts (published by the set's authors) | 11,995 | 83.7% | 93.3% | 96.5% |

The rows are not the same images: WhatFontIs publishes only totals, and it indexes all 600 fonts, since the set was built from its catalogue. gpu-font searches 110,543 families, 371 of the set's fonts among them, 180 of those outside Google Fonts, most as one preview each with weight and style unknown.

The catalogs, not the model, set these figures. The seventh step's model reads 1.6 points above the sixth's on the same images against the same catalogs (43.6% against 42.0% on all 7,415), and the previous release's 44.2% final was read among 55,428 families with 41,113 of DaFont's: under this model DaFont's eight parts shipped then, grown to 49,449 faces, read 44.1% on all 7,335 images they index and 42.0% final on 4,355; all fourteen parts, 96,212 faces, read 43.6% and 41.2% while indexing 80 more images, and on the 7,335 images both scopes index they lose 3 hits of 3,233. Each DaFont face is one preview image, weighed in the ranking like a Chromium reference of a face with every case and size; the 50,000 such faces shipped since the student cost about eight points on the images every search shares, the sixth step's photographs gave back two, and the seventh step one and a half. The whole of DaFont ships, since the fonts people ask for are there ([DaFont requests](#results) below), and weighing single-image references by their evidence is the next step for the ranking.

By kind (gpu-font top-5, all 7,415): printed objects 51.0%, textures 45.2%, scenes 35.9%; easy 51.1%, medium 50.1%, hard 36.5%. Google families the model trained on read 47.1%, the style split's test families 39.9%, the 180 fonts of the other catalogs 41.5%. Median time 192 ms a photo, p95 274 ms, WebGPU on an M4 Max, against 65 ms among half the rows.

DaFont requests ([report](dafont.json)): 16.6% top-1, 41.1% top-5 on the 151 cropped requests, searched across all seven catalogs. The previous release read 24.5% and 53.6% with 41,113 DaFont faces; this model with the eight parts shipped then (49,449 faces) reads 19.9% and 47.0%, and with all fourteen 41.1%, while the answers it indexes rise from 774 to 1,017 of the 1,573 confirmed: nine of the 151 crops lost against 243 requests it can now answer at all (the crops were all made for answers the smaller scope held, so they overstate the loss). The student read 28.5% and 47.7%; the four file-built catalogs alone, before DaFont's own fonts were indexed, 34.4% and 55.0%. Each chatbot answered a different share, so gpu-font is scored beside it on exactly the requests it answered:

| On the same requests | Requests | Chatbot top-5 | gpu-font top-5 |
|---|---:|---:|---:|
| Gemini 2.5 Flash (preview, 2025) | 65 | 6.2% | 36.9% |
| GPT-4o-mini | 37 | 18.9% | 37.8% |

On every answered request, including answers gpu-font does not index, the chatbots reach 1.5% (753) and 2.2% (360); gpu-font's ceiling there is the 64.7% it indexes. The chatbots saw whole pictures; gpu-font saw a hand crop.

Finder set ([answers](finders.json)), gpu-font top-5: Google families 5 of 10, hard cases 2 of 10, scripts 5 of 6, one to three letters 1 of 4. Probes: the matched face has the query's style (upright, italic) and weight (300, 700), the 300 and the upright and italic probes first, the 700 fifth. Classics await a judgement of each look-alike. [finders.json](finders.json) keeps a place for each of the eight finders the page compares, with how it may be run: WhatFontIs through its API, Lens and gpu-font from a script, the other five by hand, as their terms or form require.

## Web screens

Lines from the top sites' homepages, as the browser drew them ([report](screens.json)); no crop was trained on. The label is the font file, so a page set in Arial or Helvetica is labelled with a font no catalog holds, and the exact-family rule counts the answer wrong even when it names a clone: Times crops come back as TeX Gyre Termes and Playfair Display, Hiragino as Gothic A1 and IBM Plex Sans JP. So the installed fonts, 8,503 crops of 38 names, are a floor, and the web fonts the pages load, 10,879 crops of 369 families, are the read:

| | Crops | Families | Top-1 | Top-5 | By family, top-1 | By family, top-5 |
|---|---:|---:|---:|---:|---:|---:|
| All | 19,382 | 397 | 4.3% | 13.3% | 10.8% | 25.4% |
| Web fonts | 10,879 | 369 | 5.7% | 19.6% | 11.4% | 26.7% |
| Installed fonts | 8,503 | 38 | 2.4% | 5.1% | 7.1% | 13.2% |
| Under 14 px | 4,883 | 218 | 2.1% | 8.0% | 4.0% | 15.5% |
| 14–19 px | 9,910 | 329 | 4.3% | 13.3% | 10.4% | 23.1% |
| 20–31 px | 3,112 | 248 | 7.6% | 21.7% | 12.5% | 29.4% |
| 32 px and over | 1,477 | 192 | 4.3% | 13.1% | 9.6% | 22.5% |

By family, each counted once, so Arial's 2,552 crops do not speak for the rest. Google families read 21.4% top-5 (9,594 crops), Adobe Fonts' 5.3% (7,693, most of them Helvetica, Arial and Times by name): Roboto 18.8% (2,013 crops), Inter 12.3%, Open Sans 9.6%, Poppins 15.7%, Montserrat 23.0%, Verdana 30.7%, Noto Sans 31.0%. Half the crops are 14 to 19 px, where the frozen benchmark reads 87% on clean renders of 5 to 10 letters; a line of body text at 16 px on a page, anti-aliased, with its words spaced, reads 13%. 758 crops (3.9%) came back blank and 121 low-contrast. Median 112 ms a crop, p95 143 ms, WebGPU on an M4 Max.

This is the lowest read of the set and the closest to what people bring. The whole gap is at the sizes the benchmark calls small, and the model has never trained on a browser's own text rendering at those sizes: only on canvas renders and photographs. Training on page renders at 12 to 20 px is the next step for it.

## Commons photographs

Photographs of typefaces in use from Wikimedia Commons ([report](commons.json)), the 1,138 of 48 typefaces whose name a shipped catalog holds and in which a line of text was found; none trained on. gpu-font puts the typeface in the top five for 2.4% of them (27 photos), first for 0.7%, by typeface 1.0%: Futura 18 of 327, Brush Script 3 of 25, Helvetica 0 of 313, Arial 0 of 65, ITC Benguiat 0 of 54. Median 201 ms a photo, p95 384 ms.

The read measures its label and its catalogs as much as the model. The category names the photo, not the line: a Futura plate's tallest line reads "Berthold AG Berlin", a street sign's the town's name in another face, and the line read is whatever lettering is largest. The catalog's namesake is often not the typeface: Futura is a font of that name in a GitHub repository, Helvetica and Arial are Adobe Fonts entries indexed from one specimen line each, Gill Sans a DaFont look-alike. And the photographs are signage in perspective, carved, cast and weathered, at every angle. What the set does give is a floor for photographs nobody framed for a benchmark, and a list of what such a photograph needs: a line chosen for its typeface, not its size, references of the real typeface with more than one line of evidence, and perspective corrected before the crop. Until then it is kept as a floor, not a gate.

## Where it fails

**The photo, not the font** ([controls](photo-controls.json), read again with the student). 199 development photos read three ways:

| The same word in the same face | Top-1 | Top-5 |
|---|---:|---:|
| Clean Chromium render, 64 px | 52.8% | 81.9% |
| Photo, binarized (Otsu) | 23.6% | 53.8% |
| Photo | 26.1% | 48.2% |

With the earlier encoder a threshold recovered half of a 45-point gap between photo and render (45.2% against 90.0%), so texture, coloured ink and uneven light passing through preparation into the windows cost most of it. The shipped model, trained on photographed surfaces and on the development photographs themselves, reads these 199 at 48.2%, and a threshold adds 5.5 points (53.8%): the surface is mostly learned away, and the 34 points to the clean render are the photograph's geometry and blur, which no threshold restores. Thresholding still costs clean renders (72.4% top-5 from 81.9%) and moves the forum crops, mostly flat graphics, from 41.1% to 40.4%. The clean render of these 199 now reads 81.9% against 73.4% for the sixth step: the same words in the same faces, so the seventh step reads rendered Google families better and photographs of them a little worse.

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
```

`.data/dafont/gpu-font-guesses.json` holds gpu-font's answers in the LLM benchmark's own format.
