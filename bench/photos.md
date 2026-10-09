# Photos and real requests

The [frozen benchmark](style.md#frozen-benchmark) reads Chromium renders. The results below are the shipped model (`pangram-30k`, SHA-256 `6e35d208…`), which trained on the development photographs, the development sites' screens and the forum's development threads, so only the final part is a read; the failure analyses under them were made with the earlier encoder (`wider-20k`) and describe what goes wrong, with that model's numbers where they give any. People bring photos, screenshots and logos. Three sets measure that, each read as the page reads it (`scripts/photos.mjs`): every shipped catalog searched as one (the page's All), the crop at full resolution, rows folded as the page shows them. A result is right when the true family, or a family folded into its row, is among the first five rows; any weight counts.

| Set | What it holds | gpu-font can answer | Licence | Crop |
|---|---|---|---|---|
| [WhatFontIs-Bench v1.0](https://github.com/whatfontis/WhatFontIs-Bench) | 11,995 photos of one word, 7–12 letters, printed or painted on real surfaces, walls and objects; 600 fonts, 20 images each | 11,935 images of 597 fonts: 191 Google Fonts families, 116 of Adobe Fonts (69 found by the store words in their titles), 61 of DaFont, 3 of GitHub, and the 226 no other catalog held, from WhatFontIs's own renders; 3 fonts are gone from every source | CC BY 4.0, backgrounds CC0 1.0 | the set's own `crop_box` |
| DaFont forum requests, from [Max Halford's LLM benchmark](https://github.com/MaxHalford/llm-font-recognition) | 4,887 "what font is this?" posts, 1,573 with an answer the forum confirmed | 1,017 answers (64.7%), 775 of them DaFont's own fonts; 151 cropped, the rest not yet | pictures belong to their posters: links only, kept in `.data/` | by hand, [dafont-crops.json](dafont-crops.json) |
| Finder set, [issue #1](https://github.com/dy/gpu-font/issues/1) | 44 crops: 10 Google families, 10 commercial classics, 10 hard cases, 6 scripts, 4 of one to three letters, 4 probe images | 30 scored crops; classics judged as substitutes | CC BY 4.0 photos and renders of OFL fonts | fixed PNG per crop |
| Web screens, `scripts/web_screens.mjs` | lines cut from the homepages of the Chrome UX Report's top sites where robots.txt allows, each labelled by the font file Chromium drew it with; 185,903 crops from 35,738 sites at the read | 126,732 crops of 1,060 families; 2,788 with no usable font name, 56,383 in no catalog | other sites' designs: local only | the collector's line crop |
| Wikimedia Commons, typefaces by name, `scripts/commons_typefaces.py` | photographs of signs, packaging and print filed under a typeface's category; 2,648 photos of 102 typefaces | 1,138 photos of 48 typefaces with a line of text found | each photo's own free licence | the tallest line of text found (`scripts/text_boxes.py`) |

## Results

WhatFontIs-Bench ([report](whatfontis.json)). Final: fonts that never entered training or checkpoint selection (the style split's test role, and every catalog besides Google Fonts). Development: the rest.

| | Images | Top-1 | Top-5 | Top-20 |
|---|---:|---:|---:|---:|
| gpu-font, final | 8,955 | 21.3% | 43.1% | 61.3% |
| gpu-font, development (trained on since the sixth step) | 2,980 | 26.2% | 51.2% | 71.5% |
| gpu-font, all | 11,935 | 22.5% | 45.1% | 63.8% |
| WhatFontIs API, all images, among 1.2 million fonts (published by the set's authors) | 11,995 | 83.7% | 93.3% | 96.5% |

The rows are not the same images: WhatFontIs publishes only totals, and it indexes all 600 fonts, since the set was built from its catalogue. gpu-font searches 110,953 families, 597 of the set's 600 fonts among them, 406 of those outside Google Fonts, most as one to three preview lines each with weight and style unknown. A row of a family the teacher cannot tell from the true one counts as right, as on the rendered benchmark.

By the catalog that holds the font: Google Fonts 51.0% top five (3,820 images of 191 fonts), Adobe Fonts 49.9% (2,315 of 116), the WhatFontIs slice 41.7% (4,520 of 226, each font known by three lines of WhatFontIs's own render), GitHub 40.0% (60 of 3), DaFont 30.4% (1,220 of 61, each font one preview). The ninth step gave up four points on Google families' photographs (55.3% for the eighth) and gained on every other catalog's, the final set most. What a catalog knows of a font decides the read as much as the model does.

Like for like, on the same 8,955 final images with twin credit for both, the ninth step reads 43.1% top five against the eighth's 41.4%, 61.3% against 59.0% top twenty and 21.3% against 20.2% first; on all 11,935 images both read 45.1%, the ninth having given up the eighth's edge on the Google families it trained on. Twin credit itself is worth eight points of top five on the development photographs.

By kind (gpu-font top-5, all 11,935): printed objects 53.2%, textures 47.3%, scenes 36.1%; easy 53.0%, medium 49.1%, hard 39.2%. Google families the model trained on read 51.3%, the style split's test families 50.1%, the 406 fonts of the other catalogs 42.3%. Median time 98 ms a photo, p95 125 ms, WebGPU on an M4 Max, the machine otherwise idle.

DaFont requests ([report](dafont.json)): 24.5% top-1, 41.1% top-5 on the 151 cropped requests, searched across all nine catalogs with twin credit (the eighth step by the same rule: 22.5% and 46.4%; the seventh 21.2% and 44.4%). The sixth step read 24.5% and 53.6% with 41,113 DaFont faces, when the answers indexed were 687 of the 1,573 confirmed; they are 1,017 now, and the crops were all made for answers the smaller scope held, so they overstate what the larger catalog cost. Each chatbot answered a different share, so gpu-font is scored beside it on exactly the requests it answered:

| On the same requests | Requests | Chatbot top-5 | gpu-font top-5 |
|---|---:|---:|---:|
| Gemini 2.5 Flash (preview, 2025) | 65 | 6.2% | 38.5% |
| GPT-4o-mini | 37 | 18.9% | 43.2% |

On every answered request, including answers gpu-font does not index, the chatbots reach 1.5% (753) and 2.2% (360); gpu-font's ceiling there is the 64.7% it indexes. The chatbots saw whole pictures; gpu-font saw a hand crop.

Finder set ([answers](finders.json)), gpu-font top-5: 18 of the 30 scored crops, 11 first (the eighth step: 18 and 13; the seventh 13 and 9). Probes: the upright and italic probes first, each matched in its posture; the 300 probe second, behind a 400 face of a Pangram Pangram look-alike the catalog now holds; the 700 probe's family not in the five, though the first face is a 700. Classics await a judgement of each look-alike. [finders.json](finders.json) keeps a place for each of the eight finders the page compares, with how it may be run: WhatFontIs through its API, Lens and gpu-font from a script, the other five by hand, as their terms or form require.

## Web screens

Lines from the top sites' homepages, as the browser drew them ([report](screens.json)): 185,903 crops from 35,738 sites at the read, 126,732 of them naming a family a catalog holds. Sites are split by the hash of their origin: nine in ten are development, whose crops train the model since the eighth step; the held-out tenth is the read. The label is the font file, so a page set in Arial or Helvetica is labelled with a font no catalog holds, and the exact-family rule counts the answer wrong even when it names a clone (a twin by the teacher's distance counts, as in every read): so the installed fonts are a floor, and the web fonts the pages load are the read.

| | Crops | Families | Top-1 | Top-5 | By family, top-1 | By family, top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Held-out sites | 12,447 | 373 | 12.9% | 29.1% | 16.4% | 33.4% |
| Development sites, trained on | 114,285 | 1,004 | 12.3% | 28.5% | 17.0% | 31.5% |
| Web fonts, all sites | 79,405 | 1,035 | 17.9% | 40.3% | 17.2% | 31.5% |
| Installed fonts, all sites | 47,327 | 55 | 3.1% | 8.8% | 9.6% | 22.4% |
| Under 14 px | 30,268 | 559 | 7.3% | 19.9% | 8.6% | 20.7% |
| 14–19 px | 64,471 | 820 | 13.0% | 30.3% | 16.7% | 31.7% |
| 20–31 px | 21,457 | 749 | 18.0% | 37.5% | 19.8% | 36.5% |
| 32 px and over | 10,536 | 641 | 11.7% | 24.6% | 17.7% | 29.7% |

By family, each counted once, so Arial's thousands of crops do not speak for the rest. The eighth step read 27.7% top five on the 3,503 held-out crops collected by then, 40.6% on web fonts and 19.0% under 14 px; the seventh 13.3% on 19,382 crops, with no twin credit and no site held out. The set has grown to three and a half times the eighth step's, and the held-out sites still read as the development ones do. Half the crops are 14 to 19 px, where the frozen benchmark reads 88% on clean renders of 5 to 10 letters; body text at those sizes on a page, anti-aliased, its words spaced, reads 30.3%. Median 81 ms a crop, p95 96 ms, WebGPU on an M4 Max.

## Commons photographs

Photographs of typefaces in use from Wikimedia Commons ([report](commons.json)), the 1,138 of 48 typefaces whose name a shipped catalog holds and in which a line of text was found; none trained on. gpu-font puts the typeface in the top five for 5.0% of them (57 photos), first for 1.0%, by typeface 1.7%: Futura 42 of 327, Helvetica 2 of 313, Arial 0 of 65, ITC Benguiat 0 of 54; the eighth step read 3.8%, the seventh 2.4%.

The read measures its label and its catalogs as much as the model. The category names the photo, not the line: a Futura plate's tallest line reads "Berthold AG Berlin", a street sign's the town's name in another face, and the line read is whatever lettering is largest. The catalog's namesake is often not the typeface: Futura is a font of that name in a GitHub repository, Helvetica and Arial are Adobe Fonts entries indexed from one specimen line each, Gill Sans a DaFont look-alike. And the photographs are signage in perspective, carved, cast and weathered, at every angle. What the set does give is a floor for photographs nobody framed for a benchmark, and a list of what such a photograph needs: a line chosen for its typeface, not its size, references of the real typeface with more than one line of evidence, and perspective corrected before the crop. Until then it is kept as a floor, not a gate.

## Forum threads

The DaFont forum's identification requests, scraped whole (`scripts/dafont_forum.py`): 101,650 threads, 98,924 settled on a font the first post links to its page. Threads split by the hash of their id: nine in ten are development, which `train.style_forum` packs as training views; the tenth, never trained on, is this read (`scripts/forum.mjs`, [forum.json](forum.json)). The picture is the asker's; the line is the one the answerer previewed the font with (the link's `text=`), else the tallest confident one macOS Vision found (`scripts/forum_boxes.py`), with a quarter of its height around it; the answer counts when a shipped catalog holds it, a folded family or a twin included. Of the 9,933 held-out answers, 2,472 are in no shipped catalog, 451 pictures gave no readable line and 1,464 arrived after the boxing. Max Halford's older threads above are a separate read, cropped by hand.

| Ninth step, all catalogs | Threads | Families | First | Top five | Top twenty |
|---|---:|---:|---:|---:|---:|
| Every held-out thread indexed | 5,546 | 3,159 | 9.9% | 17.5% | 26.1% |
| Answers on Google Fonts | 689 | 277 | 20.3% | 33.2% | |
| Answers on DaFont | 3,863 | 2,553 | 9.0% | 15.8% | |
| Answers on Adobe Fonts | 747 | 264 | 6.6% | 11.8% | |
| Line by the previewed text | 3,260 | 2,242 | | 18.8% | |
| Line by height alone | 2,286 | 977 | | 15.6% | |
| Line under 24 px tall | 125 | 111 | | 12.0% | |

By family, each counted once: 10.4% first, 18.0% top five; 98 ms a thread at the median. The eighth step read 9.2% and 17.2% on the same threads. With three preview lines a DaFont face instead of one, the ninth read 10.8% and 19.3% here, for 4.2 MB more of catalog and a point and a half less on the final photographs ([style.md](style.md), Catalog); one line ships. This is the question people actually ask, and the hardest read here: seven answers in ten are DaFont fonts the catalog knows by one preview line, the pictures carry effects, several fonts and small text, and the line is a machine's guess at what was asked about.

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

Three measurements on the eighth step's model (`photo-30k`) say where the photographs are lost. The first two are analyses, not reports: their rows stay in `.data/style/photo-30k/`.

**How deep the true font sits.** 1,200 final photographs, all eight catalogs, 110,752 families:

| Within the first | All | Google Fonts | Adobe Fonts | WhatFontIs | DaFont |
|---|---:|---:|---:|---:|---:|
| 1 | 19.4% | 21.2% | 25.5% | 17.6% | 13.7% |
| 5 | 40.9% | 41.6% | 51.3% | 40.0% | 24.8% |
| 20 | 59.6% | 68.1% | 67.4% | 60.0% | 37.9% |
| 100 | 75.8% | 81.4% | 79.2% | 78.9% | 54.0% |
| 1,000 | 87.1% | 90.3% | 88.3% | 90.3% | 69.6% |
| 5,000 | 92.3% | 97.3% | 95.0% | 93.7% | 77.6% |

WhatFontIs reads 93.3% in its top five. Here the true font is within the first 5,000 rows 92.3% of the time, so no second stage alone reaches it: a perfect one over 200 candidates stops at 80%. The first stage has to find the font, and a fifth of DaFont's are never near the top, known as they are by one preview line.

**What the references' text costs.** 128 final photographs of 40 held-out Google families, searched among Google's 2,004: the catalog's rows, then every face of the 200 best families set clean in the photograph's own word and embedded by the same model:

| Ranking by | First | Top five | Top twenty |
|---|---:|---:|---:|
| The catalog's rows | 43.8% | 75.8% | 87.5% |
| References in the photograph's word | 61.7% | 82.0% | 89.8% |
| Both, scores added | 64.1% | 83.6% | 88.3% |

Eighteen points of first place and eight of the top five from knowing the text, with the model unchanged and no bytes added: the browser already loads these fonts for the previews.

**What a catalog knows of a font.** Top five by the catalog holding it: Google Fonts, every case and size, 55.3%; Adobe Fonts, several specimen lines, 49.4%; the WhatFontIs slice, three clean lines, 39.1%; DaFont, one preview, 27.8%.

**The crop's own words, read.** The second stage is built and measured ([reader.md](reader.md), bench/sametext.json): a 156 KB line reader reads the crop, the first 30 rows are set in its text where the page can load the font, and the match counts with the catalog's. On 400 final photographs of 42 held-out Google families, all nine catalogs: first place 26.0% → 43.8%, top five 49.5% → 65.8%, within a point of what the true text gives. The page shows the words read in every preview and re-ranks when they are corrected.

So, in the order of what each is worth:

- A cleaner input: the controls read says the same crops binarized read 53.3% top five against 48.7% raw, with the model unchanged, and a photographed wall still reaches the model as grey texture. A preparation that flattens the background, keeps the dominant ink colour and finds the lines of a multi-line crop (a projection profile; the tallest line, or each line matched alone) is a change of the pinned preparation: benchmarks re-pinned, catalogs rebuilt, the model trained on it.
- Three preview lines a DaFont face read 19.3% on the forum threads against 17.5%, for 4.2 MB more of catalog and a point and a half less on the final photographs; a per-face choice (several lines only where one is not enough) could keep the gain without the cost.
- The first stage's depth: hard negatives from its own confusions, and the families it puts beyond the first hundred; the ninth step gave up four points on Google families' photographs, and five on the 151 cropped forum requests, while gaining on the final photographs, the held-out threads and the open families.
- The same-text stage for the catalogs built from previews, whose faces the page cannot draw: a text-aware comparison, or the reader's letter positions matched glyph by glyph.
- Letterless families out of every catalog; forum posts newer than the scrape as a fresh final read.

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
node scripts/python.mjs -m train.style photos --run pangram-30k,photo-30k   # two runs' exported models on the packed photographs, development and final, against the Google references: a comparison, not the page's read
```

`.data/dafont/gpu-font-guesses.json` holds gpu-font's answers in the LLM benchmark's own format.
