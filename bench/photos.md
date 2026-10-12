# Photos and real requests

The [frozen benchmark](style.md#frozen-benchmark) reads Chromium renders. The results below are the shipped model (`pangram-30k`, SHA-256 `6e35d208…`), which trained on the development photographs, the development sites' screens and the forum's development threads, so only the final part is a read; the failure analyses under them were made with the earlier encoder (`wider-20k`) and describe what goes wrong, with that model's numbers where they give any. People bring photos, screenshots and logos. Three sets measure that, each read as the page reads it (`scripts/photos.mjs`): every shipped catalog searched as one (the page's All), the crop at full resolution and read as the page reads a crop ([Reading a crop](#reading-a-crop): its lines, each as it stands and cleaned, a steep one turned level as well), rows folded as the page shows them. A result is right when the true family, or a family folded into its row, is among the first five rows; any weight counts.

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
| gpu-font, final | 8,955 | 26.1% | 49.2% | 68.0% |
| gpu-font, development (trained on since the sixth step) | 2,980 | 33.0% | 59.2% | 79.5% |
| gpu-font, all | 11,935 | 27.8% | 51.7% | 70.8% |
| WhatFontIs API, all images, among 1.2 million fonts (published by the set's authors) | 11,995 | 83.7% | 93.3% | 96.5% |

The rows are not the same images: WhatFontIs publishes only totals, and it indexes all 600 fonts, since the set was built from its catalogue. gpu-font searches 110,953 families, 597 of the set's 600 fonts among them, 406 of those outside Google Fonts, most as one to three preview lines each with weight and style unknown. A row of a family the teacher cannot tell from the true one counts as right, as on the rendered benchmark.

By the catalog that holds the font: Google Fonts 58.8% top five (3,820 images of 191 fonts), Adobe Fonts 55.9% (2,315 of 116), the WhatFontIs slice 48.1% (4,520 of 226, each font known by three lines of WhatFontIs's own render), DaFont 35.7% (1,220 of 61, each font one preview), GitHub 35.0% (60 of 3). What a catalog knows of a font decides the read as much as the model does.

Like for like, on the same 8,955 final images with twin credit for both and the same model: 0.4.0 read them at 43.1% top five, 61.3% top twenty and 21.3% first; reading each crop as it stands and cleaned ([Reading a crop](#reading-a-crop)) gives 49.2%, 68.0% and 26.1%, every catalog gaining. Over the eighth step's model (41.4%, 59.0%, 20.2%) the ninth gained on these images, and gave up on the Google families it trained on what cleaning now more than gives back. Twin credit itself is worth eight points of top five on the development photographs.

By kind (gpu-font top-5, all 11,935): printed objects 57.0%, textures 53.3%, scenes 45.5%; easy 55.2%, medium 54.4%, hard 48.6%. Google families the model trained on read 58.6%, the style split's test families 57.4%, the 406 fonts of the other catalogs 48.4%. Median time 289 ms a photo, p95 404 ms (300 of them timed on a free GPU, an M4 Max): finding the lines, cleaning and the second reading cost about 190 ms over the bare match of 0.4.0.

DaFont requests ([report](dafont.json)): 26.5% top-1, 48.3% top-5 on the 151 cropped requests, searched across all nine catalogs with twin credit, each crop read as it stands and cleaned (0.4.0, the same model: 24.5% and 41.1%; the eighth step 22.5% and 46.4%; the seventh 21.2% and 44.4%). The sixth step read 24.5% and 53.6% with 41,113 DaFont faces, when the answers indexed were 687 of the 1,573 confirmed; they are 1,017 now, and the crops were all made for answers the smaller scope held, so they overstate what the larger catalog cost. Each chatbot answered a different share, so gpu-font is scored beside it on exactly the requests it answered:

| On the same requests | Requests | Chatbot top-5 | gpu-font top-5 |
|---|---:|---:|---:|
| Gemini 2.5 Flash (preview, 2025) | 65 | 6.2% | 46.2% |
| GPT-4o-mini | 37 | 18.9% | 48.6% |

On every answered request, including answers gpu-font does not index, the chatbots reach 1.5% (753) and 2.2% (360); gpu-font's ceiling there is the 64.7% it indexes. The chatbots saw whole pictures; gpu-font saw a hand crop.

Finder set ([answers](finders.json)), gpu-font top-5: 17 of the 30 scored crops, 13 first (0.4.0: 18 and 11; the eighth step 18 and 13; the seventh 13 and 9). Probes: the upright probe first; the italic second, behind Faustina VF Beta, a GitHub copy of the same design whose italic instance the catalog now holds; the 300 probe second, behind a 400 face of a Pangram Pangram look-alike; the 700 probe's family not in the five, though the first face is a 700. Classics await a judgement of each look-alike. [finders.json](finders.json) keeps a place for each of the eight finders the page compares, with how it may be run: WhatFontIs through its API, Lens and gpu-font from a script, the other five by hand, as their terms or form require.

## Reading a crop

The model learned from clean renders, and a photograph reaches it as grey texture under uneven light, coloured letters on a coloured wall, or two lines where it expects one. Before the frozen preparation (`src/line.mjs`, which the model binds by hash), the page now reads a crop by its readings (`src/clean.mjs`):

- **Read as it stands, and cleaned.** Each line is matched as it is and, unless it is already two-tone, cleaned as well, each family keeping its better match: cleaning lifts photographs, and a screenshot with a picture behind its letters, which cleaning hurts, keeps its own reading. Cleaning finds the ink by colour, two means in RGB seeded from the border's colour and the pixel farthest from it, each pixel placed between them, Otsu's threshold, the ink within 0.15 of it keeping its grey: black on white with soft edges. A crop already two-tone goes through as it is: one whose pixels between ink and background are edges, an ink and a background pixel among the eight neighbours, as a renderer's anti-aliasing leaves them, but for 1% of the crop. A share of all pixels between would grow as text shrinks: small screen text is nearly all edge, and a first gate built so cleaned 83% of the screen crops and halved their read.
- **Cut into its lines.** Rows whose ink reaches a tenth of the busiest row's, split where a gap of columns wider than the line's height parts them; up to six lines, none under two fifths of the tallest, so the clipped edge of a line above or below is not one. Each line is matched, and each family keeps its best line.
- **Turned, when steep.** A line whose ink lies along an axis 8 to 45 degrees off level, and gathers into rows when turned by it, is read turned level as well as it stands. Not inside the preparation: a calligraphic Arabic line can lie diagonally while level (two of the catalog benchmark's 27,703 queries did), so both readings are matched and each family keeps the better.

The same model and catalogs throughout (the variable fonts as their named instances, below):

| | As it stands | Cleaned, when not two-tone | Both, each family at its better (shipped) |
|---|---:|---:|---:|
| WhatFontIs-Bench final, 8,955 photographs, top five | 42.5% | 48.1% | 49.2% |
| The same, first place | 21.0% | 26.1% | 26.1% |
| Controls: 199 development photographs, top five | 48.7% | 58.3% | 57.3% |
| Controls: the same words rendered clean | 78.9% | 78.9% | 78.9% |
| Forum threads, 5,546 held out, top five | 17.4% | 18.0% | 18.7% |
| Web screens, 1,200 held-out crops, top five | 28.3% | 26.9% | 28.1% |

Cleaning alone lifts photographs and costs a screenshot whose letters stand on a picture or a gradient: of the 1,200 screen crops it cleans 246, which read 13.4% as they stand and 6.5% cleaned. Read both ways, those crops read 12.6% and the photographs keep their gain. The crop's own embedding, which the model's heads and the same-text stage use, is its first line's cleaned reading when there is one. Reading a crop's lines one by one lifts Pangram Pangram's whole product pictures from 18.1% to 32.7% in the top ten, and telling the lines apart however tight the leading to 47.5% (below). The steep turn is tested on turned words; no set measures it alone yet.

## Pangram Pangram's own pictures

The foundry's 728 product pictures (the store's feed, kept in `.data/pangram`, never shipped): 667 of the 58 designs the catalog holds, the rest bundles. Each read whole, as a person dropping the picture would, and as the tallest line macOS Vision finds in it with a quarter of its height around (`scripts/pangram-specimens.mjs`); right when any family of the design is in the first ten. Mostly clean, designed images: an "Aa" cover, a stack of weights and italics, a word in a scene.

| In the first ten | Whole picture | Its tallest line |
|---|---:|---:|
| 0.4.0 | 10.5% | 17.1% |
| Every named instance of a variable font in the catalog | 18.4% | 45.9% |
| And the crop read by its lines, as it stands and cleaned, turned (this release) | 32.7% | 46.7% |
| And two lines never joined, however tight the leading | 47.5% | 47.1% |

The catalog had indexed each variable font at its file's default, PP Kyoto's Thin and PP Monument's Book, so a bold crop met only a thin row: `scripts/instances.py` now makes every named instance a static file (and the nine CSS weights an axis spans when the file names fewer), 5,971 of 349 files, and the file-built catalogs index each as a face: Pangram Pangram's 351 rows became 4,580. The six designs held out of training read 53.8% by their tallest line, the 52 trained on 46.1% (with the first cleaner; the shipped reading moves these by under a point). By length: "Aa" covers 46.0%, words of four to eight letters 34.3%, longer lines 55.3%. Ten designs are found from every picture, eighteen from none: the first stage still crowds a design out with look-alikes from 110,953 families, the next step's work.

The line finder joined a run of rows to its band when the gap was under a fifth of the band's height, and the band grew as it joined: on a tightly set picture the descender fragments between lines chained three lines into one band, the other lines fell under two fifths of it and were dropped, and 307 of the 461 pictures that are not covers were read whole, 9.4% of them in the first ten against 61.0% of the 154 read by their lines. A run is now measured by its tallest unbroken stretch and two runs each tall enough to be a line never join (`bands` in src/clean.mjs): 325 pictures read by their lines, and the whole pictures went from 32.7% to 47.5%. The single-line reads did not move: the finder set's 44 answers are the same, the same-text read's ranks the same, the final photographs 49.2% top five as before, the forum crops 48.3%, the forum threads 18.7%, the held-out web screens 29.1%, the Commons photographs 4.5%.

## Web screens

Lines from the top sites' homepages, as the browser drew them ([report](screens.json)): 185,903 crops from 35,738 sites at the read, 126,732 of them naming a family a catalog holds. Sites are split by the hash of their origin: nine in ten are development, whose crops train the model since the eighth step; the held-out tenth is the read. The label is the font file, so a page set in Arial or Helvetica is labelled with a font no catalog holds, and the exact-family rule counts the answer wrong even when it names a clone (a twin by the teacher's distance counts, as in every read): so the installed fonts are a floor, and the web fonts the pages load are the read.

| | Crops | Families | Top-1 | Top-5 | By family, top-1 | By family, top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Held-out sites | 12,447 | 373 | 12.7% | 29.0% | 16.3% | 32.9% |
| Development sites, trained on | 114,285 | 1,004 | 12.3% | 28.4% | 17.1% | 31.7% |
| Web fonts, all sites | 79,405 | 1,035 | 17.9% | 40.4% | 17.4% | 31.8% |
| Installed fonts, all sites | 47,327 | 55 | 3.1% | 8.4% | 9.1% | 21.2% |
| Under 14 px | 30,268 | 559 | 7.1% | 19.4% | 8.6% | 20.5% |
| 14–19 px | 64,471 | 820 | 12.9% | 30.0% | 16.6% | 31.3% |
| 20–31 px | 21,457 | 749 | 18.2% | 38.0% | 19.7% | 37.5% |
| 32 px and over | 10,536 | 641 | 12.2% | 26.0% | 19.3% | 32.5% |

By family, each counted once, so Arial's thousands of crops do not speak for the rest. The eighth step read 27.7% top five on the 3,503 held-out crops collected by then, 40.6% on web fonts and 19.0% under 14 px; the seventh 13.3% on 19,382 crops, with no twin credit and no site held out. The set has grown to three and a half times the eighth step's, and the held-out sites still read as the development ones do. Half the crops are 14 to 19 px, where the frozen benchmark reads 88% on clean renders of 5 to 10 letters; body text at those sizes on a page, anti-aliased, its words spaced, reads 30.0%. 0.4.0 read the held-out sites at 29.1% and web fonts at 40.3%: a screenshot is two-tone and reads as it did. Median 100 ms a crop, p95 224 ms (300 timed on a free GPU, an M4 Max).

## Commons photographs

Photographs of typefaces in use from Wikimedia Commons ([report](commons.json)), the 1,138 of 48 typefaces whose name a shipped catalog holds and in which a line of text was found; none trained on. gpu-font puts the typeface in the top five for 4.5% of them (51 photos), first for 1.3%, by typeface 1.8%: Futura 36 of 327, Helvetica 0 of 313, Arial 0 of 65, ITC Benguiat 1 of 54; 0.4.0 read 5.0%, the eighth step 3.8%, the seventh 2.4%.

The read measures its label and its catalogs as much as the model. The category names the photo, not the line: a Futura plate's tallest line reads "Berthold AG Berlin", a street sign's the town's name in another face, and the line read is whatever lettering is largest. The catalog's namesake is often not the typeface: Futura is a font of that name in a GitHub repository, Helvetica and Arial are Adobe Fonts entries indexed from one specimen line each, Gill Sans a DaFont look-alike. And the photographs are signage in perspective, carved, cast and weathered, at every angle. What the set does give is a floor for photographs nobody framed for a benchmark, and a list of what such a photograph needs: a line chosen for its typeface, not its size, references of the real typeface with more than one line of evidence, and perspective corrected before the crop. Until then it is kept as a floor, not a gate.

## Forum threads

The DaFont forum's identification requests, scraped whole (`scripts/dafont_forum.py`): 101,650 threads, 98,924 settled on a font the first post links to its page. Threads split by the hash of their id: nine in ten are development, which `train.style_forum` packs as training views; the tenth, never trained on, is this read (`scripts/forum.mjs`, [forum.json](forum.json)). The picture is the asker's; the line is the one the answerer previewed the font with (the link's `text=`), else the tallest confident one macOS Vision found (`scripts/forum_boxes.py`), with a quarter of its height around it; the answer counts when a shipped catalog holds it, a folded family or a twin included. Of the 9,933 held-out answers, 2,472 are in no shipped catalog, 451 pictures gave no readable line and 1,464 arrived after the boxing. Max Halford's older threads above are a separate read, cropped by hand.

| Ninth step, all catalogs, the crop read as it stands and cleaned | Threads | Families | First | Top five | Top twenty |
|---|---:|---:|---:|---:|---:|
| Every held-out thread indexed | 5,546 | 3,159 | 10.5% | 18.7% | 27.9% |
| Answers on Google Fonts | 689 | 277 | 20.9% | 35.6% | |
| Answers on DaFont | 3,863 | 2,553 | 9.7% | 16.8% | |
| Answers on Adobe Fonts | 747 | 264 | 6.7% | 13.2% | |
| Line by the previewed text | 3,260 | 2,242 | | 20.0% | |
| Line by height alone | 2,286 | 977 | | 16.8% | |
| Line under 24 px tall | 125 | 111 | | 12.8% | |

By family, each counted once: 11.3% first, 19.3% top five. 0.4.0 read 9.9% and 17.5% on the same threads with the same model, the crop as it was; the eighth step 9.2% and 17.2%. With three preview lines a DaFont face instead of one, the ninth read 10.8% and 19.3% here, for 4.2 MB more of catalog and a point and a half less on the final photographs ([style.md](style.md), Catalog); one line ships. This is the question people actually ask, and the hardest read here: seven answers in ten are DaFont fonts the catalog knows by one preview line, the pictures carry effects, several fonts and small text, and the line is a machine's guess at what was asked about.

## Where it fails

**The photo, not the font** ([controls](photo-controls.json), read as the page reads a crop). 199 development photos, the shipped model:

| The same word in the same face | Top-1 | Top-5 |
|---|---:|---:|
| Clean Chromium render, 64 px | 54.8% | 78.9% |
| Photo, as the page now reads it (as it stands and cleaned) | 29.1% | 57.3% |
| Photo, through the soft colour cleaner whatever its tone | 28.1% | 57.8% |
| Photo, binarized by Otsu's threshold on luminance | 26.6% | 53.3% |
| Photo, as it is (0.4.0) | 23.1% | 48.7% |

With the earlier encoder a threshold recovered half of a 45-point gap between photo and render (45.2% against 90.0%). The models since learned most of the surface away, and the cleaner takes the rest: ink found by colour, not luminance, gains where Otsu on luminance gains least, and keeping the edge's greys (the soft band) holds what a hard threshold loses. Twenty points remain to the clean render: the photograph's geometry, blur and the letters' own wear, which no cleaning restores. On the 151 cropped forum requests the page's reading gives 46.4% against 39.7% as they are.

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

**The crop's own words, read.** The second stage is built and measured ([reader.md](reader.md), bench/sametext.json): a 156 KB line reader reads the crop, the first 30 rows are set in its text where the page can load the font, and the match counts with the catalog's. On 400 final photographs of 42 held-out Google families, all nine catalogs, the crop read as the page reads it: first place 31.0% → 45.8%, top five 56.0% → 71.8%, a point under what the true text gives in the top five. The page shows the words read in every preview and re-ranks when they are corrected.

So, in the order of what each is worth:

- The first stage's depth: on Pangram Pangram's pictures eighteen designs of 54 are found from no picture, crowded out by look-alikes. What crowds them (`scripts/pangram-drawn.mjs`, an analysis whose rows stay in `.data/pangram`): every picture behind the eighteen is the product's cover, a white "Aa" on black, so no training on photographs or cleaned views could have reached them. Among Pangram's own 4,580 rows the design is in the first ten for 84.4% of the tallest lines, among all 110,953 families for 46.7%; of the rows above a missed cover's design, 30% are Adobe's (4% of the families) and 31% DaFont's (87%). With Pangram's faces drawn for the same-text stage, as Google's are, the tallest lines read 30.1% first and 50.3% in the first ten over 30 rows with the words the reader read, 31.3% and 57.2% over 100 rows with the words Vision read, from 12.5% and 46.7%; the covers alone 53.2% first from 7.6%; 42 and 49 designs found from some picture, from 36. Pangram's font host answers any origin, so the page could draw them; the foundry's permission covered collecting and training, and loading from its host is a question to ask it. A hub correction was tried on the same queries and rejected: taking from each family's score its mean cosine to its ten nearest background queries (CSLS) lowers the tallest lines' first ten from 45.6% to 26.2% at a quarter weight and to 8.0% at half, with 1,500 catalog rows as the background, and as much with the queries themselves; the families nearest everything are Pangram's own, which carry a row for every weight and both cases, so the correction punishes the catalog it was meant to help (`.data/pangram/drawn-lines.json` holds the embeddings). What does reach the covers is a row of their own: `scripts/catalog-files.mjs --short` adds the cover line "Aa" (src/references.mjs SHORT, at the three sizes, as the case lines are) as a third row of every face with both cases, 13,740 rows of 4,580 faces against 9,160, 2.30 MB against 2.03 MB on disk, and read in the shipped catalog's place (`pangram-drawn.mjs --pangram`) the 206 covers go from 46.1% to 98.1% in the first ten and from 9.2% to 75.7% first, among all 110,953 families, with no words set; 56 of the 58 designs with a cover are found from it, the two left being Bitmap Fonts and Editorial Old, whose cover is an italic the catalog lacks. The tallest lines, 614, go from 47.1% to 62.4% in the first ten and from 13.2% to 32.1% first, the lines that are not covers from 46.5% to 48.3% (50 of them rank better, 21 worse, by a place or two); 53 of the 54 designs with a line are found from one, from 36. Only Pangram's catalog carries the row in this read, so its covers meet no other catalog's "Aa"; the fair measure is the row in every file-built catalog, and the price is the rows. Then a run against the look-alikes of the other catalogs, judged on a read at full catalog depth as the page reads. The tenth step tried hard negatives from the model's own confusions, cleaned views and lines along circles in one run and did not pass the ninth on rendered text or raw photographs ([style.md](style.md#tenth-step-rejected-cleaned-views-circles-the-models-own-confusions)); the next run should select on a read that cleans its photographs, since that is what it was trained for, and try the parts one at a time.
- Three preview lines a DaFont face read 19.3% on the forum threads against 17.5%, for 4.2 MB more of catalog and a point and a half less on the final photographs; a per-face choice (several lines only where one is not enough) could keep the gain without the cost.
- The same-text stage for the catalogs built from previews, whose faces the page cannot draw: a text-aware comparison, or the reader's letter positions matched glyph by glyph.
- The styles a store sells that its web fonts do not carry (PP Editorial Old's italics, PP Charlevoix's Extralight), letterless families out of every catalog, and forum posts newer than the scrape as a fresh final read.

## Reproduce

With `npm run demo` running and WebGPU in Chromium:

```sh
node scripts/whatfontis.mjs      # downloads the set (1.9 GB) into .data/whatfontis, writes bench/whatfontis.json
node scripts/dafont.mjs          # downloads the scrape and the 153 pictures, writes bench/dafont.json
node scripts/finders.mjs         # writes the 44 crops to .data/finders, gpu-font's answers to bench/finders.json
node scripts/photo-controls.mjs  # writes bench/photo-controls.json
node scripts/pangram-specimens.mjs   # the foundry's product pictures of .data/pangram, whole and by their tallest line (scripts/text_boxes.py boxes them)
node scripts/whatfontis.mjs --clean "colour soft"   # any read with a candidate cleaner of scripts/clean-candidates.mjs, or none; the report goes to .data/style
node scripts/screens.mjs         # the web screens read of .data/screens (scripts/web_screens.mjs collects them), writes bench/screens.json
node scripts/commons.mjs         # the Commons read of .data/photos/commons (scripts/commons_typefaces.py collects them), writes bench/commons.json
node scripts/python.mjs -m train.style_photos   # packs the Google-family pictures for train.style as the page reads a crop (as it stands and cleaned): photos-development, tracked at every checkpoint, and photos-final
node scripts/python.mjs -m train.style photos --run pangram-30k,photo-30k   # two runs' exported models on the packed photographs, development and final, against the Google references: a comparison, not the page's read
```

`.data/dafont/gpu-font-guesses.json` holds gpu-font's answers in the LLM benchmark's own format.
