# gpu-font

Find the font in an image, in your browser. Crop a line of text from a screenshot or a design and get the closest font families, with weight and style, from Google Fonts and other free catalogs.

[![Video: a crop of "NIGHT MARKET" matched to Oswald Medium, then Akshar Medium and Hubot Sans SemiBold](https://img.youtube.com/vi/vw1CAHE7Q8I/maxresdefault.jpg)](https://youtu.be/vw1CAHE7Q8I)

**[Try it](https://dy.github.io/gpu-font/)** · [Watch the video](https://youtu.be/vw1CAHE7Q8I)

- Runs on the user's device, on WebGPU or the CPU: no server, no API key, the image is never uploaded.
- Names the face, not only the family: Bold Italic, Light and so on.
- Searches 2,004 Google Fonts families, every weight and italic, or one of seven other catalogs: DaFont, Adobe Fonts, GitHub font repositories, Debian, Font Library, WhatFontIs and Other, 102,726 distinct families in all.
- Kinds of one family with identical letters fold into one row, in every catalog: IBM Plex Sans KR and IBM Plex Sans Arabic show under IBM Plex Sans, Khmer OS Fasthand with Khmer OS Freehand. A different font that borrows the letters, like Parastoo with Lora's Latin, keeps its own row.
- Finds a family by name anywhere in the ranking, at its rank for the crop.

It is experimental: on rendered crops of every Google Fonts family, on text it never trained on, the right family is in the top five 93% of the time and first 73%.

## Usage

```sh
npm install gpu-font
```

```js
import { createMatcher } from 'gpu-font'

const matcher = await createMatcher()   // Google Fonts

// imageData: a crop of text, such as canvas.getContext('2d').getImageData(x, y, width, height)
const [best] = await matcher.match(imageData)

best.face.family      // 'Playfair Display'
best.face.styleName   // 'Regular'
best.score            // 0.87
```

`match` returns families best first, or `[]` when the image holds no text. A crop of several lines is read line by line; a line that is not already two-tone is read as it stands and cleaned (its ink found by colour and drawn black on white), and a line set steeply on a slant is read turned level as well; each family keeps its best reading. Each match has:

- `face`: the closest face: `family`, `styleName`, `weight` (100–900), `style` (`normal` or `italic`), and `sourceUrl` outside Google Fonts.
- `score`: similarity from -1 to 1, higher is closer.
- `siblings`: names of same-named families with the same letters, folded into this one.

`matcher.threshold` is a score below which the crop's font is probably in no catalog, when the model ships one; it is `null` now. Whether a photographed font is in the catalog turned out not to be readable from its match ([measured](https://github.com/dy/gpu-font/blob/main/bench/photos.md#rejection)): the closest design comes back either way, and a low score means a far design, not an absent one.

`matcher.destroy()` releases the GPU.

The crop's own words rank better than any reference line. `createReader()` loads a 156 KB line reader; `rerank` sets the first 30 matches in a text, drawn by a function you supply for the faces you can load, and sorts them by the catalog score plus the match in that text ([measured](https://github.com/dy/gpu-font/blob/main/bench/reader.md): first place 26% → 44% on photographs):

```js
import { createMatcher, createReader, rerank } from 'gpu-font'

const matcher = await createMatcher(), reader = await createReader()
const matches = await matcher.match(imageData)
const text = reader.read(imageData)?.text ?? ''   // 'HUGGING'; or let the person type it
const ranked = await rerank(matches, text, { embedding: await matcher.embed(imageData), encoder: matcher.encoder, draw })
// draw(match, text, size) → ImageData of the match's face drawing `text` at `size` px, or null when you cannot load that face
```

Without a bundler, import it from a CDN:

```html
<script type="module">
  import { createMatcher } from 'https://cdn.jsdelivr.net/npm/gpu-font@0.4/src/match.mjs'
</script>
```

In Node, pass raw RGBA pixels, for example from [sharp](https://sharp.pixelplumbing.com):

```js
import sharp from 'sharp'

const { data, info } = await sharp('crop.png').ensureAlpha().raw().toBuffer({ resolveWithObject: true })
const matches = await matcher.match({ data, width: info.width, height: info.height })
```

## Catalogs

A matcher searches one catalog. A catalog holds font names, links and style vectors, not font files: get the fonts from their source.

```js
import { createMatcher, catalogs } from 'gpu-font'

const matcher = await createMatcher(catalogs.debian)
```

| `catalogs.` | Families | |
|---|---:|---|
| `google-fonts` (default) | 2,004 | [Google Fonts](https://fonts.google.com) |
| `debian` | 2,165 | [Debian's font packages](https://packages.debian.org/sid/fonts/) |
| `fontlibrary` | 1,221 | [Font Library](https://fontlibrary.org/) |
| `dafont` | 96,212 | [DaFont](https://www.dafont.com/), from its preview images: one face a family, weight and style unknown |
| `adobe-fonts` | 4,784 | [Adobe Fonts](https://fonts.adobe.com/), from its specimen pages: one face a family |
| `github` | 3,564 | Openly licensed font repositories on GitHub, from their files |
| `whatfontis` | 226 | [WhatFontIs](https://www.whatfontis.com/), from its own renders: the fonts of its benchmark no other catalog holds, one face a family |
| `pangram-pangram` | 201 | [Pangram Pangram](https://pangrampangram.com/), from its font files, by the foundry's permission: every cut and its variable twin, weight and style known |
| `other` | 576 | Sources under 100 families: Uncut, Use & Modify, Velvetyne, Fontshare, Fontsource, Bye Bye Binary, Collletttivo, Tunera, Latin Modern, TeX Gyre, Paper Mono and more |

DaFont, Adobe Fonts and WhatFontIs are indexed from the images their sites show, not from font files, and ship as names, links and style vectors, the first two by the maintainer's decision; Pangram Pangram ships the same way from its files, under the permission it gave to train on its library. Their terms are recorded in [bench/foundries.json](https://github.com/dy/gpu-font/blob/main/bench/foundries.json), and any of these catalogs comes down at its source's request.

A catalog built for this model also works from its URL: `createMatcher('https://example.com/my-catalog.json')`. To build one from your own font files, index them on the [Catalogs](https://dy.github.io/gpu-font/catalogs.html#my-fonts) page and press Download JSON.

A catalog fits only the model that built it: `createMatcher` refuses one whose `encoderSha256` differs. Each new model comes in a new minor version (0.5, 0.6…) with the shipped catalogs rebuilt, and the page always runs the latest. npm's default range, `^0.4.0`, stays within 0.4, so your catalog keeps working until you upgrade; then build it again.

## Offline

Everything the matcher needs ships in the package: the model, the catalogs and the modules, read from the package's own files, so `createMatcher()` works with no network once installed, in Node or bundled. The page needs the network only for previews, which it draws with fonts from Google Fonts or shows as small pictures, stored with the site or, for DaFont, loaded from DaFont; matching itself runs in the browser and sends nothing.

## Share and embed

The page's address holds what is on show. A font sample is its font, `?sample=`, and its text, `?text=`; any other source is `?style=`, the crop's style, the numbers the model reads from it, never its pixels; `?catalog=` is the catalog searched. Copy the address, or press Link under the matches, to share them. A link made with another model says so over the page's own sample.

`?catalog=` takes `all`, `google-fonts`, `dafont`, `adobe-fonts`, `github`, `debian`, `fontlibrary`, `other`, `my-fonts`, or one source inside Other, such as `collletttivo` or `fontshare`. With `embed`, the page shows the matcher alone, searching only that catalog, to put on your own site:

```html
<iframe src="https://dy.github.io/gpu-font/?catalog=collletttivo&embed" allow="clipboard-write" style="width: 100%; height: 960px; border: 0"></iframe>
```

## How good it is

On rendered crops of 5–10 characters from every Google Fonts family, on text the model never trained on ([method and full results](https://github.com/dy/gpu-font/blob/main/bench/style.md)):

| | |
|---|---:|
| Right family in the top five | 93% |
| Right family first | 73% |
| Top five: Latin, other scripts, Chinese | 94%, 85%, 68% |
| Top five at 16 px, at 32 px and larger | 88%, 95% |
| Weight of the matched face, 100–900 | off by 6 on average |
| Italic or upright | 99.8% right |

Families the model never trained on are found more often than Google's own: 751 held-out open families, searched with all nine catalogs at once (110,953 families) and scored on the exact family, 74% top five, against 71% for Google Fonts families under the same search ([other catalogs](https://github.com/dy/gpu-font/blob/main/bench/style.md#other-catalogs)).

On photographs of one word printed or painted on real surfaces ([WhatFontIs-Bench](https://github.com/dy/gpu-font/blob/main/bench/photos.md)), searched across all nine catalogs (111,000 families), 597 of the set's 600 fonts indexed: 49% top five and 68% top twenty on the 8,955 images of 448 fonts the model never trained on, 26% first, each crop read as the page reads it: line by line, each line as it stands and cleaned, a steep line turned level as well; WhatFontIs itself reads 93% on the set, which was built from its catalogue. On DaFont forum requests, cropped by hand, 48% top five, with two thirds of the forum's confirmed answers indexed. On lines cut from the top websites' homepages, labelled by the font the browser drew them with and never trained on, 29% top five, 40% for web fonts.

On 5,546 real "what font is this?" requests from the DaFont forum, held out from training ([forum threads](https://github.com/dy/gpu-font/blob/main/bench/photos.md#forum-threads)): 18.7% top five and 10.5% first, with seven answers in ten DaFont fonts the catalog knows by one preview line.

On Pangram Pangram's own 667 product pictures ([photos.md](https://github.com/dy/gpu-font/blob/main/bench/photos.md#pangram-pangrams-own-pictures)), the design is in the first ten for 48% of whole pictures and 47% of their largest lines, from 11% and 17%: the catalogs hold every named weight and style of a variable font, and a picture's lines are read one by one.

With the crop's own words ([reader.md](https://github.com/dy/gpu-font/blob/main/bench/reader.md)): on 400 of those photographs, held-out Google families, the page's second stage lifts first place from 31% to 46% and the top five from 56% to 72%, the words read by a 156 KB line reader (character error 11%) or typed.

Not yet: hand-drawn letters, icons, symbols and emoji, and matching capitals when a catalog holds only lowercase (57% top five).

## Speed and size

- `createMatcher()` downloads about 1.2 MB, compressed: the model (6-bit weights, 0.34 MB) and the Google Fonts catalog (4-bit vectors, 0.88 MB). DaFont's catalog adds 5.8 MB, GitHub 1.0 MB, Adobe Fonts 0.84 MB, Debian 0.37 MB, Font Library 0.32 MB, Pangram Pangram 0.31 MB, Other 0.18 MB and WhatFontIs 0.03 MB; the page's All downloads them all, about 9.7 MB.
- With WebGPU, a match takes 0.02 to 0.06 s by the machine's load. Reading a photograph's lines as they stand and cleaned adds about 0.2 s; a screenshot, two-tone, about 0.05. Without it, it runs on the CPU: about 1.1 s in Chromium and 1.1–1.7 s in Node. All measured on an Apple M4 Max.
- The line reader is 0.16 MB compressed and reads a line in about 0.07 s on the CPU; setting the first 30 matches in its text takes about 0.24 s on WebGPU once their fonts are loaded. Without WebGPU the page shows the words read but keeps the catalog's order: on the CPU the thirty faces at three sizes would take half a minute.

## Development

Running the demo, training the model and building catalogs: [development.md](https://github.com/dy/gpu-font/blob/main/development.md).

## Inspired by

[gpu-lexer](https://gpu-lexer.vercel.app/), [gpu-time](https://github.com/arikchakma/gpu-time), [gpu-query](https://github.com/safzanpirani/gpu-query).

## License

[MIT](https://github.com/dy/gpu-font/blob/main/LICENSE). This work is also offered in the spirit described by the [Krishnized license](https://github.com/krishnized/license), which does not alter the MIT terms.
