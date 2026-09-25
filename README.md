# gpu-font

Find the font in an image, in your browser. Crop a line of text from a screenshot or a design and get the closest font families, with weight and style, from Google Fonts and other free catalogs.

[![A crop of "Quiet rivers flow" matched to Lora Regular, then Sumana](https://raw.githubusercontent.com/dy/gpu-font/main/og.png)](https://dy.github.io/gpu-font/)

**[Try it](https://dy.github.io/gpu-font/)**

- Runs on the user's device, on WebGPU or the CPU: no server, no API key, the image is never uploaded.
- Names the face, not only the family: Bold Italic, Light and so on.
- Searches 2,004 Google Fonts families, every weight and italic, or one of five other catalogs: DaFont, Adobe Fonts, Debian, Font Library and Other, 44,091 family names in all.
- Kinds of one family with identical letters fold into one row: IBM Plex Sans KR and IBM Plex Sans Arabic show under IBM Plex Sans. A different font that borrows the letters, like Parastoo with Lora's Latin, keeps its own row.

It is experimental: on rendered crops of every Google Fonts family, on text it never trained on, the right family is in the top five 91% of the time and first 70%.

## Usage

```sh
npm install gpu-font
```

```js
import { createMatcher } from 'gpu-font'

const matcher = await createMatcher()   // Google Fonts

// imageData: one line of text, such as canvas.getContext('2d').getImageData(x, y, width, height)
const [best] = await matcher.match(imageData)

best.face.family      // 'Playfair Display'
best.face.styleName   // 'Regular'
best.score            // 0.87
```

`match` returns families best first, or `[]` when the image holds no text. Each match has:

- `face`: the closest face: `family`, `styleName`, `weight` (100–900), `style` (`normal` or `italic`), and `sourceUrl` outside Google Fonts.
- `score`: similarity from -1 to 1, higher is closer.
- `siblings`: names of same-named families with the same letters, folded into this one.

`matcher.threshold` is a score below which the crop's font is probably in no catalog, when the model ships one; it is `null` now. Whether a photographed font is in the catalog turned out not to be readable from its match ([measured](https://github.com/dy/gpu-font/blob/main/bench/photos.md#rejection)): the closest design comes back either way, and a low score means a far design, not an absent one.

`matcher.destroy()` releases the GPU.

Without a bundler, import it from a CDN:

```html
<script type="module">
  import { createMatcher } from 'https://cdn.jsdelivr.net/npm/gpu-font@0.1/src/match.mjs'
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
| `debian` | 2,174 | [Debian's font packages](https://packages.debian.org/sid/fonts/) |
| `fontlibrary` | 1,226 | [Font Library](https://fontlibrary.org/) |
| `dafont` | 34,975 | [DaFont](https://www.dafont.com/), from its preview images: one face a family, weight and style unknown |
| `adobe-fonts` | 4,784 | [Adobe Fonts](https://fonts.adobe.com/), from its specimen pages: one face a family |
| `other` | 379 | Sources under 100 families: Fontshare, Uncut, Velvetyne, Fontsource, Collletttivo, Latin Modern, TeX Gyre and more |

DaFont and Adobe Fonts are indexed from the images their sites show, not from font files, and ship as names, links and style vectors by the maintainer's decision; their terms are recorded in [bench/foundries.json](https://github.com/dy/gpu-font/blob/main/bench/foundries.json), and either catalog comes down at its source's request.

A catalog built for this model also works from its URL: `createMatcher('https://example.com/my-catalog.json')`. To build one from your own font files, index them on the [Catalogs](https://dy.github.io/gpu-font/catalogs.html#my-fonts) page and press Download JSON.

A catalog fits only the model that built it: `createMatcher` refuses one whose `encoderSha256` differs. Each new model comes in a new minor version (0.2, 0.3…) with the shipped catalogs rebuilt, and the page always runs the latest. npm's default range, `^0.1.0`, stays within 0.1, so your catalog keeps working until you upgrade; then build it again.

## Offline

Everything the matcher needs ships in the package: the model, the catalogs and the modules, read from the package's own files, so `createMatcher()` works with no network once installed, in Node or bundled. The page needs the network only for previews, which it draws with fonts from Google Fonts; matching itself runs in the browser and sends nothing.

## Share and embed

The page's address holds the matches on show: `?style=` is the crop's style, the 128 numbers the model reads from it, never its pixels, and `?catalog=` the catalog searched. Copy the address, or press Link under the matches, to share them. A link made with another model says so instead of showing matches.

`?catalog=` takes `all`, `google-fonts`, `debian`, `fontlibrary`, `dafont`, `adobe-fonts`, `other`, `my-fonts`, or one source inside Other, such as `collletttivo` or `fontshare`. With `embed`, the page shows the matcher alone, searching only that catalog, to put on your own site:

```html
<iframe src="https://dy.github.io/gpu-font/?catalog=collletttivo&embed" allow="clipboard-write" style="width: 100%; height: 960px; border: 0"></iframe>
```

## How good it is

On rendered crops of 5–10 characters from every Google Fonts family, on text the model never trained on ([method and full results](https://github.com/dy/gpu-font/blob/main/bench/style.md)):

| | |
|---|---:|
| Right family in the top five | 91% |
| Right family first | 70% |
| Top five: Latin, other scripts, Chinese | 92%, 83%, 60% |
| Top five at 16 px, at 32 px and larger | 85%, 92% |
| Weight of the matched face, 100–900 | off by 6 on average |
| Italic or upright | 99.8% right |

Families the model never trained on are found as often as Google's own: 3,563 families from the file-built catalogs, searched with all six catalogs at once (45,542 families) and scored on the exact family, 74% top five, against 73% for Google Fonts families under the same search; with the four free catalogs alone, 77% and 77% ([other catalogs](https://github.com/dy/gpu-font/blob/main/bench/style.md#other-catalogs)).

On photographs of one word printed or painted on real surfaces ([WhatFontIs-Bench](https://github.com/dy/gpu-font/blob/main/bench/photos.md)), searched across all six catalogs: 44% top five and 62% top twenty on 6,775 images of 339 fonts; with the four free catalogs alone, 53% and 70% on the 3,960 images of the 198 fonts they hold. It does best on one line in one font, several letters, on a plain background.

Not yet: hand-drawn letters, icons, symbols and emoji, and matching capitals when a catalog holds only lowercase (55% top five).

## Speed and size

- `createMatcher()` downloads about 1.2 MB, compressed: the model (6-bit weights, 0.35 MB) and the Google Fonts catalog (4-bit vectors, 0.85 MB). DaFont's catalog adds 2.3 MB, Adobe Fonts 0.94 MB, Debian 0.29 MB, Font Library 0.27 MB, and Other 0.11 MB; the page's All downloads them all, about 4.7 MB.
- With WebGPU, a match takes about 0.05 s. Without it, it runs on the CPU: about 1.2 s in Chromium and 1.1–1.7 s in Node. All measured on an Apple M4 Max.

## Development

Running the demo, training the model and building catalogs: [development.md](https://github.com/dy/gpu-font/blob/main/development.md).

## Inspired by

[gpu-lexer](https://gpu-lexer.vercel.app/), [gpu-time](https://github.com/arikchakma/gpu-time), [gpu-query](https://github.com/safzanpirani/gpu-query).

## License

[MIT](https://github.com/dy/gpu-font/blob/main/LICENSE). This work is also offered in the spirit described by the [Krishnized license](https://github.com/krishnized/license), which does not alter the MIT terms.
