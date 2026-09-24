# gpu-font

Find the font in an image, in your browser. Crop a line of text from a screenshot or a design and get the closest font families, with weight and style, from Google Fonts and other free catalogs.

[![A crop of "Quiet rivers flow" matched to Lora Regular, then Sumana](https://raw.githubusercontent.com/dy/gpu-font/main/og.png)](https://dy.github.io/gpu-font/)

**[Try it](https://dy.github.io/gpu-font/)**

- Runs on the user's device, on WebGPU or the CPU: no server, no API key, the image is never uploaded.
- Names the face, not only the family: Bold Italic, Light and so on.
- Searches 2,004 Google Fonts families, every weight and italic, or one of four other free catalogs, 3,593 families in all.
- Families with identical letters fold into one row: IBM Plex Sans KR and IBM Plex Sans Arabic show under IBM Plex Sans.

It is experimental: on fonts it never saw in training, the right family is in the top five 87% of the time and first 69%.

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
- `siblings`: names of families with the same letters, folded into this one.

`matcher.destroy()` releases the GPU.

Without a bundler, import it from a CDN:

```html
<script type="module">
  import { createMatcher } from 'https://cdn.jsdelivr.net/npm/gpu-font/src/match.mjs'
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

const matcher = await createMatcher(catalogs.fontshare)
```

| `catalogs.` | Families | |
|---|---:|---|
| `google-fonts` (default) | 2,004 | [Google Fonts](https://fonts.google.com) |
| `debian` | 1,509 | [Debian's font packages](https://packages.debian.org/sid/fonts/) |
| `fontshare` | 39 | [Fontshare](https://www.fontshare.com) |
| `collletttivo` | 17 | [Collletttivo](https://www.collletttivo.it) |
| `other` | 24 | DejaVu, Bitstream Vera, Droid, D-DIN, Velvetyne and more |

A catalog built for this model also works from its URL: `createMatcher('https://example.com/my-catalog.json')`.

## How good it is

On crops of 5–10 characters from 300 Google Fonts families the model never saw in training ([method and full results](https://github.com/dy/gpu-font/blob/main/bench/style.md)):

| | |
|---|---:|
| Right family in the top five | 87% |
| Right family first | 69% |
| Top five: Latin, other scripts, Chinese | 88%, 75%, 58% |
| Weight of the matched face, 100–900 | off by 5 on average |
| Italic or upright | 99% right |

Those crops are rendered text; screenshots and photos are not measured yet. It does best on one line in one font, several letters, on a plain background.

Not yet: hand-drawn letters, icons, symbols and emoji, and matching capitals when a catalog holds only lowercase (54% top five).

## Speed and size

- `createMatcher()` downloads about 6 MB, compressed: the model and the Google Fonts catalog. Debian's catalog adds 0.7 MB; the others are under 40 KB.
- With WebGPU, a match takes about 0.1 s. Without it, it runs on the CPU: about 10 s in Chromium, 12–20 s in Node. All measured on an Apple M4 Max.

## Development

Running the demo, training the model and building catalogs: [development.md](https://github.com/dy/gpu-font/blob/main/development.md).

## License

[MIT](https://github.com/dy/gpu-font/blob/main/LICENSE). This work is also offered in the spirit described by the [Krishnized license](https://github.com/krishnized/license), which does not alter the MIT terms.
