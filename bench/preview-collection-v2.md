# Second raster-reference batch

Collected 24 additional upright Regular faces from public [Fontshare](https://www.fontshare.com/)
specimen pages under the [collection contract](../preview-handoff.md). All records remain
`role: "reference"`; this batch makes no training or recognition-accuracy claim.

## Archive

- Local directory: `.data/previews/pilot-v2/` — manifest, attempts, report, contact sheet,
  source config, quality checks, SHA-256 inventory, images and evidence.
- Manifest SHA-256: `170d6cba0cf7ca6690aaac5808afd9b80eea015bd10397b3e539cf4cb17b0ff3`.
- Transfer: `.data/previews/pilot-v2.tar.gz`, 10,762,851 bytes;
  SHA-256 `c9729e7bc62b91179fa0b8ee4615ec942f40027b9ef70b8728fb57822530abb4`.
- Directory: 175 files / 12,687,337 bytes. Extraction to a separate temporary directory
  preserved every byte. The contact sheet decoded all 141 references with networking disabled.
- The original `pilot-v1` archive's 206 files remain byte-identical.

## Coverage

| Measure | Result |
| --- | --- |
| Accepted families / genuine faces | 24 / 24 |
| Controlled references | 120: all five recipes for every face |
| Provided source-page phrases | 21, with 20 unique strings of 9–25 characters |
| Accepted records | 141 |
| Requested slots across 25 candidate families | 150; 9 rejected or unavailable |
| Categories | 9 sans, 7 serif, 2 slab, 1 mono, 5 display |
| Scripts | Latn: 117 records; Zyyy: 24 digits-only records |
| Logged review/attempt events | 8: 5 rejection events, 2 region corrections, 1 evidence correction |
| Duplicate-pixel warnings | 0 |

The 24 families are absent by exact name from the first pilot and the pinned 2,055-family
Google Fonts inventory. This does not prove unseen designs or exclude cross-source aliases.
The collection adds one-source coverage; independent source diversity remains limited.

Visual comparison pairs include Amulya/Ranade, Bespoke Sans/Plein, Bespoke Serif/Recia and
Nippo/Technor. RX100's equal Latin glyph advances verify its mono classification. Full family,
face, designer, source and phrase details are in the archive report and
[pinned source configuration](preview-sources-v2.json).

## Verification and limits

Controlled specimens use each source's actual static font bytes at 48 CSS px, device scale 2,
zoom 1, with synthesis disabled. Source IDs, page labels and resource paths establish face
identity; binary metadata verifies weight, slant and glyph coverage. All 24 binaries have
placeholder family names, recorded in their flags. The rendered line must differ from fallback.
Numeric weights are preserved exactly: Author 375,
Nippo 378 and Paquito 300 despite the source's Regular/400 labels.

Provided specimens retain the site's existing phrases in a wider viewport. All accepted
phrases match the selected face's loaded resource. They are source-page screenshots, without
text substitution or raster alteration. No style labels count toward provided coverage.

Kihim's Regular face is italic in its binary and visible shape; it was rejected and replaced by
Pilcrow Rounded. Ranade, Striper and Pilcrow Rounded lack accepted optional provided specimens;
the source config pins those observed gaps so a complete resume does not repeat failed requests.
No blocked source or rate-limit response occurred.

Visual review corrected Bespoke Serif's region to exclude adjacent UI, expanded Gambarino's
region to include its full J overhang, and replaced Pilcrow Rounded's evidence after the page
scrolled away from the style label. Original screenshots and earlier evidence remain retained.
The collector now checks all four search margins and uses immediate scrolling.

All 168 retained PNGs fully decode. Accepted records pass offline validation with zero errors
or warnings; all regions retain at least 8 physical pixels of margin, and all 24 evidence
hashes match. Every accepted face's crops and evidence were visually inspected. A complete
resume reused 141 records with zero additions, replacements, attempts or byte changes.
Direct collection/source tests: **17 passed, 0 failed**. The coordinated full suite passed
**68 JavaScript and 71 Python tests**. All seven catalogue browser checks and the existing
100-font UI suite also passed in the root session.

The [ITF Free Font License](https://www.fontshare.com/licenses/itf-ffl) and source attribution are
recorded in the archive. Third-party captures and cached binaries stay local and gitignored.
These references improve coverage and preserve useful text; recognition quality still requires
independent queries and measurement.

## Commands

```sh
node scripts/preview-sources-v2.mjs
PREVIEW_ARCHIVE=.data/previews/pilot-v2 PREVIEW_CACHE=.data/previews/.cache/fonts-v2 PREVIEW_SOURCES=bench/preview-sources-v2.json node scripts/preview-collect.mjs
node scripts/preview-validate.mjs .data/previews/pilot-v2
node scripts/preview-contact-sheet.mjs .data/previews/pilot-v2
node --test tests/preview-collection.test.mjs tests/preview-sources-v2.test.mjs
```

The source generator accepts a retained public catalogue response through
`PREVIEW_FONTSHARE_CATALOGUE`. Re-encoding or validating the archive requires no source visits
and no font cache. Copy the entire archive when moving it to another workspace.
