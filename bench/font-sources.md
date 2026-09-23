# Font sources: coverage, access and terms

Survey of where catalogue references can come from, checked 2026-09-23 by reading each site's
robots.txt **and** its terms of use (robots.txt alone understated the restrictions). Quotes and
URLs are from the scouting reports; "none found" means no clause was found on the named page, not
that permission was granted.

## 1. Font files we may render ourselves (no scraping)

| Source | Status |
| --- | --- |
| Google Fonts | in the corpus (2,004 trained families; colour fonts and Noto Emoji excluded) |
| Fontshare (ITF) | ~100 families, 15 captured; free for commercial use, licence silent on ML |
| Velvetyne, Collletttivo, Uncut.wtf (~150), Open Foundry, fontlibrary.org (~1,100) | open licences, mostly OFL; check per family |
| Linux defaults: DejaVu, Liberation, GNU FreeFont, URW base-35 (Nimbus) | free, **missing from the corpus** |
| Icon fonts: Material Symbols, Font Awesome Free, Remix Icon, Bootstrap Icons, Tabler | Apache/OFL/MIT; Lucide, Twemoji, Fluent Emoji are SVG-only |
| Music (SMuFL): Bravura, Petaluma, Leland, Gonville, Sebastian | OFL or public domain |
| Maths: STIX Two Math (in corpus), Latin Modern Math (missing) | OFL / GUST |
| macOS system fonts (~4,000 faces on this Mac) | SF Pro/SF Mono/New York licensed only for Apple-platform mockups; others not verified |

## 2. Preview-only sources

| Source | Families | Terms on automated capture |
| --- | --- | --- |
| MyFonts | 130,321 pages | **ban**: "any robot, spider … to retrieve, index, 'scrape,' 'data mine' … without our express prior, written consent" (verified) |
| Adobe Fonts | 5,616 | not verifiable: adobe.com refuses automated clients |
| Fontspring | not counted | **ban**: automated or non-automated scraping, robots, spiders |
| Creative Market | — | **ban**; robots.txt also disallows product pages |
| YouWorkForThem | ~30k product pages | **ban**: "spider, crawl, or scrape" |
| Future Fonts | dozens | restricted: no copying any part of the Service without written consent |
| I Love Typography | ~100+ | restricted: no reproducing any part of the site |
| Fontstand | hundreds | robots.txt disallows AI crawlers by name |
| Type Network | not counted | only a font EULA found; no site terms located |
| ParaType | 1,359 | none found (no terms page located); robots allows /fonts/ |
| Pangram Pangram | ~62 | **ban**: "spider, crawl, or scrape" |
| Sharp Type | — | **ban**: no robots or automated retrieval |
| Lineto | — | **ban** on AI/ML use of font data |
| Commercial Type | 123 | **ban** on using the font software with AI |
| ABC Dinamo | ~50 | **ban**: "use the fonts … to train artificial intelligence" |
| Frere-Jones | 5 | **ban**: no website content for ML/AI training |
| Typotheque | — | robots.txt blocks ClaudeBot by name |
| P22 (incl. ATF, Hamilton Wood Type) | 312 | **ban**: "spider, crawl, or scrape" |
| Font Diner | 34 | **ban**, explicitly naming "AI tools (such as agentic AI)" |
| Canada Type | — | restricted: personal, non-commercial viewing |
| Letterhead Fonts | hundreds | restricted: no republication; previews are static images |
| Klim | ~50 | none found (/license/, /privacy-policy/) |
| Grilli Type | 21 | none found (no terms page; privacy only) |
| Swiss Typefaces | 18 | none found (/support/termsofservice/); tester is a canvas |
| OH no Type Co | 7+ | none found (/license) |
| Production Type | not counted | none found (EULA and terms of use) |
| Displaay | ~29 | none found (/terms-and-conditions) |
| CoType | 3 families | no terms page found |
| House Industries (housefonts.com) | not counted | no terms page found |
| Colophon | — | closed 2025; sold via Monotype/MyFonts |
| Letraset | — | now Monotype/ITC, on MyFonts |
| Windows fonts | ~100 | no files off Windows; Microsoft publishes specimen images per font |

## 3. Rankings and real-world ground truth

| Source | Use | Terms |
| --- | --- | --- |
| Google Fonts metadata (`fonts.google.com/metadata/fonts`) | per-family popularity rank, ~1,946 families | none found |
| HTTP Archive Web Almanac | measured web usage per family, yearly | open (`Allow: /`) |
| Typewolf | top-50 list and a 1,144-font index of trendy commercial type | none found |
| Fonts In Use | usage counts and labelled real images | **ban** on data mining and image reuse; ask permission |
| Identifont | top-10 searches | **ban** on automation and republishing |
| Flickr, Creative Commons, `type:face=` machine tags | labelled real-world photos (the source Fonts In Use imports from) | per-image CC licence |
