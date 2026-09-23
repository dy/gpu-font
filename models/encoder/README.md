# Style encoder

The shared encoder the demo uses. How it was trained and measured: [bench/style.md](../../bench/style.md).

- `encoder.json`: 3.04M-parameter convolutional encoder (`font-conv96-192-288-384-384-v5`), packed int8, 128-dimensional L2-normalized output, no font labels. Typed heads (weight, italic, script, Google category and fine class) travel in the same file as small float layers; the browser reads them with `readHeads`. SHA-256 `62e556d8b0a34aa18618cc9ddaa34a632f49117480f08b6e6cdcd998103b0605`.
- `google-fonts.json`: version 3, every face of the 2,004 Google Fonts families (8,132 faces) with 23,234 int8 references rendered by Chromium: lowercase and uppercase Latin and each other script. Faces carry a style name, weight, script coverage and, on default faces, twins per script (families whose letters in that script differ less than one font's own re-renders), which the page folds into one row. Bound to the exact encoder and preparation hashes.
- `catalogs/`: Fontshare and Velvetyne catalogs derived from local preview captures (names, links, vectors; no images). Only sources whose terms `bench/foundries.json` records as `permitted` or `none-found` are written here; others stay in `.data/catalogs/withheld/`. Format: version 3, one reference per capture (both alphabets, two phrases, digits).
- `best.pt`: the selected PyTorch checkpoint with heads and face proxies; not part of the browser payload.

Trained on 1,403 families: 30,000 steps at 1.36M parameters, 10,000 more, then widened 1.5× with its function preserved and trained 20,000 steps. The 301 development families selected every checkpoint and the reference renderer; the 300 test families never entered training or selection. On the frozen Chromium benchmark those held-out families reach 87.3% top-five and 68.6% top-one (identical designs counted). The preparation is unchanged (`deskew-windows`, 128×48, up to three windows). Adding fonts requires references only; changing the encoder requires regenerating every catalog. Reproduce with the commands in the report.

The PyTorch checkpoint `best.pt` is not committed (the repository holds scripts and JSON only); training writes it locally.
