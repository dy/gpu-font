# Experimental Google Fonts encoder

This is the shared encoder currently used by the demo. It comes from the [frozen Google Fonts experiment](../../bench/encoder.md) and has not met the retrieval or rejection gates. [Subsequent refinement experiments](../../bench/encoder-quality.md) are evaluated separately before replacing it.

- `encoder.json`: 349,920 parameters, packed int8 tensors, 128-dimensional L2-normalized output, no font labels. SHA-256 `2d14ce60f0e988f2bef2e0d95648957b823521536bd25d528ddebe0b7ace5d32`.
- `google-fonts.json`: 2,004 eligible Google Fonts families, one selected normal face and one int8 reference vector per family. Exact encoder/preparation hashes prevent mixing incompatible catalogs. Weight/style fields describe the source face; they are not inferred attributes.
- `best.pt`: selected floating-point PyTorch checkpoint for reproducibility, excluded from browser payload.

The encoder was trained from scratch on 1,403 families, selected using 301 development families, then frozen before indexing the remaining 300. Adding those 300 required reference generation only. The classification head used during training is discarded.

Prepare tight grayscale windows with the exact hashed `deskew-windows` preparation in the encoder, preserving aspect ratio and at most 128×48 pixels per window. Normalize each projection, average all windows from the source, then normalize again. References use different strings from queries. Rebuild catalogs whenever encoder or preparation changes.

Reproduce indexing and evaluation with `node scripts/python.mjs -m train.encoder_catalog test` after preparing the pinned final dataset; see the experiment report for all data/training commands. Verify CPU/Metal parity with `node checks/encoder.mjs`. The demo loads source catalogs through `src/catalog.mjs`, checks encoder/preparation bindings, and searches their reference vectors using the shared encoder. The package is not yet published.

Final unseen-family macro top-5 is 54.75%; query-weighted top-1/top-5 is 26.78%/47.31%. The synthetic rejection threshold accepts only 0.0277% of present-font queries. Use these artifacts as a measured baseline, not a reliable arbitrary-image detector.
