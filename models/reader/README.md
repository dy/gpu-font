# Line reader

The line reader the demo and the package ship beside the encoder. How it was trained and measured: [bench/reader.md](../../bench/reader.md).

- `reader.json`: `read-24-48-96-160-v1`, 187,264 weights at 6 bits (222 KB, 156 KB gzipped), reading space, digits and the Latin letters from the strip `prepareLine`'s `line` sampler gives, 48 pixels tall. Connectionist temporal classification over eight-pixel columns; `src/reader.mjs` runs it on the CPU and decodes greedily. Trained 20,000 steps on the style stream's views of the training families; on unseen fonts its character error is 11.9% on clean renders and 29.3% on damaged ones, 11.0% on the final photographs.

The PyTorch checkpoint (`.data/read/reader-20k/best.pt`) is not committed; `node scripts/python.mjs -m train.read export --run reader-20k --bits 6` writes the file from it.
