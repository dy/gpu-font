# Weight and italic pilot

The network can learn weight and italic alongside family identity without becoming large. This pilot recognizes **40 genuine faces**: the original ten families × weights 400/700 × upright/italic. It is a separate candidate, not the demo's 100-family regular-only model.

The held-out synthetic test has 2,560 crops from eight new strings, two renderers, four faces per family and four conditions. The same renderers also occur in training; this is not independent screenshot accuracy. Single letters share the training alphabet. Exact counts and confusion matrices are in [faces-test.json](faces-test.json).

| Test condition | Crops | Family | Weight 400/700 | Upright/italic | Exact family + weight + style |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 2,560 | 80.7% | 97.4% | 97.0% | 78.0% |
| Clean | 640 | 94.5% | 99.4% | 98.9% | 92.8% |
| Small | 640 | 69.4% | 92.2% | 95.6% | 65.0% |
| Rotated | 640 | 72.2% | 99.5% | 94.7% | 68.8% |
| Shadow | 640 | 86.9% | 98.4% | 98.9% | 85.6% |

Family, weight and style predictions sum the joint probabilities over the other attributes. They are separate marginal decisions; the highest-scoring individual face can belong to a different family from the highest family marginal. The report's `knownFamily*` fields are oracle diagnostics conditioned on the correct family, not deployable accuracy claims.

## Data and training

Sources are pinned to Google Fonts revision `e44c4b011a820c2cbe2fd2cfa8052037d7edb571`. [font-faces.json](font-faces.json) records original Git blob hashes, SHA-256 hashes, variable-axis instances and OFL text. Weight and italic labels are checked against each font's OS/2 metadata. Every face contains the 62 Latin letters/digits. Italic comes from the actual italic font; rotation augmentation is applied to both upright and italic examples.

Training uses 40,640 crops, 2,560 development crops and 2,560 test crops. Multi-character strings are disjoint across these roles and final strings are checked against the historical 100-family corpus. Synthetic modifiers include partial edge cuts, downscaling, rotation, shadows, margins, blur and JPEG. The deployed JavaScript sampler prepares every crop; this experiment does not change normalization or apply synthetic italic labels.

The preserved 100-family encoder initializes the pilot. A 40-class head starts with each family's original row repeated across its four faces. All parameters then train for 15,000 updates, batch 64, AdamW with cosine learning rate 0.0008 → 0.000025. Validation selected update 12,000. The float training loop took about 213 seconds on the M4 Max, excluding data generation/export; its original export stopped at the quantization gate, so the recovered machine report leaves that duration/history unset instead of reconstructing missing records.

Int8 export reduced joint validation accuracy from 73.55% to 72.42%, exceeding the one-point allowance. A bounded **2,000-update quantization-aware pass** took 22.8 seconds; validation selected update 1,000, reaching **74.06%**. No test results selected these checkpoints. [faces-training.json](faces-training.json) preserves float, initial int8 and final int8 measurements.

The final model has **85,912 parameters**, **132,105 bytes JSON** and **86,248 bytes Brotli**. CPU and WebGPU outputs agree with independent PyTorch outputs within `1e-4` absolute logit error, including minimum/maximum/odd sizes and repeated inference. See [faces-browser.json](faces-browser.json).

## Scope and next use

This identifies two discrete weights, not continuous variable-axis coordinates. It has no unknown-family rejection calibration and no evidence of generalization to other families or intermediate weights. Do not attach its scores to all 100 demo families or label them calibrated certainty. Family identity remains the harder task, especially for short/degraded text.

The useful follow-up is joint family and attribute supervision over the same expanded face catalog, evaluated on new held-out faces and screenshots. Preserve stroke width and natural proportions during preparation; skeleton-only inputs would discard evidence needed for weight.

## Reproduce

```sh
node scripts/python.mjs scripts/font-faces.py
node scripts/faces.mjs data
node scripts/python.mjs -m train.faces train
# Only after validation selects the artifact:
node scripts/python.mjs -m train.faces evaluate
node checks/artifact.mjs .data/faces/model.json
```

Training requires MPS. The browser check uses the demo server on port 4179. `train.faces finish` resumes export/QAT from a completed float checkpoint after an interrupted export. The retained artifact and folded checkpoint are in [models/faces](../models/faces/README.md).
