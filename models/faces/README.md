# Ten-family weight/style pilot

`model.json` contains the validated int8 classifier for 40 faces. `faces.json` maps its output labels to family, weight and style. See [the experiment](../../bench/faces.md) and [source hashes/licenses](../../bench/font-faces.json).

`best.pt` is the selected **folded** QAT checkpoint. Load it with `Classifier(40, training=False, context=True, dilations=[1,1,2,2,1])`, then load the `state` member. Batch-normalization statistics are already folded into the convolutions. Exporting it with the preparation metadata in `model.json` reproduces that artifact.

This candidate is not selected by the demo build. It covers only weights 400/700 and upright/italic in the original ten families, without unknown-family rejection calibration.
