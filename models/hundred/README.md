# Preserved 100-family checkpoint

Baseline before the preparation/style experiments. Model SHA-256: `66b98a49d1bebe9b2e56bc896f1fbe9b8e76ece52e74608aeae58ef3dd4cefbb`. Training, calibration and evaluation provenance are in `bench/hundred-training.json` and `bench/hundred-test.json`.

`model.json` is the int8 inference artifact; `calibration.json` contains its validation calibration; `best.pt` retains the PyTorch state for continued training. Restore these three files into `.data/hundred/` to reproduce this checkpoint. Font binaries and rendered datasets are regenerated from the pinned manifests and are not included here.

The PyTorch checkpoint `best.pt` is not committed (the repository holds scripts and JSON only); training writes it locally.
