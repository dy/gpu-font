"""Deployment invariants: padding, quantization and ownership cannot change logits."""
import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from train.ten import batch
from train.ten_model import Classifier, export, load_export
from train.ten_data import transform
from train import ten
from train.robustness import sha
from PIL import Image, ImageDraw


class TenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_step_counts_reject_before_model_or_checkpoint_access(self):
        for override, configured in [(0, 12000), (-1, 12000), (False, 12000), (1.5, 12000), (None, 0), (None, -1)]:
            with self.subTest(override=override, configured=configured), \
                    patch.object(ten, 'load_data', return_value=({'training': {'steps': configured}}, None, None)), \
                    patch.object(ten, 'Classifier') as classifier, patch.object(torch, 'load') as checkpoint:
                with self.assertRaisesRegex(ValueError, 'steps must be a positive integer'):
                    ten.train(override)
                classifier.assert_not_called()
                checkpoint.assert_not_called()

    def test_one_step_trains_and_replaces_an_old_checkpoint(self):
        settings = {'steps': 12000, 'threads': 1, 'seed': 13, 'batch': 2, 'learningRate': .001}
        config = {'training': settings, 'data': {'fonts': ['inter', 'roboto'], 'input': {'width': 128, 'height': 48, 'windows': 3}}}
        samples = [{'role': role, 'family': family, 'renderer': renderer, 'condition': 'clean'}
                   for family in config['data']['fonts'] for role, renderer in [('train', 'pillow'), ('train', 'chromium'), ('validation', 'chromium')]]
        windows = [{'source': i, 'width': 7, 'height': 3, 'offset': i * 21} for i in range(len(samples))]
        manifest = {'dataConfig': config['data'], 'samples': samples, 'windows': windows}
        pixels = np.arange(len(samples) * 21, dtype=np.uint8)
        torch.manual_seed(settings['seed'])
        initial_bias = Classifier(2).head.bias.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / '.data/ten'; data.mkdir(parents=True)
            (root / 'bench').mkdir()
            for name in ['src/input.mjs', 'src/prepare.mjs', 'train/ten_model.py']:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
            (data / 'prepared.json').write_text(json.dumps(manifest))
            (data / 'best.pt').write_bytes(b'an older, unusable checkpoint')
            with patch.object(ten, 'ROOT', root), patch.object(ten, 'DATA', data), \
                    patch.object(ten, 'load_data', return_value=(config, manifest, pixels)), redirect_stdout(StringIO()):
                ten.train(1)
            checkpoint = torch.load(data / 'best.pt', weights_only=True)
            self.assertEqual(checkpoint['step'], 1)
            self.assertFalse(torch.equal(checkpoint['state']['head.bias'], initial_bias))
            report = json.loads((root / 'bench/ten-training.json').read_text())
            self.assertEqual(report['steps'], 1)
            self.assertEqual(report['selectedStep'], 1)
            self.assertEqual(report['validationInt8']['groups']['all']['count'], 2)
            self.assertEqual(report['modelSha256'], sha(data / 'model.json'))

    def test_masked_batch_matches_individual_tight_inputs_and_reuse(self):
        torch.manual_seed(13)
        model = Classifier(10, dilations=[1, 1, 2, 2]).eval()
        images = [np.zeros((1, 1), np.uint8), np.arange(21, dtype=np.uint8).reshape(3, 7), np.full((48, 128), 255, np.uint8), np.full((47, 127), 128, np.uint8)]
        windows, offset = [], 0
        for image in images:
            windows.append({'width': image.shape[1], 'height': image.shape[0], 'offset': offset})
            offset += image.size
        pixels = np.concatenate([i.flatten() for i in images])
        with torch.no_grad():
            combined = model(*batch(pixels, windows, [0, 1, 2, 3]))
            for i in [0, 0, 1, 3, 2, 0]:
                individual = model(*batch(pixels, windows, [i]))[0]
                torch.testing.assert_close(individual, combined[i], atol=2e-7, rtol=1e-5)
        self.assertEqual(combined.shape, (4, 10))
        self.assertTrue(torch.isfinite(combined).all())

    def test_batchnorm_fold_and_int8_artifact_decode(self):
        torch.manual_seed(7)
        model = Classifier(10, dilations=[1, 1, 2, 2])
        # Nontrivial running statistics ensure the fold cannot silently skip BN.
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        before = model.convs[0].weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(torch.rand(8, 1, 23, 61)), torch.arange(8))
            loss.backward()
            self.assertTrue(torch.isfinite(model.convs[0].weight.grad).all())
            self.assertTrue(model.convs[0].weight.grad.abs().sum() > 0)
            optimizer.step()
        self.assertFalse(torch.equal(before, model.convs[0].weight))
        model.eval()
        image = torch.rand(2, 1, 27, 83)
        with torch.no_grad():
            torch.testing.assert_close(model(image), model.folded()(image), atol=2e-7, rtol=1e-5)
        artifact, quantized = export(model, [str(i) for i in range(10)], {'height': 48, 'width': 128, 'windows': 3})
        loaded = load_export(artifact)
        with torch.no_grad():
            torch.testing.assert_close(loaded(image), quantized(image), atol=0, rtol=0)
        for mutate in [lambda a: a.update(version=2), lambda a: a['layers'].pop(), lambda a: a['layers'][0].update(shape=[1])]:
            bad = copy.deepcopy(artifact); mutate(bad)
            with self.assertRaises(ValueError):
                load_export(bad)

    def test_training_augmentation_preserves_labeled_ink_and_polarity(self):
        image = Image.new('L', (100, 30), 255)
        ImageDraw.Draw(image).text((5, 5), 'Text text', fill=0)
        for light in [False, True]:
            plan = {'index': 2, 'position': .5, 'light': light}
            for condition in ['clean', 'crop', 'small', 'rotate', 'blur', 'jpeg', 'train']:
                result = transform(image, plan, condition)
                self.assertEqual(result.mode, 'L')
                self.assertGreater(result.width, 0)
                self.assertGreater(result.height, 0)
                lo, hi = result.getextrema()
                self.assertGreater(hi - lo, 100)
            self.assertEqual(transform(image, plan, 'train').tobytes(), transform(image, plan, 'train').tobytes())

    def test_manifest_rejects_leakage_stale_files_and_invalid_packed_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / '.data/ten'; data.mkdir(parents=True)
            pins = [('fontsSha256', 'bench/fonts.json'), ('textsSha256', 'bench/ten-texts.json'),
                    ('generatorSha256', 'train/ten_data.py'), ('augmentSha256', 'train/robustness.py'),
                    ('inputSha256', 'src/input.mjs'), ('preparationSha256', 'src/prepare.mjs'),
                    ('runnerSha256', 'scripts/ten.mjs')]
            config = {'fonts': ['inter'], 'input': {'width': 128, 'height': 48}}
            manifest = {'dataConfig': config, 'samples': [{'role': role, 'family': 'inter', 'text': word} for role, word in zip(['train', 'validation', 'test'], ['hello', 'world', 'font'])],
                        'windows': [{'source': i, 'width': 1, 'height': 1, 'offset': i} for i in range(3)]}
            for key, name in pins:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
                manifest[key] = sha(path)
            (root / 'bench/ten.json').write_text(json.dumps({'data': config}))
            (data / 'prepared.u8').write_bytes(bytes([0, 128, 255])); manifest['tensorSha256'] = sha(data / 'prepared.u8')
            def write(value):
                (data / 'prepared.json').write_text(json.dumps(value))
            with patch.object(ten, 'ROOT', root), patch.object(ten, 'DATA', data):
                write(manifest)
                _, loaded, pixels = ten.load_data()
                values, sizes = batch(pixels, loaded['windows'], [0, 1, 2])
                torch.testing.assert_close(values.flatten(), torch.tensor([0., 128 / 255, 1.]))
                self.assertEqual(sizes.tolist(), [[1, 1]] * 3)
                for mutate in [lambda m: m['samples'][1].update(text='HELLO'), lambda m: m['samples'][2].update(role='train'),
                               lambda m: m['windows'][1].update(offset=0), lambda m: m['windows'][2].update(source=3),
                               lambda m: m['windows'][0].update(width=0), lambda m: m['windows'][0].update(width=1.5),
                               lambda m: m['windows'][1].update(source=True), lambda m: m['windows'][0].update(height=True), lambda m: m['windows'].__setitem__(0, None), lambda m: m['windows'].pop(),
                               lambda m: m.update(inputSha256='stale')]:
                    bad = copy.deepcopy(manifest); mutate(bad); write(bad)
                    with self.assertRaises(ValueError):
                        ten.load_data()
                del pixels
                for payload in [b'', bytes([0, 128]), bytes([0, 128, 255, 0])]:
                    (data / 'prepared.u8').write_bytes(payload)
                    bad = copy.deepcopy(manifest); bad['tensorSha256'] = sha(data / 'prepared.u8'); write(bad)
                    with self.assertRaises(ValueError):
                        ten.load_data()


if __name__ == '__main__':
    unittest.main()
