"""Evidence for length slices, synthetic augmentation and honest abstention."""
import unittest
import json
import copy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch
from train import hundred
from train.robustness import sha
import torch
from train.ten import batch
from train.ten_model import Classifier, export, load_export
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from train.hundred_data import band, transform
from train.hundred import probabilities, metrics, calibrate


class HundredTests(unittest.TestCase):
    def test_invalid_steps_reject_before_resume_model_access(self):
        for value, configured in [(0, 40000), (-1, 40000), (False, 40000), (1.5, 40000), (None, 0), (None, -1)]:
            with self.subTest(value=value, configured=configured), patch.object(hundred, 'load_data', return_value=({'training': {'steps': configured}}, None, None)), patch.object(hundred, 'read') as read:
                with self.assertRaisesRegex(ValueError, 'steps must be a positive integer'):
                    hundred.train(value, resume=True)
                read.assert_not_called()

    def test_worse_one_step_resume_preserves_starting_weights(self):
        self.assert_resume_preserved(context=False)

    def test_worse_one_step_context_resume_preserves_initialized_weights(self):
        self.assert_resume_preserved(context=True)

    def assert_resume_preserved(self, context):
        torch.set_num_threads(1); torch.manual_seed(71)
        original = Classifier(2)
        samples = [{'family': family, 'renderer': renderer, 'role': role, 'text': 'aa', 'length': '2-3', 'condition': 'clean'}
                   for family, renderer, role in [('a', 'pillow', 'train'), ('a', 'chromium', 'train'), ('b', 'pillow', 'train'), ('b', 'chromium', 'train'), ('a', 'chromium', 'validation'), ('b', 'chromium', 'validation'), ('u', 'chromium', 'unknown-validation')]]
        windows = [{'source': i, 'offset': i * 21, 'width': 7, 'height': 3} for i in range(len(samples))]
        pixels = np.arange(len(samples) * 21, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / '.data/hundred'; data.mkdir(parents=True)
            (root / 'bench').mkdir()
            for name in ['src/input.mjs', 'src/prepare.mjs', 'train/ten_model.py']:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
            preparation = {'width': 128, 'height': 48, 'windows': 3, 'sha256': sha(root / 'src/input.mjs'), 'normalizerSha256': sha(root / 'src/prepare.mjs')}
            config = {'data': {'fonts': ['a', 'b'], 'input': {k: preparation[k] for k in ['width', 'height', 'windows']}},
                      'training': {'steps': 40000, 'batch': 2, 'seed': 71, 'learningRate': .002, 'threads': 1, 'device': 'cpu', 'dilations': [1, 1, 1, 1]}}
            manifest = {'dataConfig': config['data'], 'samples': samples, 'windows': windows, 'inputSha256': preparation['sha256'], 'preparationSha256': preparation['normalizerSha256']}
            artifact, quantized = export(original, ['a', 'b'], preparation)
            expected = artifact
            if context:
                initial = Classifier(2, context=True)
                initial.convs.load_state_dict(original.convs.state_dict(), strict=False)
                initial.norms.load_state_dict(original.norms.state_dict(), strict=False)
                initial.head.load_state_dict(original.head.state_dict())
                torch.nn.init.dirac_(initial.convs[-1].weight); torch.nn.init.zeros_(initial.convs[-1].bias)
                expected, quantized = export(initial, ['a', 'b'], preparation)
            (data / 'model.json').write_text(json.dumps(artifact))
            (data / 'prepared.json').write_text(json.dumps(manifest))
            torch.save({'state': original.state_dict(), 'step': 9}, data / 'best.pt')
            (root / 'bench/hundred-training.json').write_text(json.dumps({'modelSha256': sha(data / 'model.json'), 'totalSeconds': 1}))
            calls = []
            def predict(model, manifest, pixels, roles):
                calls.append(roles)
                if roles == ['validation', 'unknown-validation']:
                    for key, value in quantized.state_dict().items():
                        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
                    return [4, 5, 6], np.array([[8., 0.], [0., 8.], [0., 0.]]), np.arange(3)
                if len(calls) == 1:
                    return [4, 5], np.array([[8., 0.], [0., 8.]]), np.arange(2)
                self.assertFalse(torch.equal(model.head.bias, original.head.bias), 'The optimizer must actually perform a step')
                return [4, 5], np.array([[0., 8.], [8., 0.]]), np.arange(2)
            with patch.object(hundred, 'ROOT', root), patch.object(hundred, 'DATA', data), patch.object(hundred, 'load_data', return_value=(config, manifest, pixels)), patch.object(hundred, 'predict', side_effect=predict), redirect_stdout(StringIO()):
                hundred.train(1, resume=True, context=context)
            self.assertEqual(json.loads((data / 'model.json').read_text()), expected)
            report = json.loads((root / 'bench/hundred-training.json').read_text())
            self.assertEqual(report['selectedStep'], 0)
            self.assertEqual(report['history'][0]['step'], 0)
            self.assertEqual(report['history'][1]['groups']['all']['accuracy'], 0)
            self.assertEqual(report['validationInt8']['groups']['all']['accuracy'], 1)

    def test_context_network_fold_quantization_and_masked_shapes(self):
        torch.set_num_threads(1); torch.manual_seed(31)
        model = Classifier(100, context=True, dilations=[1, 1, 2, 2, 1]).eval()
        windows = [{'width': 1, 'height': 1, 'offset': 0}, {'width': 127, 'height': 47, 'offset': 1}]
        pixels = np.arange(1 + 127 * 47, dtype=np.uint8)
        with torch.no_grad():
            together = model(*batch(pixels, windows, [0, 1]))
            for index in [0, 0, 1, 0]:
                torch.testing.assert_close(together[index], model(*batch(pixels, windows, [index]))[0], atol=1e-7, rtol=1e-5)
            x, sizes = batch(pixels, windows, [0, 1])
            torch.testing.assert_close(model(x, sizes), model.folded()(x, sizes), atol=1e-7, rtol=1e-5)
            artifact, quantized = export(model, [str(i) for i in range(100)], {'width': 128, 'height': 48, 'windows': 3})
            self.assertEqual(len(artifact['layers']), 6)
            torch.testing.assert_close(quantized(x, sizes), load_export(artifact)(x, sizes), atol=0, rtol=0)

    def test_manifest_rejects_unknown_training_leakage_and_packed_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / '.data/hundred'; data.mkdir(parents=True)
            (root / 'bench').mkdir()
            config = {'fonts': ['a'], 'input': {'width': 128, 'height': 48, 'windows': 3}}
            (root / 'bench/hundred.json').write_text(json.dumps({'data': config}))
            (root / 'bench/fonts-100.json').write_text(json.dumps({'fonts': [{'id': 'a', 'split': 'train'}, {'id': 'u', 'split': 'unknown-validation'}, {'id': 'v', 'split': 'unknown-test'}]}))
            roles = ['train', 'validation', 'test', 'unknown-validation', 'unknown-test']
            samples = [{'family': family, 'role': role, 'text': text} for family, role, text in zip(['a', 'a', 'a', 'u', 'v'], roles, ['aa', 'bb', 'cc', 'bb', 'cc'])]
            manifest = {'dataConfig': config, 'samples': samples, 'windows': [{'source': i, 'offset': i, 'width': 1, 'height': 1} for i in range(5)]}
            pins = [('fontsSha256', 'bench/fonts-100.json'), ('textsSha256', 'bench/hundred-texts.json'), ('generatorSha256', 'train/hundred_data.py'), ('runnerSha256', 'scripts/hundred.mjs'), ('inputSha256', 'src/input.mjs'), ('preparationSha256', 'src/prepare.mjs'), ('tensorSha256', '.data/hundred/prepared.u8')]
            for key, name in pins:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                if key == 'tensorSha256': path.write_bytes(bytes([0, 64, 128, 192, 255]))
                elif not path.exists(): path.write_text('fixture')
                manifest[key] = sha(path)
            def write(value): (data / 'prepared.json').write_text(json.dumps(value))
            with patch.object(hundred, 'ROOT', root), patch.object(hundred, 'DATA', data):
                write(manifest)
                _, _, pixels = hundred.load_data(); np.testing.assert_array_equal(pixels, [0, 64, 128, 192, 255]); del pixels
                changes = [lambda m: m['samples'][3].update(role='train'), lambda m: m['samples'][1].update(role='unknown-validation'), lambda m: m['samples'][0].update(text='BB'), lambda m: m['samples'][4].update(family='absent'), lambda m: m['windows'][1].update(offset=0), lambda m: m['windows'][-1].update(source=5), lambda m: m['windows'][0].update(width=0), lambda m: m['windows'][0].update(width=1.5), lambda m: m['windows'][1].update(source=True), lambda m: m['windows'][0].update(height=True), lambda m: m['windows'].__setitem__(0, None), lambda m: m['windows'].pop(), lambda m: m.update(inputSha256='changed')]
                for mutate in changes:
                    bad = copy.deepcopy(manifest); mutate(bad); write(bad)
                    with self.assertRaises(ValueError): hundred.load_data()
                for payload in [b'', bytes(4), bytes(6)]:
                    (data / 'prepared.u8').write_bytes(payload); bad = copy.deepcopy(manifest); bad['tensorSha256'] = sha(data / 'prepared.u8'); write(bad)
                    with self.assertRaises(ValueError): hundred.load_data()

    def test_length_bands_count_letters_not_spaces(self):
        self.assertEqual([band(s) for s in ['i', 'ri', 'riv', 'font', 'sevenxx', 'two words']], ['1', '2-3', '2-3', '4-7', '4-7', '8+'])

    def test_augmentation_preserves_ink_and_margin_preserves_every_source_pixel(self):
        image = Image.new('L', (70, 40), 255)
        ImageDraw.Draw(image).text((8, 8), 'riv', fill=0, font=ImageFont.load_default(size=24))
        plan = {'index': 7, 'length': '2-3', 'light': False}
        for condition in ['clean', 'edge', 'small', 'rotate', 'shadow', 'margin', 'blur', 'jpeg', 'augmented']:
            a = transform(image, plan, condition)
            self.assertLess(a.getextrema()[0], 100); self.assertGreater(a.getextrema()[1], 200)
            self.assertEqual(a.tobytes(), transform(image, plan, condition).tobytes())
            b = transform(image, {**plan, 'light': True}, condition)
            np.testing.assert_array_equal(np.asarray(b), 255 - np.asarray(a))
        padded = transform(image, plan, 'margin')
        self.assertEqual(padded.crop((7, 40, 77, 80)).tobytes(), image.tobytes())
        with self.assertRaises(ValueError): transform(image, plan, 'invalid')

    def test_probabilities_average_each_source_and_keep_extreme_logits_finite(self):
        logits = np.array([[1000, 998], [998, 1000], [1000, 998]])
        p = probabilities(logits, np.array([0, 0, 1]), 2, 2)
        np.testing.assert_allclose(p[0], [.5, .5])
        np.testing.assert_allclose(p[1], [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
        np.testing.assert_allclose(p.sum(1), 1)
        for temperature in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError): probabilities(logits, np.array([0, 0, 1]), 2, temperature)

    def test_empty_prediction_and_missing_calibration_groups_fail_before_inference(self):
        model = unittest.mock.Mock()
        with self.assertRaisesRegex(ValueError, 'No windows'):
            hundred.predict(model, {'windows': [], 'samples': []}, np.array([], dtype=np.uint8), ['test'])
        model.eval.assert_not_called()
        for family in ['a', 'unknown']:
            manifest = {'dataConfig': {'fonts': ['a', 'b']}, 'samples': [{'family': family}]}
            with self.assertRaisesRegex(ValueError, 'known and unknown'):
                calibrate(manifest, [0], np.array([[8., 0.]]), np.array([0]))

    def test_calibration_separates_unknowns_and_abstains_when_precision_is_impossible(self):
        samples = [{'family': f, 'length': '1', 'condition': 'clean'} for f in ['a'] * 200 + ['b'] * 200 + ['unknown'] * 100]
        manifest = {'dataConfig': {'fonts': ['a', 'b']}, 'samples': samples}
        logits = np.array([[8., 0.]] * 200 + [[0., 8.]] * 200 + [[0., 0.]] * 100)
        owners = np.arange(len(samples)); sources = list(owners)
        c = calibrate(manifest, sources, logits, owners)
        r = c['validation']['rejection']
        self.assertEqual(r['acceptedCorrect'], 400); self.assertEqual(r['acceptedAccuracy'], 1)
        self.assertEqual(r['unknownAccepted'], 0); self.assertEqual(r['coverage'], 1)
        # Same scores but incorrect labels: there is no honest high-precision threshold.
        for sample in samples[:200]: sample['family'] = 'b'
        c = calibrate(manifest, sources, logits, owners)
        self.assertGreater(c['threshold'], 1)
        self.assertEqual(c['validation']['rejection']['accepted'], 0)
        self.assertIsNone(c['validation']['rejection']['acceptedAccuracy'])
        unfiltered = metrics(manifest, sources, probabilities(logits, owners, len(sources)))
        self.assertEqual(unfiltered['groups']['all']['accuracy'], .5)
        self.assertEqual(unfiltered['groups']['all']['count'], 400)


if __name__ == '__main__': unittest.main()
