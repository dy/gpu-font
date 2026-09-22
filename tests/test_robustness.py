import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from train.robustness import augment, condition, metrics, threshold_at_fpr, validate_samples
from train import robustness


class RobustnessTests(unittest.TestCase):
    def test_tight_crop_preserves_ink_and_uniform_scale(self):
        source = Image.new('L', (40, 30), 255)
        ImageDraw.Draw(source).rectangle((10, 5, 29, 14), fill=0)
        tight = augment(source, {})
        self.assertEqual(tight.size, (20, 10))
        self.assertTrue((np.asarray(tight)[:, :, :3] == 0).all())
        self.assertEqual(augment(source, {'scale': .5}).size, (10, 5))
        self.assertEqual(augment(source, {'window': .5, 'position': 1}).size, (10, 10))
        light = augment(source, {'light': True})
        np.testing.assert_array_equal(np.asarray(light)[:, :, :3], 255 - np.asarray(tight)[:, :, :3])
        self.assertEqual(source.size, (40, 30))
        self.assertEqual(source.getpixel((0, 0)), 255)
        for plan in [{'window': 0}, {'position': -1}, {'scale': 0}, {'pad': -1}]:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                augment(source, plan)
        with self.assertRaisesRegex(ValueError, 'Empty'):
            augment(Image.new('L', (1, 1), 255), {})
        self.assertEqual(augment(Image.new('L', (1, 1), 0), {}).size, (1, 1))

    def test_variations_keep_nonblank_rgba_and_repeated_calls_independent(self):
        image = Image.new('L', (50, 30), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 2, 9, 25), fill=0)
        draw.rectangle((9, 15, 45, 20), fill=0)
        saved = np.asarray(augment(image, {})).copy()
        for name in ['clean', 'tight', 'short', 'small', 'rotate-left', 'rotate-right', 'blur', 'jpeg', 'mixed']:
            with self.subTest(name=name):
                actual = augment(image, condition(name))
                self.assertEqual(actual.mode, 'RGBA')
                self.assertTrue((np.asarray(actual)[:, :, 3] == 255).all())
                self.assertGreater(np.asarray(ImageOps.invert(actual.convert('L'))).sum(), 0)
        np.testing.assert_array_equal(np.asarray(augment(image, {})), saved)

    def test_float32_threshold_respects_fpr_with_ties_and_boundaries(self):
        probabilities = np.array([.9, .8, .8, .8, .3, .1], dtype=np.float32)
        labels = [True, False, False, False, False, False]
        for fpr in [0, .2, .4, .8]:
            with self.subTest(fpr=fpr):
                threshold = threshold_at_fpr(probabilities, labels, fpr)
                result = metrics(probabilities, labels, threshold)
                self.assertLessEqual(result['falsePositiveRate'], fpr)
        threshold = threshold_at_fpr(np.array([1, 1], dtype=np.float32), [True, False], 0)
        self.assertEqual(metrics([1, 1], [True, False], threshold)['falsePositives'], 0)
        for probs, ys, fpr in [([], [], .1), ([.5], [True], .1), ([.5], [False], .1), ([np.nan, .3], [True, False], .1), ([.5, .3], [True, False], 1)]:
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                threshold_at_fpr(probs, ys, fpr)
        self.assertEqual(metrics([.9, .7, .3, .2], [True, False, True, False], .5),
                         {'positives': 2, 'negatives': 2, 'truePositives': 1, 'falsePositives': 1,
                          'recall': .5, 'falsePositiveRate': .5, 'precision': .5})

    def test_text_and_family_leakage_and_positive_only_data_reject(self):
        samples = [dict(role=role, text=role, family=family, split='train')
                   for role in ['train', 'calibration', 'query'] for family in ['inter', 'roboto']]
        validate_samples(samples, 'inter')
        bad = [dict(s) for s in samples]
        bad[2]['text'] = 'train'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_samples(bad, 'inter')
        bad = [dict(s) for s in samples]
        bad[2]['split'] = 'unseen-dev'
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            validate_samples(bad, 'inter')
        with self.assertRaisesRegex(ValueError, 'negative'):
            validate_samples([s for s in samples if s['family'] == 'inter'], 'inter')
        with self.assertRaises(ValueError):
            validate_samples([], 'inter')
        bad = [dict(s) for s in samples]
        bad[-1]['split'] = 'unseen-dev'
        with self.assertRaisesRegex(ValueError, 'Family crosses'):
            validate_samples(bad, 'inter')

    def test_prepared_geometry_boundaries_and_freshness_reject_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['bench', 'src', 'train', '.data/robustness/pilot']:
                (root / name).mkdir(parents=True, exist_ok=True)
            data = root / '.data/robustness'
            config = {'target': 'inter', 'inputs': {'pilot': {'width': 1, 'height': 1, 'padding': 0}}}
            (root / 'bench/robustness.json').write_text(json.dumps(config))
            for name in ['bench/fonts.json', 'src/prepare.mjs', 'train/model.py']:
                (root / name).write_text('version A')
            samples = [dict(role=role, text=role, family=family, split='train')
                       for role in ['train', 'calibration', 'query'] for family in ['inter', 'roboto']]
            manifest = {'samples': samples, **config['inputs']['pilot'],
                        'configSha256': robustness.sha(root / 'bench/robustness.json'),
                        'fontsSha256': robustness.sha(root / 'bench/fonts.json'),
                        'preparationSha256': robustness.sha(root / 'src/prepare.mjs'),
                        'encoderSha256': robustness.sha(root / 'train/model.py'),
                        'generatorSha256': robustness.sha(Path(robustness.__file__))}
            tensor = data / 'pilot/prepared.f32'
            def save(value):
                (data / 'pilot/prepared.json').write_text(json.dumps({**value, 'tensorSha256': robustness.sha(tensor)}))
            with patch.object(robustness, 'ROOT', root), patch.object(robustness, 'DATA', data):
                # Six one-pixel samples require exactly 24 bytes, even when hashes match.
                for size in [0, 23, 25, 28]:
                    tensor.write_bytes(bytes(size)); save(manifest)
                    with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'tensor length'):
                        robustness.train('pilot')
                tensor.write_bytes(np.ones(6, dtype='<f4').tobytes())
                save({**manifest, 'width': 2})
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    robustness.train('pilot')
                save(manifest)
                (root / 'train/model.py').write_text('version B')
                with self.assertRaisesRegex(ValueError, 'Stale'):
                    robustness.train('pilot')
                (root / 'train/model.py').write_text('version A')
                tensor.write_bytes(np.full(6, np.nan, dtype='<f4').tobytes()); save(manifest)
                with self.assertRaisesRegex(ValueError, 'Invalid prepared pixels'):
                    robustness.train('pilot')


if __name__ == '__main__':
    unittest.main()
