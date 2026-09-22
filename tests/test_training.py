import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.fixtures import validate_config, check_coverage
from scripts import fixtures
from train.model import Encoder, contrastive_loss, export_model, load_export
from train.train import validate_samples, evaluate
from train import train as training


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradient_update_and_json_round_trip(self):
        torch.manual_seed(5)
        model = Encoder()
        pixels = torch.rand(4, 1, 32, 128)
        labels = torch.tensor([0, 0, 1, 1])
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        before = model.project.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            vectors = model(pixels)
            self.assertTrue(torch.allclose(vectors.norm(dim=1), torch.ones(4), atol=1e-6))
            loss = contrastive_loss(vectors, labels)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)
            optimizer.step()
        self.assertFalse(torch.equal(before, model.project.weight))
        restored = load_export(json.loads(json.dumps(export_model(model))))
        with torch.no_grad():
            torch.testing.assert_close(restored(pixels), model(pixels), rtol=0, atol=0)
            self.assertEqual(tuple(restored(pixels[:1]).shape), (1, 32))

    def test_contrastive_loss_rewards_family_agreement_and_requires_positives(self):
        labels = torch.tensor([0, 0, 1, 1])
        good = torch.tensor([[1., 0], [1., 0], [0, 1.], [0, 1.]])
        bad = good[[0, 2, 1, 3]]
        self.assertLess(contrastive_loss(good, labels), contrastive_loss(bad, labels))
        with self.assertRaises(ValueError):
            contrastive_loss(good[:1], labels[:1])
        with self.assertRaises(ValueError):
            contrastive_loss(good[:0], labels[:0])
        with self.assertRaises(ValueError):
            contrastive_loss(good, torch.arange(4))

    def test_export_rejects_wrong_architecture_names_shape_count_and_nan(self):
        original = export_model(Encoder())
        name = next(iter(original['tensors']))
        mutations = [lambda a: a.update(architecture='other'),
                     lambda a: a['tensors'].pop(name),
                     lambda a: a['tensors'][name].update(shape=[1]),
                     lambda a: a['tensors'][name]['values'].pop(),
                     lambda a: a['tensors'][name]['values'].__setitem__(0, float('nan'))]
        for mutate in mutations:
            artifact = copy.deepcopy(original)
            mutate(artifact)
            with self.assertRaises(ValueError):
                load_export(artifact)

    def test_family_and_text_leakage_and_missing_catalog_are_rejected(self):
        samples = [dict(family='a', split='train', role=role, text=text, renderer='pillow')
                   for role, text in [('train', 'first'), ('prototype', 'second'), ('query', 'third')]]
        validate_samples(samples)
        cases = [[], samples[:2]]
        for field, value in [('split', 'unseen-dev'), ('text', 'third'), ('role', 'unknown')]:
            case = copy.deepcopy(samples); case[0][field] = value; cases.append(case)
        case = copy.deepcopy(samples); case[-1]['family'] = 'missing'; cases.append(case)
        case = copy.deepcopy(samples); case[-1]['split'] = 'unseen-dev'; cases.append(case)
        for case in cases:
            with self.assertRaises(ValueError):
                validate_samples(case)

    def test_retrieval_uses_prototypes_and_macro_averages_families(self):
        samples = [dict(family=family, role=role, renderer='browser', split='unseen-dev') for family, role in
                   [('a', 'prototype'), ('b', 'prototype'), ('a', 'query'), ('a', 'query'), ('b', 'query')]]
        vectors = torch.tensor([[1., 0], [0, 1.], [1., 0], [1., 0], [1., 0]])
        groups, catalog = evaluate(vectors, samples)
        self.assertEqual(groups['browser/unseen-dev']['top1'], .5)
        self.assertEqual(groups['browser/unseen-dev']['top5'], 1)
        self.assertEqual(groups['browser/unseen-dev']['queries'], 3)
        self.assertEqual(catalog[1], {'family': 'b', 'vector': [0., 1.]})

    def test_config_disjointness_and_glyph_coverage_fail_closed(self):
        config = {'fonts': [{'id': 'a', 'split': 'train'}], 'training': {'texts': ['train']}, 'evaluation': {'texts': ['proto', 'query']}}
        validate_config(config)
        config['training']['texts'].append('query')
        with self.assertRaises(ValueError): validate_config(config)
        class Font:
            def getBestCmap(self): return {ord('a'): 'a', ord(' '): 'space'}
        check_coverage(Font(), ['a a'])
        with self.assertRaisesRegex(ValueError, 'Missing glyphs'):
            check_coverage(Font(), ['ab'])

    def test_preview_rejects_each_stale_source_and_corrupt_tensor(self):
        with tempfile.TemporaryDirectory(prefix='gpu-font-test-') as folder:
            root = Path(folder)
            data = root / '.data'
            data.mkdir()
            sources = {'configSha256': root / 'bench/pilot.json',
                       'fontsSha256': root / 'bench/fonts.json',
                       'preparationSha256': root / 'src/prepare.mjs',
                       'tensorSha256': data / 'prepared.f32'}
            for path in sources.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'original')
            manifest = {key: fixtures.sha(path) for key, path in sources.items()}
            fixtures.write_json(data / 'prepared.json', manifest)
            with patch.object(fixtures, 'ROOT', root), patch.object(fixtures, 'DATA', data):
                for path in sources.values():
                    path.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'Stale or corrupt preparation'):
                        fixtures.preview({})
                    path.write_bytes(b'original')

    def test_python_readers_reject_truncated_and_trailing_tensor_bytes(self):
        with tempfile.TemporaryDirectory(prefix='gpu-font-test-') as folder:
            root = Path(folder)
            data = root / '.data'
            data.mkdir()
            config = {'seed': 0, 'training': {'threads': 2, 'batchPerFamily': 2}}
            fixtures.write_json(root / 'bench/pilot.json', config)
            fixtures.write_json(root / 'bench/fonts.json', {})
            (root / 'src').mkdir()
            (root / 'src/prepare.mjs').write_text('kernel')
            samples = [dict(family='a', split='train', role=role, text=role) for role in ['train', 'prototype', 'query']]
            manifest = {'samples': samples, 'height': 1, 'width': 1,
                        'configSha256': fixtures.sha(root / 'bench/pilot.json'),
                        'fontsSha256': fixtures.sha(root / 'bench/fonts.json'),
                        'preparationSha256': fixtures.sha(root / 'src/prepare.mjs')}
            # Three float32s require exactly 12 bytes. Hashes match in every case:
            # length validation must reject the corrupt structure independently.
            with patch.object(fixtures, 'ROOT', root), patch.object(fixtures, 'DATA', data), patch.object(training, 'ROOT', root):
                for size in [0, 11, 13, 16]:
                    (data / 'prepared.f32').write_bytes(bytes(size))
                    fixtures.write_json(data / 'prepared.json', {**manifest, 'tensorSha256': fixtures.sha(data / 'prepared.f32')})
                    for run in [lambda: fixtures.preview(config), training.run]:
                        with self.subTest(bytes=size, reader=run):
                            with self.assertRaisesRegex(ValueError, 'Prepared tensor length mismatch'):
                                run()


if __name__ == '__main__':
    unittest.main()
