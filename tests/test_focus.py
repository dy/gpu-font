import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from train import focus


class FocusTests(unittest.TestCase):
    def test_text_split_is_case_insensitive_disjoint_and_repeatable(self):
        counts = {'train': 512, 'calibration': 64, 'query': 64}
        a = focus.text_pools(counts, 31415)
        self.assertEqual(a, focus.text_pools(counts, 31415))
        self.assertNotEqual(a, focus.text_pools(counts, 31416))
        self.assertEqual(a, focus.text_pools(counts, 31415))  # A → A → B → A
        self.assertEqual({role: len(pool) for role, pool in a.items()}, counts)
        texts = [text.lower() for pool in a.values() for text in pool]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertTrue(all(text.strip() for text in texts))
        # Minimum split: one distinct source string per role.
        self.assertEqual(len({t for pool in focus.text_pools(dict.fromkeys(counts, 1), 0).values() for t in pool}), 3)
        for invalid in [{}, {'train': 1}, {**counts, 'query': 0}, {**counts, 'train': None}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                focus.text_pools(invalid, 0)

    def test_patch_preserves_pixels_and_height_without_stretching(self):
        array = np.full((12, 42), 255, dtype=np.uint8)
        array[1:11, 1:41] = 0
        array[3:9, 11:31] = 100
        source = Image.fromarray(array)
        original = source.tobytes()
        settings = {'width': 20, 'height': 10, 'padding': 0}
        a = focus.window(source, settings)
        self.assertEqual(a.size, (20, 10))
        np.testing.assert_array_equal(np.asarray(a), array[1:11, 11:31])
        self.assertEqual(a.tobytes(), focus.window(source, settings).tobytes())
        self.assertEqual(focus.window(source, {**settings, 'width': 1000}).size, (40, 10))
        self.assertEqual(a.tobytes(), focus.window(source, settings).tobytes())
        self.assertEqual(source.tobytes(), original)
        minimum = {'width': 4, 'height': 4, 'padding': 0}
        self.assertEqual(focus.window(Image.new('L', (1, 1), 0), minimum).getpixel((0, 0)), 0)
        with self.assertRaisesRegex(ValueError, 'Empty'):
            focus.window(Image.new('L', (1, 1), 255), minimum)
        for bad in [None, {}, {**settings, 'width': None}, {**settings, 'width': 3},
                    {**settings, 'height': 1025}, {**settings, 'padding': -1},
                    {**settings, 'padding': 5}, {**settings, 'padding': 1.5},
                    {**settings, 'height': float('nan')}, {**settings, 'width': float('inf')}]:
            with self.subTest(settings=bad), self.assertRaisesRegex(ValueError, 'geometry'):
                focus.window(source, bad)

    def test_window_uses_configured_padding_at_zero_default_and_boundary(self):
        # Twenty pixels wide and ten high: usable aspect changes with padding.
        source = Image.fromarray(np.tile(np.arange(100, dtype=np.uint8), (10, 1)))
        for padding, expected_width in [(0, 20), (2, 27), (4, 60)]:
            with self.subTest(padding=padding):
                settings = {'width': 20, 'height': 10, 'padding': padding}
                actual = focus.window(source, settings)
                left = (100 - expected_width) // 2
                self.assertEqual(actual.size, (expected_width, 10))
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(source)[:, left:left + expected_width])
        # Default production geometry must keep the original 124/28 aspect.
        self.assertEqual(focus.window(source, {'width': 128, 'height': 32, 'padding': 2}).size, (44, 10))

    def test_tensor_reader_checks_boundaries_values_geometry_and_freshness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / '.data/focus'
            data.mkdir(parents=True)
            config = {'input': {'width': 1, 'height': 1, 'padding': 0}, 'target': 'inter'}
            files = {'configSha256': 'bench/focus.json', 'fontsSha256': 'bench/fonts.json',
                     'encoderSha256': 'train/model.py', 'augmentationSha256': 'train/robustness.py',
                     'rendererSha256': 'scripts/fixtures.py', 'preparationSha256': 'src/prepare.mjs',
                     'runnerSha256': 'scripts/focus.mjs'}
            for path in files.values():
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(config) if path == 'bench/focus.json' else 'A')
            samples = [dict(role=role, text=role, family=family, split='train')
                       for role in ['train', 'calibration', 'query'] for family in ['inter', 'roboto']]
            manifest = {key: focus.sha(root / path) for key, path in files.items()}
            manifest.update(generatorSha256=focus.sha(Path(focus.__file__)), samples=samples, input=config['input'])
            tensor = data / 'prepared.f32'
            valid = np.arange(6, dtype='<f4') / 5
            def save(raw, change=None):
                tensor.write_bytes(raw)
                (data / 'prepared.json').write_text(json.dumps({**manifest, 'tensorSha256': focus.sha(tensor), **(change or {})}))
            with patch.object(focus, 'ROOT', root), patch.object(focus, 'DATA', data):
                save(valid.tobytes())
                np.testing.assert_array_equal(focus.load_data()[2].ravel(), valid)
                np.testing.assert_array_equal(focus.load_data()[2].ravel(), valid)
                # Advanced indexing owns its data; the tensor needs no second full copy.
                import torch
                mapped = focus.load_data()[2]
                selected = mapped[[0, 2, 4]]
                self.assertFalse(np.shares_memory(selected, mapped))
                tensor_view = torch.from_numpy(selected)
                tensor_view[0] = .75
                self.assertEqual(float(selected[0, 0, 0, 0]), .75)
                np.testing.assert_array_equal(focus.load_data()[2].ravel(), valid)
                tensor.write_bytes(np.ones(6, dtype='<f4').tobytes())
                with self.assertRaisesRegex(ValueError, 'Stale'):
                    focus.load_data()
                for size in [0, 23, 25, 28]:
                    save(bytes(size))
                    with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'length'):
                        focus.load_data()
                for value in [float('nan'), float('inf'), -.01, 1.01]:
                    save(np.full(6, value, dtype='<f4').tobytes())
                    with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'pixels'):
                        focus.load_data()
                save(valid.tobytes(), {'input': {'width': 2, 'height': 1, 'padding': 0}})
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    focus.load_data()
                save(valid.tobytes())
                (root / 'scripts/focus.mjs').write_text('B')
                with self.assertRaisesRegex(ValueError, 'Stale'):
                    focus.load_data()
                (root / 'scripts/focus.mjs').write_text('A')
                np.testing.assert_array_equal(focus.load_data()[2].ravel(), valid)

    def test_preview_exact_pixels_repeat_and_reject_stale_sources_or_reports(self):
        from scripts import focus_preview as preview
        with tempfile.TemporaryDirectory() as tmp:
            root, data = Path(tmp), Path(tmp) / '.data/focus'
            data.mkdir(parents=True)
            (root / 'bench').mkdir()
            raw = bytes([0, 0, 0, 255])
            (data / 'source.rgba').write_bytes(raw)
            (data / 'prepared.json').write_text('{}')
            report = {'manifestSha256': focus.sha(data / 'prepared.json')}
            for policy in ['line', 'patch']:
                (root / f'bench/focus-{policy}.json').write_text(json.dumps(report))
            samples = [dict(role='query', renderer='chromium', family='inter', condition='clean',
                            policy=policy, text='A', plan=0, offset=0, width=1, height=1)
                       for policy in ['line', 'patch']]
            pixels = np.stack([np.full((1, 32, 128), .25), np.full((1, 32, 128), .75)])
            loaded = ({'policies': ['line', 'patch'], 'target': 'inter'},
                      {'samples': samples, 'rawSha256': focus.sha(data / 'source.rgba')}, pixels)
            with patch.object(preview, 'ROOT', root), patch.object(preview, 'DATA', data), patch.object(preview, 'load_data', return_value=loaded):
                path = preview.run()
                original = path.read_bytes()
                self.assertEqual(preview.run().read_bytes(), original)
                with Image.open(path) as image:
                    self.assertEqual(image.getpixel((20, 80)), (0, 0, 0))
                    self.assertEqual(image.getpixel((340, 80)), (64, 64, 64))
                    self.assertEqual(image.getpixel((870, 80)), (191, 191, 191))
                pixels[0] = .5
                with Image.open(preview.run()) as image:
                    self.assertEqual(image.getpixel((340, 80)), (128, 128, 128))
                pixels[0] = .25
                self.assertEqual(preview.run().read_bytes(), original)
                (data / 'source.rgba').write_bytes(bytes([255, 255, 255, 255]))
                with self.assertRaisesRegex(ValueError, 'Source checksum'):
                    preview.run()
                (data / 'source.rgba').write_bytes(raw)
                (data / 'prepared.json').write_text('{"new":true}')
                with self.assertRaisesRegex(ValueError, 'different inputs'):
                    preview.run()
                (data / 'prepared.json').write_text('{}')
                self.assertEqual(preview.run().read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
