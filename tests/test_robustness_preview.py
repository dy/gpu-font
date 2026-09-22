import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFont

from scripts import robustness_preview as preview


class RobustnessPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / '.data/robustness'
        for directory in ['bench', '.data/robustness/pilot', '.data/robustness/detail']:
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self.enterContext(patch.object(preview.ImageFont, 'truetype', return_value=ImageFont.load_default()))
        self.enterContext(patch.object(preview, 'ROOT', self.root))
        self.enterContext(patch.object(preview, 'DATA', self.data))
        self.config = {'inputs': {'pilot': {'width': 128, 'height': 32, 'padding': 2},
                                  'detail': {'width': 256, 'height': 64, 'padding': 2}}}
        self.write(self.root / 'bench/robustness.json', self.config)
        self.raw = bytes([0, 0, 0, 255])
        (self.data / 'source.rgba').write_bytes(self.raw)
        self.manifest = {'configSha256': preview.sha(self.root / 'bench/robustness.json'),
                         'rawSha256': preview.sha(self.data / 'source.rgba'),
                         'samples': [{'role': 'query', 'renderer': 'chromium', 'family': 'inter', 'condition': 'tight',
                                      'text': 'Quiet rivers flow', 'size': 28, 'dpr': 1, 'polarity': 'dark',
                                      'width': 1, 'height': 1, 'offset': 0}]}
        self.write(self.data / 'samples.json', self.manifest)
        self.prepared = {}
        self.tensors = {}
        for name, settings in self.config['inputs'].items():
            tensor = np.full(settings['width'] * settings['height'], .25 if name == 'pilot' else .75, dtype='<f4').tobytes()
            self.tensors[name] = tensor
            (self.data / name / 'prepared.f32').write_bytes(tensor)
            self.prepared[name] = {**self.manifest, **settings, 'tensorSha256': preview.sha(self.data / name / 'prepared.f32')}
            self.save_prepared(name)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def save_prepared(self, name):
        path = self.data / name / 'prepared.json'
        self.write(path, self.prepared[name])
        self.write(self.root / f'bench/robustness-{name}.json', {'manifestSha256': preview.sha(path)})

    def test_repeated_render_uses_exact_tensor_and_source_pixels(self):
        first = preview.run().read_bytes()
        self.assertEqual(preview.run().read_bytes(), first)  # A → A
        with Image.open(preview.run()) as image:
            self.assertEqual(image.size, (1060, 188))
            self.assertEqual(image.getpixel((16, 84)), (0, 0, 0))
            self.assertEqual(image.getpixel((408, 84)), (64, 64, 64))
            self.assertEqual(image.getpixel((748, 84)), (191, 191, 191))
        # A → different B: a valid newly trained tensor changes only its specimen.
        path = self.data / 'pilot/prepared.f32'
        path.write_bytes(np.full(4096, .5, dtype='<f4').tobytes())
        self.prepared['pilot']['tensorSha256'] = preview.sha(path)
        self.save_prepared('pilot')
        with Image.open(preview.run()) as image:
            self.assertEqual(image.getpixel((408, 84)), (128, 128, 128))
            self.assertEqual(image.getpixel((748, 84)), (191, 191, 191))
        path.write_bytes(self.tensors['pilot'])
        self.prepared['pilot']['tensorSha256'] = preview.sha(path)
        self.save_prepared('pilot')
        self.assertEqual(preview.run().read_bytes(), first)  # B → A

    def test_stale_corrupt_empty_and_boundary_inputs_reject(self):
        source = self.data / 'source.rgba'
        for size in [0, 3, 5]:
            source.write_bytes(bytes(size))
            with self.subTest(raw_size=size), self.assertRaisesRegex(ValueError, 'source length'):
                preview.run()
        source.write_bytes(bytes([255, 255, 255, 255]))
        with self.assertRaisesRegex(ValueError, 'source checksum'):
            preview.run()
        source.write_bytes(self.raw)
        manifest_path = self.data / 'samples.json'
        self.write(manifest_path, {**self.manifest, 'rawSha256': 'a-new-render'})
        with self.assertRaises(ValueError):
            preview.run()
        # Keep checksums consistent but pair a new source manifest with old tensors.
        self.write(manifest_path, {**self.manifest, 'generatorSha256': 'new-generator'})
        with self.assertRaisesRegex(ValueError, 'Stale preview inputs'):
            preview.run()
        self.write(manifest_path, self.manifest)
        for name in ['pilot', 'detail']:
            tensor = self.data / name / 'prepared.f32'
            count = len(self.tensors[name])
            for size in [0, count - 1, count + 1]:
                tensor.write_bytes(bytes(size))
                with self.subTest(name=name, size=size), self.assertRaisesRegex(ValueError, 'tensor length'):
                    preview.run()
            tensor.write_bytes(bytes(count))
            with self.assertRaisesRegex(ValueError, 'tensor checksum'):
                preview.run()
            tensor.write_bytes(np.full(count // 4, np.nan, dtype='<f4').tobytes())
            self.prepared[name]['tensorSha256'] = preview.sha(tensor)
            self.save_prepared(name)
            with self.assertRaisesRegex(ValueError, 'Invalid preview pixels'):
                preview.run()
            tensor.write_bytes(self.tensors[name])
            self.prepared[name]['tensorSha256'] = preview.sha(tensor)
            self.save_prepared(name)
            report = self.root / f'bench/robustness-{name}.json'
            self.write(report, {'manifestSha256': 'previous-run'})
            with self.assertRaisesRegex(ValueError, 'training report'):
                preview.run()
            self.save_prepared(name)
        self.write(self.root / 'bench/robustness.json', {**self.config, 'seed': 999})
        with self.assertRaisesRegex(ValueError, 'configuration'):
            preview.run()
        self.write(self.root / 'bench/robustness.json', self.config)
        self.write(manifest_path, {**self.manifest, 'samples': []})
        with self.assertRaisesRegex(ValueError, 'source length'):
            preview.run()
        self.write(manifest_path, self.manifest)
        self.assertTrue(preview.run().is_file())


if __name__ == '__main__':
    unittest.main()
