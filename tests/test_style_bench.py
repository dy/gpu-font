import tempfile
import unittest
from pathlib import Path
from unittest import mock

from train import style_bench
from train.encoder_data import save
from train.robustness import read, sha

REASON = 'A sampler was added beside the one the benchmark uses.'


class RepinTests(unittest.TestCase):
    """repin moves a benchmark's pins to changed modules only when preparing again gives the very same pixels, and writes
    the move into every file's `repins` with its evidence."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name); out = root/'.data/style'; out.mkdir(parents=True)
        (root/'module.mjs').write_text('old'); (root/'other.mjs').write_text('same')
        pins = {'module.mjs': sha(root/'module.mjs'), 'other.mjs': sha(root/'other.mjs')}
        earlier = [{'file': 'other.mjs', 'from': 'a', 'to': pins['other.mjs'], 'date': '2026-09-24', 'reason': 'An earlier move.'}]
        save(out/'bench-plan.json', {'pins': pins, 'repins': earlier, 'queries': [{'text': 'ab'}]})
        save(out/'bench-browser.json', {'planSha256': sha(out/'bench-plan.json'), 'sha256': 'renders', 'images': [1]})  # one render for the plan's one query
        save(out/'bench.json', {'pins': pins, 'repins': earlier, 'planSha256': sha(out/'bench-plan.json'), 'sha256': 'pixels', 'samples': [1, 2], 'windows': [1, 2, 3]})
        save(out/'bench-sketch.json', {'pins': pins, 'planSha256': sha(out/'bench-plan.json'), 'sha256': 'sketch', 'samples': [1], 'windows': [1]})
        (out/'bench.u8').write_bytes(b'old pixels'); (out/'bench-sketch.u8').write_bytes(b'old sketch')
        (root/'module.mjs').write_text('new')  # the module changed after the benchmark was frozen
        self.root, self.out, self.earlier = root, out, earlier
        self.patches = [mock.patch.object(style_bench, 'ROOT', root), mock.patch.object(style_bench, 'OUT', out), mock.patch.object(style_bench, 'PINS', ['module.mjs', 'other.mjs']),
                        mock.patch.object(style_bench, 'stage', lambda name: root)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in self.patches: p.stop()
        self.tmp.cleanup()

    def faking(self, pixels, sketch='sketch'):
        """pack and sketches as the real ones behave from the renders: a fresh manifest with the plan's pins, the plan's hash and the given pixel hash, no repins."""
        def pack(name):
            (self.out/f'{name}.u8').write_bytes(pixels.encode())  # pack writes the pixels beside the manifest
            plan = read(self.out/f'{name}-plan.json'); save(self.out/f'{name}.json', {'pins': plan['pins'], 'planSha256': sha(self.out/f'{name}-plan.json'), 'sha256': pixels, 'samples': [1, 2], 'windows': [1, 2, 3]})
        def sketches(refresh=False):
            plan = read(self.out/'bench-plan.json'); save(self.out/'bench-sketch.json', {'pins': plan['pins'], 'planSha256': sha(self.out/'bench-plan.json'), 'sha256': sketch, 'samples': [1], 'windows': [1]})
        return [mock.patch.object(style_bench, 'pack', pack), mock.patch.object(style_bench, 'sketches', sketches)]

    def files(self):
        return {name: (self.out/name).read_bytes() for name in ['bench-plan.json', 'bench-browser.json', 'bench.json', 'bench-sketch.json', 'bench.u8', 'bench-sketch.u8']}

    def test_unchanged_pixels_move_the_pins_and_record_the_move_with_its_evidence_everywhere(self):
        with self.faking('pixels')[0], self.faking('pixels')[1]: self.assertEqual(style_bench.repin('bench', REASON), ['module.mjs'])
        plan = read(self.out/'bench-plan.json'); current = {'module.mjs': sha(self.root/'module.mjs'), 'other.mjs': sha(self.root/'other.mjs')}
        self.assertEqual(plan['pins'], current); self.assertEqual(plan['queries'], [{'text': 'ab'}])  # only the pins and the record moved
        for name in ['bench-browser', 'bench', 'bench-sketch']: self.assertEqual(read(self.out/f'{name}.json')['planSha256'], sha(self.out/'bench-plan.json'), name)
        bench = read(self.out/'bench.json'); self.assertEqual(bench['pins'], current); self.assertEqual(bench['repins'][:1], self.earlier)  # the earlier moves are kept
        entry = bench['repins'][1]; self.assertEqual((entry['file'], entry['from'], entry['to']), ('module.mjs', sha_of('old'), sha(self.root/'module.mjs')))
        self.assertRegex(entry['date'], r'^\d{4}-\d{2}-\d{2}$'); self.assertTrue(entry['reason'].startswith(REASON)); self.assertIn('2 queries, 3 windows, pixels sha256 pixels unchanged', entry['reason'])
        self.assertEqual(len(bench['repins']), 2); self.assertEqual(len(plan['repins']), 2); self.assertIn("plan's text", plan['repins'][1]['reason'])
        sketch = read(self.out/'bench-sketch.json'); self.assertEqual(len(sketch['repins']), 1); self.assertIn('1 queries, 1 windows, pixels sha256 sketch', sketch['repins'][0]['reason'])
        with self.faking('pixels')[0], self.faking('pixels')[1]: self.assertEqual(style_bench.repin('bench', REASON), [])  # nothing left to move

    def test_changed_pixels_restore_every_file_and_say_so(self):
        before = self.files()
        with self.faking('different')[0], self.faking('different')[1]:
            with self.assertRaisesRegex(ValueError, r'preparation changed.*bench\.json'): style_bench.repin('bench', REASON)
        self.assertEqual(self.files(), before)

    def test_a_changed_sketch_alone_restores_too(self):
        before = self.files()
        with self.faking('pixels', sketch='other')[0], self.faking('pixels', sketch='other')[1]:
            with self.assertRaisesRegex(ValueError, r'bench-sketch\.json'): style_bench.repin('bench', REASON)
        self.assertEqual(self.files(), before)

    def test_a_benchmark_without_a_sketch_slice_repins_alone(self):
        pins = read(self.out/'bench-plan.json')['pins']
        save(self.out/'catalog-plan.json', {'pins': pins, 'queries': [{'text': 'cd'}]}); save(self.out/'catalog.json', {'pins': pins, 'planSha256': 'x', 'sha256': 'pixels', 'samples': [], 'windows': []})
        renders = read(self.out/'bench-browser.json'); save(self.out/'bench-browser.json', {**renders, 'planSha256': sha(self.out/'catalog-plan.json')})  # the stage holds this benchmark's renders
        sketch = (self.out/'bench-sketch.json').read_bytes()
        with self.faking('pixels')[0]: self.assertEqual(style_bench.repin('catalog', REASON), ['module.mjs'])
        catalog = read(self.out/'catalog.json'); self.assertEqual(read(self.out/'catalog-plan.json')['pins']['module.mjs'], sha(self.root/'module.mjs')); self.assertEqual(catalog['planSha256'], sha(self.out/'catalog-plan.json'))
        self.assertEqual([e['file'] for e in catalog['repins']], ['module.mjs']); self.assertEqual((self.out/'bench-sketch.json').read_bytes(), sketch)  # another benchmark's slice is not touched

    def test_renders_of_another_benchmark_stop_the_move_before_any_file_changes(self):
        renders = read(self.out/'bench-browser.json'); save(self.out/'bench-browser.json', {**renders, 'images': [1, 2, 3]})  # three renders for a plan of one query
        before = self.files()
        with self.faking('pixels')[0], self.faking('pixels')[1]:
            with self.assertRaisesRegex(ValueError, r'3 renders for a plan of 1 queries.*by hand'): style_bench.repin('bench', REASON)
        self.assertEqual(self.files(), before)

    def test_a_stop_while_packing_restores_every_file_the_pixels_included(self):
        import os, signal
        before = self.files()
        def pack(name):
            (self.out/'bench.u8').write_bytes(b'half written'); os.kill(os.getpid(), signal.SIGTERM)  # stopped mid-pack
        with mock.patch.object(style_bench, 'pack', pack):
            with self.assertRaises(KeyboardInterrupt): style_bench.repin('bench', REASON)
        self.assertEqual(self.files(), before)
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)  # the handler is put back

    def test_a_pack_that_fails_restores_too(self):
        before = self.files()
        with mock.patch.object(style_bench, 'pack', mock.Mock(side_effect=RuntimeError('node died'))):
            with self.assertRaises(RuntimeError): style_bench.repin('bench', REASON)
        self.assertEqual(self.files(), before)


def sha_of(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == '__main__': unittest.main()
