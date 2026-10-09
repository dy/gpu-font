import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.corpus import ROOT
from train.read import CLASSES, HEIGHT, Reader, artifact_of, decode, distance, encode, fold, strips


class ReaderTests(unittest.TestCase):
    def test_labels_fold_accents_to_their_letters_and_refuse_what_has_no_class(self):
        self.assertEqual([fold(t) for t in ['Quiet rivers', 'café 2026', 'ÀÉÎÕÜ ñ', 'straße', 'a+b', '']], ['Quiet rivers', 'cafe 2026', 'AEIOU n', None, None, ''])
        self.assertEqual(decode(encode('Hi 5')), 'Hi 5')
        self.assertEqual(encode(' '), [1]); self.assertEqual(len(CLASSES), 63)

    def test_greedy_decoding_collapses_repeats_and_drops_blanks(self):
        h, i = CLASSES.index('h') + 1, CLASSES.index('i') + 1
        self.assertEqual(decode([0, h, h, 0, i, 0, 0]), 'hi'); self.assertEqual(decode([h, h, 0, h]), 'hh'); self.assertEqual(decode([0, 0]), ''); self.assertEqual(decode([]), '')

    def test_edit_distance(self):
        self.assertEqual([distance(a, b) for a, b in [('', ''), ('abc', 'abc'), ('abc', 'abd'), ('abc', 'ab'), ('', 'abc'), ('kitten', 'sitting')]], [0, 0, 1, 1, 3, 3])

    def test_a_strip_reads_the_same_alone_as_padded_beside_a_wider_one(self):
        torch.manual_seed(0); model = Reader().eval()
        narrow = np.random.default_rng(0).integers(0, 256, (HEIGHT, 100), np.uint8); wide = np.random.default_rng(1).integers(0, 256, (HEIGHT, 300), np.uint8)
        alone, columns = model(*strips([{'windows': [narrow]}], 'cpu')); both, columns2 = model(*strips([{'windows': [narrow]}, {'windows': [wide]}], 'cpu'))
        self.assertEqual((columns.tolist(), columns2.tolist()), ([13], [13, 38]))  # ceil(100/8), ceil(300/8)
        self.assertEqual((alone.shape[0], alone.shape[2]), (1, len(CLASSES) + 1))
        torch.testing.assert_close(alone[0, :13], both[0, :13], atol=1e-5, rtol=1e-4)  # columns beyond a strip's width never reach its own
        self.assertEqual(both.shape[1], 40)  # the batch's columns: the widest strip padded to the bucket (320 px), strided by 8

    def test_the_folded_reader_is_the_trained_function(self):
        torch.manual_seed(1); model = Reader()
        for norm in model.norms:  # non-trivial statistics, as training leaves them
            if hasattr(norm, 'running_mean'): norm.running_mean.uniform_(-1, 1); norm.running_var.uniform_(.5, 2); norm.weight.data.uniform_(.5, 1.5); norm.bias.data.uniform_(-.5, .5)
        model.eval(); strip = np.random.default_rng(2).integers(0, 256, (HEIGHT, 160), np.uint8); x, w = strips([{'windows': [strip]}], 'cpu')
        with torch.no_grad(): a, _ = model(x, w); b, _ = model.folded()(x, w)
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        self.assertLess(sum(p.numel() for p in model.folded().parameters()), 200_000)  # the whole reader, under 200k weights

    def test_the_line_preparer_gives_one_strip_where_the_windows_preparer_gives_windows(self):
        from PIL import Image, ImageDraw, ImageFont
        from train.style_data import Preparer
        image = Image.new('L', (300, 60), 255); ImageDraw.Draw(image).text((10, 10), 'Quiet rivers', fill=0, font=ImageFont.load_default(size=32)); gray = np.asarray(image)
        line, windows = Preparer('line'), Preparer()
        try:
            strips = line(gray); self.assertEqual(len(strips), 1); self.assertEqual(strips[0].shape[0], HEIGHT); self.assertGreater(strips[0].shape[1], HEIGHT)
            cut = windows(gray); self.assertGreaterEqual(len(cut), 2); self.assertTrue(all(w.shape[0] <= 48 and w.shape[1] <= 128 for w in cut))
            self.assertEqual(line(np.full((20, 40), 255, np.uint8)), [])  # a blank line prepares to nothing in either sampler
            self.assertEqual(line(gray)[0].shape, strips[0].shape)      # the child stays up: A → blank → A
        finally: line.close(); windows.close()

    def test_the_browser_reads_a_strip_as_the_exported_reader_does(self):
        torch.manual_seed(3); model = Reader()
        for norm in model.norms:
            if hasattr(norm, 'running_mean'): norm.running_mean.uniform_(-.5, .5); norm.running_var.uniform_(.5, 2)
        artifact, restored = artifact_of(model, 6); strip = np.random.default_rng(4).integers(0, 256, (HEIGHT, 200), np.uint8)
        self.assertEqual([l['kind'] for l in artifact['layers']][5], 'fold'); self.assertEqual(artifact['layers'][-1]['kind'], 'head')
        with torch.no_grad(): expected, columns = restored(*strips([{'windows': [strip]}], 'cpu'))
        script = '''
          import { readFile } from 'node:fs/promises'
          import { readReader, readLine } from './src/reader.mjs'
          const dir = process.env.READ_DIR, model = readReader(JSON.parse(await readFile(`${dir}/reader.json`, 'utf8')))
          const bytes = new Uint8Array(await readFile(`${dir}/strip.bin`)), width = bytes.length / 48
          const { text, columns, logits } = readLine(model, { width, height: 48, pixels: Float32Array.from(bytes, b => b / 255) })
          console.log(JSON.stringify({ text, columns, logits: [...logits] }))'''
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'reader.json').write_text(json.dumps(artifact)); (Path(tmp)/'strip.bin').write_bytes(strip.tobytes())
            out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT, env={**os.environ, 'READ_DIR': tmp})
        read = json.loads(out.stdout); logits = np.array(read['logits']).reshape(len(CLASSES) + 1, -1).T
        self.assertEqual(logits.shape, (int(columns[0]), len(CLASSES) + 1))
        np.testing.assert_allclose(logits, expected[0, :int(columns[0])].numpy(), atol=2e-3, rtol=1e-3)  # the browser computes the exported function
        self.assertEqual(read['columns'], expected[0, :int(columns[0])].argmax(1).tolist()); self.assertEqual(read['text'], decode(read['columns']))


if __name__ == '__main__': unittest.main()
