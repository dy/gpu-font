import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


class DemoReferenceTests(unittest.TestCase):
    def test_preview_label_validates_shipped_face_and_recovers_after_invalid_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            (root / 'dist/assets/fonts').mkdir(parents=True)
            runner = root / 'scripts/demo_reference.py'
            shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts/demo_reference.py', runner)
            (root / 'dist/assets/model.json').write_text(json.dumps({'fonts': ['fixture']}))
            # There is deliberately no training-cache font: check the actual browser asset.
            for weight, selection, angle, valid in [(400, 0, 0, True), (400, 0, 0, True),
                    (700, 0, 0, False), (400, 1, 0, False), (400, 512, 0, False),
                    (400, 0, -12, False), (400, 0, 0, True)]:
                with self.subTest(weight=weight, selection=selection, angle=angle):
                    font = FontBuilder(1000, isTTF=True)
                    font.setupGlyphOrder(['.notdef', 'A'])
                    pen = TTGlyphPen(None)
                    pen.moveTo((0, 0)); pen.lineTo((400, 0)); pen.lineTo((200, 700)); pen.closePath()
                    font.setupGlyf({'.notdef': TTGlyphPen(None).glyph(), 'A': pen.glyph()})
                    font.setupHorizontalMetrics({'.notdef': (500, 0), 'A': (500, 0)})
                    font.setupHorizontalHeader(ascent=800, descent=-200)
                    font.setupCharacterMap({c: 'A' for c in range(32, 127)})
                    font.setupNameTable({'familyName': 'Fixture', 'styleName': 'Test'})
                    font.setupOS2(version=4, usWeightClass=weight, fsSelection=selection)
                    font.setupHead(macStyle=2 if selection & 1 else 0)
                    font.setupPost(italicAngle=angle); font.setupMaxp()
                    font.save(root / 'dist/assets/fonts/fixture.ttf')
                    # Stop at model loading: this test concerns preview assets, not inference.
                    with patch('train.ten_model.load_export', side_effect=RuntimeError('fonts validated')) as load:
                        with self.assertRaisesRegex(RuntimeError if valid else ValueError,
                                'fonts validated' if valid else 'upright 400-weight face: fixture'):
                            runpy.run_path(str(runner), run_name='__main__')
                        self.assertEqual(load.call_count, int(valid))
