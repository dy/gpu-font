import json
from pathlib import Path
import sys
import tempfile
import unittest

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables.ttProgram import Program
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import previews as P


def font_file(folder, draw, fpgm=None):
    """A TrueType font whose letters a-z, A-Z and 0-9 are the glyph `draw(pen)` makes; `fpgm` a hinting program for FreeType to run."""
    letters = [chr(c) for c in [*range(48, 58), *range(65, 91), *range(97, 123)]]
    builder = FontBuilder(1000, isTTF=True); builder.setupGlyphOrder(['.notdef', 'shape']); builder.setupCharacterMap({ord(c): 'shape' for c in letters})
    pen = TTGlyphPen(None); draw(pen); shape = pen.glyph(); empty = TTGlyphPen(None).glyph()
    builder.setupGlyf({'.notdef': empty, 'shape': shape}); builder.setupHorizontalMetrics({'.notdef': (500, 0), 'shape': (600, 0)})
    builder.setupHorizontalHeader(ascent=800, descent=-200); builder.setupNameTable({'familyName': 'Test', 'styleName': 'Regular'}); builder.setupOS2(); builder.setupPost()
    if fpgm is not None:
        builder.font['fpgm'] = newTable('fpgm'); builder.font['fpgm'].program = Program(); builder.font['fpgm'].program.fromBytecode(fpgm)
    path = Path(folder) / f'{len(list(Path(folder).iterdir()))}.ttf'; builder.save(str(path))
    return str(path)


def square(pen):
    pen.moveTo((100, 0)); pen.lineTo((100, 700)); pen.lineTo((500, 700)); pen.lineTo((500, 0)); pen.closePath()


def stroke(pen):  # out and back along one line: a contour that encloses no area, as a single-line font draws
    pen.moveTo((100, 0)); pen.lineTo((100, 700)); pen.lineTo((100, 0)); pen.closePath()


class PreviewTests(unittest.TestCase):
    def test_a_face_sets_its_family_name_in_capitals_or_its_own_letters_when_it_cannot(self):
        latin = {ord(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'}
        self.assertEqual(P.wording(latin, 'Mr Eaves XL'), 'Mr Eaves XL')
        self.assertEqual(P.wording({ord(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}, 'Kayak'), 'KAYAK')  # capitals only
        self.assertEqual(P.wording(latin, '花園肉丸'), 'ABCDEFGHIJKLMNOP')  # a name in another script
        self.assertEqual(P.wording({ord(c) for c in 'Rose Knight'}, 'Rose & Knight'), 'KReghinost')  # a sign it lacks: its own letters
        self.assertEqual(P.wording(set(), 'Anything'), '')

    def test_a_captured_face_shows_its_name_line_from_whichever_archive_holds_it_else_a_word_line(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot, live = Path(folder) / 'snapshot', Path(folder) / 'live'
            line = lambda face, recipe: {'faceId': face, 'recipeId': recipe, 'image': {'path': f'images/{face}-{recipe}.png'}, 'region': {'x': 1, 'y': 2, 'width': 3, 'height': 4}}
            for archive, records in ((snapshot, [line('a', 'words-a-v1'), line('a', 'latin-lower-v1'), line('b', 'words-b-v1'), line('c', 'digits-v1')]), (live, [line('a', 'name-v1')])):
                archive.mkdir(); (archive / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            found = P.captured([snapshot, live])
            self.assertEqual(found['a'], ('image', str(live / 'images/a-name-v1.png'), (1, 2, 4, 6)))
            self.assertEqual(found['b'][1], str(snapshot / 'images/b-words-b-v1.png'))
            self.assertEqual(found['c'][1], str(snapshot / 'images/c-digits-v1.png'))
            self.assertEqual(P.captured([]), {})

    def test_black_ink_is_cut_at_the_middle_and_a_faint_face_keeps_every_mark_clearly_darker_than_white(self):
        black = Image.new('L', (8, 8), 255); black.putpixel((0, 0), 0); black.putpixel((1, 0), 140)
        self.assertEqual(P.cut(black, False), 128)
        grey = Image.new('L', (8, 8), 255); grey.putpixel((0, 0), 200)
        self.assertAlmostEqual(P.cut(grey, True), 200 + 55 * 5 / 6)

    def test_a_picture_in_grey_ink_keeps_its_letters_and_one_in_black_is_cut_as_before(self):
        with tempfile.TemporaryDirectory() as folder:
            P.OUT = Path(folder)
            for fill in (0, 200):  # black ink, then grey
                image = Image.new('L', (400, 80), 255); ImageDraw.Draw(image).rectangle((20, 10, 380, 70), fill=fill)
                source = Path(folder) / f'{fill}.png'; image.save(source)
                face_id, size, reason = P.picture((f'face-{fill}', ('image', str(source), None)))
                self.assertIsNone(reason, fill); picture = Image.open(P.path(face_id))
                self.assertEqual(picture.height, P.HEIGHT); self.assertEqual([v for v, n in enumerate(picture.convert('L').histogram()) if n], [0, 255], 'black and white only')
                self.assertEqual(picture.convert('L').getpixel((picture.width // 2, P.HEIGHT // 2)), 0, 'the ink stays ink')

    def test_a_face_that_encloses_no_area_says_so_and_a_filled_one_is_not_faint(self):
        with tempfile.TemporaryDirectory() as folder:
            canvas, faint = P.line(font_file(folder, square), 400, False, 'Test')
            self.assertFalse(faint); self.assertEqual(canvas.getextrema()[0], 0)
            with self.assertRaisesRegex(ValueError, 'enclose next to no area'): P.line(font_file(folder, stroke), 400, False, 'Test')

    def test_a_hinting_program_freetype_cannot_run_is_dropped_and_the_face_still_renders(self):
        with tempfile.TemporaryDirectory() as folder:
            broken = font_file(folder, square, fpgm=b'\x91')  # 0x91 is no TrueType instruction, as in Ume P Gothic S4
            with self.assertRaisesRegex(OSError, 'invalid opcode'): P.setting(broken, 'Quiet', 400, False, 96)
            canvas, faint = P.line(broken, 400, False, 'Test')
            self.assertFalse(faint); self.assertEqual(canvas.getextrema()[0], 0)
            self.assertNotIn('fpgm', TTFont(P.unhinted(broken)))


if __name__ == '__main__':
    unittest.main()
