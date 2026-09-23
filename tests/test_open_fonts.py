import tempfile
import unittest
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables import otTables

from scripts.open_fonts import colour_letters, symbol_encoded


def build(folder, names, coloured=()):
    """A tiny TrueType font mapping A and B to glyphs of the given names; `coloured` get COLR layers."""
    pen = TTGlyphPen(None); pen.moveTo((0, 0)); pen.lineTo((0, 500)); pen.lineTo((400, 0)); pen.closePath()
    square = pen.glyph()
    order = ['.notdef', *names, 'layer']
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({ord('A'): names[0], ord('B'): names[1]})
    fb.setupGlyf({name: square for name in order})
    fb.setupHorizontalMetrics({name: (500, 0) for name in order})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({'familyName': 'Test', 'styleName': 'Regular'})
    fb.setupOS2(); fb.setupPost()
    if coloured:
        fb.setupCPAL([[(0, 0, 0, 1)]])
        fb.setupCOLR({name: [('layer', 0)] for name in coloured}, version=0)
    path = Path(folder) / f'{"-".join(names)}.ttf'
    fb.save(str(path))
    return path


class SymbolEncoding(unittest.TestCase):
    def test_letters_named_as_themselves_are_text(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(symbol_encoded(build(folder, ['A', 'uni0042'])))

    def test_letters_named_after_other_characters_are_symbols(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(symbol_encoded(build(folder, ['Alpha', 'Beta'])))
            self.assertTrue(symbol_encoded(build(folder, ['a10', 'a11'])))  # Zapf Dingbats naming

    def test_names_that_denote_nothing_standard_decide_nothing(self):
        # Euler Math names A "mupA"; CID-keyed Unifont names it "cid00066". Neither is a symbol font.
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(symbol_encoded(build(folder, ['mupA', 'mupB'])))
            self.assertFalse(symbol_encoded(build(folder, ['cid00066', 'cid00067'])))


class ColourLetters(unittest.TestCase):
    def test_colour_on_letters_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(colour_letters(build(folder, ['A', 'B'], coloured=['A'])))

    def test_colour_only_on_unmapped_alternates_does_not(self):
        # OpenDyslexic colours stylistic-set alternates; the letters themselves render plain.
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(colour_letters(build(folder, ['A', 'B'], coloured=['layer'])))


if __name__ == '__main__':
    unittest.main()
