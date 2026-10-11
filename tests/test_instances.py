import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables.TupleVariation import TupleVariation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import instances as I


def variable_font(path, instances, axes=(('wght', 100, 100, 900, 'Weight'), ('ital', 0, 0, 1, 'Italic'))):
    """A TrueType font with one letter whose stem widens with the weight axis, and the named instances given as (style, location)."""
    fb = FontBuilder(1000, isTTF=True); fb.setupGlyphOrder(['.notdef', 'A'])
    pen = TTGlyphPen(None); pen.moveTo((0, 0)); pen.lineTo((200, 0)); pen.lineTo((100, 600)); pen.closePath()
    fb.setupGlyf({'.notdef': TTGlyphPen(None).glyph(), 'A': pen.glyph()}); fb.setupHorizontalMetrics({'.notdef': (500, 0), 'A': (500, 0)}); fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupCharacterMap({ord('A'): 'A'}); fb.setupNameTable({'familyName': 'Variable test', 'styleName': 'Thin'}); fb.setupOS2(usWeightClass=100); fb.setupPost(); fb.setupMaxp()
    fb.setupFvar(list(axes), [{'location': {tag: default for tag, _, default, _, _ in axes} | location, 'stylename': style} for style, location in instances])
    gvar = newTable('gvar'); gvar.version = 1; gvar.reserved = 0
    gvar.variations = {'A': [TupleVariation({'wght': (0, 1, 1)}, [(0, 0), (200, 0), (100, 0), (0, 0), (200, 0), (0, 0), (0, 0)])]}; fb.font['gvar'] = gvar; fb.save(str(path))


class InstanceTests(unittest.TestCase):
    def test_slope_comes_from_the_name_the_slant_or_an_italic_axis_past_its_lower_end_and_weight_from_the_axis(self):
        axes = {'wght': (100, 450, 900), 'ital': (100, 100, 900)}  # PP Monument's italic axis runs 100 to 900
        self.assertEqual(I.style_of('Condensed Bold', {'wght': 700, 'ital': 100}, axes, 400), {'weight': 700, 'italic': False, 'width': 100})
        self.assertEqual(I.style_of('Condensed Bold Italic', {'wght': 700, 'ital': 900}, axes, 400), {'weight': 700, 'italic': True, 'width': 100})
        self.assertEqual(I.style_of('Book', {'wght': 350, 'ital': 900}, axes, 400), {'weight': 350, 'italic': True, 'width': 100})
        self.assertEqual(I.style_of('Thin Italic', {'wght': 100, 'ital': 900}, {'wght': (100, 100, 900), 'ital': (900, 900, 900)}, 100), {'weight': 100, 'italic': True, 'width': 100})  # an axis with no range: the name
        self.assertEqual(I.style_of('Narrow', {'wdth': 75, 'slnt': -10}, {'wdth': (50, 100, 150), 'slnt': (-15, 0, 0)}, 500), {'weight': 500, 'italic': True, 'width': 75})  # no weight axis: the file's
        self.assertEqual(I.slug('Extended Semi-Bold Italic'), 'extended-semi-bold-italic'); self.assertEqual(I.slug('***'), 'instance')

    def test_the_weights_the_axis_spans_that_no_instance_names_are_added_at_the_other_axes_defaults_slanted_when_the_file_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'vf.ttf'; variable_font(path, [('Book', {'wght': 350}), ('Regular', {'wght': 400})], axes=(('wght', 200, 400, 700, 'Weight'), ('ital', 0, 1, 1, 'Italic')))
            with TTFont(path) as font:
                font['post'].italicAngle = -12.0
                found = I.named_instances(font)
            self.assertEqual([n for n, _ in found], ['Book', 'Regular', 'ExtraLight Italic', 'Light Italic', 'Medium Italic', 'SemiBold Italic', 'Bold Italic'])  # 400 named already; 100, 800, 900 outside the axis
            self.assertEqual(found[2][1], {'wght': 200.0, 'ital': 1.0})
            with TTFont(path) as font: self.assertEqual([n for n, _ in I.named_instances(font)][2:], ['ExtraLight', 'Light', 'Medium', 'SemiBold', 'Bold'])  # upright file

    def test_the_plan_lists_each_instance_once_with_unique_slugs_and_the_default_nearest_regular_upright(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'store/pp').mkdir(parents=True)
            variable_font(root/'store/pp/Kyoto-VF.ttf', [('Thin', {'wght': 100}), ('Regular', {'wght': 400}), ('Book', {'wght': 400}), ('Bold', {'wght': 700}), ('BOLD', {'wght': 701}), ('Bold Italic', {'wght': 700, 'ital': 1})])
            variable_font(root/'store/pp/Plain-VF.ttf', [])
            inventory = {'store': 'store', 'families': [
                {'id': 'pp-kyoto', 'selected': 'pp/Kyoto-VF.ttf', 'faces': [{'path': 'pp/Kyoto-VF.ttf', 'weight': 100, 'axes': {'wght': {}}}, {'path': 'pp/Plain-VF.ttf', 'weight': 100, 'axes': {'wght': {}}}]},
                {'id': 'gone', 'excluded': 'color', 'selected': 'pp/Kyoto-VF.ttf', 'faces': [{'path': 'pp/Kyoto-VF.ttf', 'weight': 100, 'axes': {'wght': {}}}]}]}
            with patch.object(I, 'ROOT', root): faces, families = I.plan(inventory)
            self.assertEqual(list(faces), ['pp/Kyoto-VF.ttf', 'pp/Plain-VF.ttf'])  # the excluded family is not instanced; the file without named instances gets the weight ladder
            self.assertEqual([r['id'].split('@')[1] for r in faces['pp/Kyoto-VF.ttf']], ['thin', 'regular', 'bold', 'bold-2', 'bold-italic', 'extralight', 'light', 'medium', 'semibold', 'extrabold', 'black'])  # Book, at Regular's coordinates, dropped; BOLD's slug made unique; the unnamed weights after
            self.assertEqual([r['weight'] for r in faces['pp/Plain-VF.ttf']], [100, 200, 300, 400, 500, 600, 700, 800, 900])
            bold = faces['pp/Kyoto-VF.ttf'][2]
            self.assertEqual((bold['path'], bold['styleName'], bold['weight'], bold['italic'], bold['width'], bold['axes']), ('instances/pp-kyoto/Kyoto-VF@bold.ttf', 'Bold', 700, False, 100, {'wght': 700.0, 'ital': 0.0}))
            self.assertEqual(families, {'pp-kyoto': {'default': 'pp-kyoto/Kyoto-VF@regular'}})

    def test_an_instance_is_a_static_font_at_its_coordinates_kept_once_written_and_the_manifest_leaves_out_what_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'store/pp').mkdir(parents=True); variable_font(root/'store/pp/Kyoto-VF.ttf', [('Thin', {'wght': 100}), ('Bold', {'wght': 700})])
            inventory = {'store': 'store', 'families': [{'id': 'pp-kyoto', 'selected': 'pp/Kyoto-VF.ttf', 'faces': [{'path': 'pp/Kyoto-VF.ttf', 'weight': 100, 'axes': {'wght': {}}}]}]}
            with patch.object(I, 'ROOT', root), patch.object(I, 'OPEN', root/'store/pp/Kyoto-VF.ttf'):
                manifest, failed = I.build(inventory, 1, run=lambda jobs: map(I.instance, jobs))
                self.assertEqual(failed, []); records = manifest['faces']['pp/Kyoto-VF.ttf']
                bold = root/'store'/records[1]['path']; thin = root/'store'/records[0]['path']
                with TTFont(bold) as font: self.assertNotIn('fvar', font); self.assertNotIn('gvar', font); wide = font['glyf']['A'].coordinates[1][0]
                with TTFont(thin) as font: self.assertLess(font['glyf']['A'].coordinates[1][0], wide)  # the bold stem is wider
                self.assertEqual(records[1]['blob'], I.blob(bold.read_bytes()))
                before = bold.stat().st_mtime_ns; I.build(inventory, 1, run=lambda jobs: map(I.instance, jobs)); self.assertEqual(bold.stat().st_mtime_ns, before)  # kept, not rewritten
                (root/'store/instances/pp-kyoto/stale.ttf').write_bytes(b'x'); I.build(inventory, 1, run=lambda jobs: map(I.instance, jobs))
                self.assertFalse((root/'store/instances/pp-kyoto/stale.ttf').exists())  # an instance no face lists any more is removed
                self.assertEqual(manifest['families'], {'pp-kyoto': {'default': 'pp-kyoto/Kyoto-VF@regular'}})  # the ladder's Regular, nearest 400 upright
                broken, failed = I.build(inventory, 1, run=lambda jobs: [(None, 'ValueError: no') for _ in jobs])
                self.assertEqual((broken['faces'], broken['families'], [f for f, _ in failed][:2]), ({}, {}, ['pp-kyoto/Kyoto-VF@thin', 'pp-kyoto/Kyoto-VF@bold']))

    def test_a_positioning_store_the_instancer_cannot_read_is_dropped_and_the_instance_still_made(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'vf.ttf'; variable_font(path, [('Bold', {'wght': 700})])
            with TTFont(path) as font:  # a GDEF with a variation store the instancer trips on
                from fontTools.ttLib.tables import otTables
                from fontTools.varLib.varStore import OnlineVarStoreBuilder
                builder = OnlineVarStoreBuilder(['wght', 'ital']); builder.setSupports([{'wght': (0, 1, 1)}]); builder.storeDeltas([10])
                gdef = newTable('GDEF'); gdef.table = otTables.GDEF(); gdef.table.Version = 0x00010003; gdef.table.GlyphClassDef = gdef.table.AttachList = gdef.table.LigCaretList = gdef.table.MarkAttachClassDef = gdef.table.MarkGlyphSetsDef = None; gdef.table.VarStore = builder.finish(); font['GDEF'] = gdef; font.save(str(path))
            calls = []
            real = I.instantiateVariableFont
            def flaky(font, coordinates, inplace):
                calls.append(getattr(font['GDEF'].table, 'VarStore', None) is None)
                if len(calls) == 1: raise IndexError('list index out of range')
                return real(font, coordinates, inplace=inplace)
            with unittest.mock.patch.object(I, 'instantiateVariableFont', flaky):
                data = I.instantiate(str(path), {'wght': 700.0, 'ital': 0.0})
            self.assertEqual(calls, [False, True]); self.assertNotIn('fvar', TTFont(io.BytesIO(data)))
            with unittest.mock.patch.object(I, 'instantiateVariableFont', unittest.mock.Mock(side_effect=IndexError('no'))):
                variable_font(Path(tmp)/'plain.ttf', [('Bold', {'wght': 700})])
                with self.assertRaises(IndexError): I.instantiate(str(Path(tmp)/'plain.ttf'), {'wght': 700.0, 'ital': 0.0})  # no store to drop: the error stands


if __name__ == '__main__': unittest.main()
