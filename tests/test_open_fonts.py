import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from scripts.open_fonts import MANIFEST, ROOT, colour_letters, debian_licences, family_name, is_font, members, name_key, symbol_encoded


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


class FamilyName(unittest.TestCase):
    def test_a_font_declares_its_family_and_an_unreadable_one_has_none(self):
        from fontTools.ttLib import TTFont
        with tempfile.TemporaryDirectory() as folder:
            path = build(folder, ['A', 'B'])
            self.assertEqual(family_name(path), 'Test')
            font = TTFont(path); del font['name']; font.save(path)  # a real Debian font ships like this
            self.assertIsNone(family_name(path))
            path.write_bytes(b'\x00\x01\x00\x00 truncated')
            self.assertIsNone(family_name(path))


class FontSignature(unittest.TestCase):
    def test_fonts_are_known_by_signature_not_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(is_font(build(folder, ['A', 'B']).read_bytes()))  # TrueType
        self.assertTrue(is_font(b'OTTO\x00\x0b'))  # CFF OpenType
        # What repositories commit under font names: a macOS "._" fork, a Git LFS pointer, an error page.
        self.assertFalse(is_font(b'\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X'))
        self.assertFalse(is_font(b'version https://git-lfs.github.com/spec/v1\noid sha256:'))
        self.assertFalse(is_font(b'<!DOCTYPE html>'))
        self.assertFalse(is_font(b''))


class StoredFont(unittest.TestCase):
    def test_a_non_font_answer_is_refused_and_a_font_is_fetched_once(self):
        import scripts.open_fonts as open_fonts
        calls = []
        answers = {'https://cdn/type1.ttf': b'\x80\x01%!PS-AdobeFont-1.0', 'https://cdn/real.ttf': b'OTTO font'}
        fake = lambda url: calls.append(url) or answers[url]
        with tempfile.TemporaryDirectory() as folder:
            store, request = open_fonts.STORE, open_fonts.corpus.request
            open_fonts.STORE, open_fonts.corpus.request = Path(folder), fake
            try:
                self.assertIsNone(open_fonts.stored_font('a/type1.ttf', 'https://cdn/type1.ttf'))
                self.assertFalse((Path(folder) / 'a/type1.ttf').exists())
                self.assertEqual(open_fonts.stored_font('a/real.ttf', 'https://cdn/real.ttf'), b'OTTO font')
                self.assertEqual(open_fonts.stored_font('a/real.ttf', 'https://cdn/real.ttf'), b'OTTO font')
                self.assertEqual(calls, ['https://cdn/type1.ttf', 'https://cdn/real.ttf'])
            finally:
                open_fonts.STORE, open_fonts.corpus.request = store, request


class Members(unittest.TestCase):
    def test_a_loose_file_is_its_own_member_named_without_query(self):
        self.assertEqual(list(members(b'font', 'https://host/fonts/FSEX302.ttf?raw=1', 'file')), [('FSEX302.ttf', b'font')])


class DebianPackage(unittest.TestCase):
    def test_a_deb_yields_its_installed_files_and_its_licences(self):
        import tarfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'debian-binary').write_text('2.0\n')
            with tarfile.open(root / 'control.tar.gz', 'w:gz'): pass
            files = {'usr/share/fonts/truetype/x/A.ttf': b'\x00\x01\x00\x00font', 'usr/share/doc/fonts-x/copyright':
                     b'Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n\nFiles: *\nLicense: OFL-1.1\n\nFiles: debian/*\nLicense: GPL-2+\n\nLicense: OFL-1.1\n full text\n'}
            with tarfile.open(root / 'data.tar.gz', 'w:gz') as data:
                for name, content in files.items():
                    path = root / 'tree' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
                    data.add(path, arcname='./' + name)
            # The common ar format dpkg writes (macOS's ar adds a BSD symbol table instead).
            ar = b'!<arch>\n' + b''.join(
                b'%-16s%-12d%-6d%-6d%-8s%-10d`\n' % (name.encode(), 0, 0, 0, b'100644', len(body)) + body + b'\n' * (len(body) % 2)
                for name in ['debian-binary', 'control.tar.gz', 'data.tar.gz'] for body in [(root / name).read_bytes()])
            got = dict(members(ar, 'https://deb/fonts-x.deb', 'deb'))
            self.assertEqual(got, files)
            self.assertEqual(debian_licences(files['usr/share/doc/fonts-x/copyright']), 'OFL-1.1; GPL-2+')
            self.assertEqual(debian_licences(b'no machine-readable fields'), 'see the package copyright file')


class GitPin(unittest.TestCase):
    def test_a_commit_is_archived_under_the_prefix_includes_expect_and_its_pin_holds(self):
        import scripts.open_fonts as open_fonts
        git = lambda *args, cwd: subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder) / 'karrik.git'
            repo.mkdir(); (repo / 'fonts').mkdir(); (repo / 'fonts/Karrik.otf').write_bytes(b'OTTO')
            git('init', '-q', cwd=repo); git('add', '.', cwd=repo)
            git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'font', cwd=repo)
            commit = git('rev-parse', 'HEAD', cwd=repo)
            store, open_fonts.STORE = open_fonts.STORE, Path(folder) / 'store'
            try:
                data, digest = open_fonts.fetch(str(repo), None, commit)
                self.assertEqual(list(members(data, str(repo), 'git')), [(f'karrik-{commit}/fonts/Karrik.otf', b'OTTO')])
                self.assertEqual(open_fonts.fetch(str(repo), digest, commit)[1], digest)
                with self.assertRaises(ValueError): open_fonts.fetch(str(repo), '0' * 64, commit)
            finally:
                open_fonts.STORE = store


class Inventory(unittest.TestCase):
    """Invariants of the committed bench/open-fonts.json."""

    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(MANIFEST.read_text())
        cls.google = {name_key(f['family']) for f in json.loads((ROOT / 'bench/corpus.json').read_text())['families']}

    def test_no_family_duplicates_one_google_fonts_carries(self):
        twice = [f['family'] for f in self.inventory['families'] if name_key(f['family']) in self.google]
        self.assertEqual(twice, [])

    def test_no_family_is_taken_from_two_sources(self):
        names = [name_key(f['family']) for f in self.inventory['families']]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])

    def test_ids_are_unique_and_each_family_selects_one_of_its_faces(self):
        ids = [f['id'] for f in self.inventory['families']]
        self.assertEqual(len(ids), len(set(ids)))
        for family in self.inventory['families']:
            self.assertIn(family['selected'], [face['path'] for face in family['faces']], family['family'])

    def test_the_committed_inventory_stays_small(self):
        # Letters are counted here and kept in full beside the files; licence text is kept once per family.
        self.assertLess(MANIFEST.stat().st_size, 12_000_000)
        for family in self.inventory['families']:
            self.assertNotIn('alphabets', family)
            self.assertTrue(all(isinstance(count, int) and count >= 8 for count in family['letters'].values()), family['family'])
            self.assertTrue(all('embeddedLicense' not in face for face in family['faces']), family['family'])

    def test_every_archive_is_pinned_by_hash(self):
        # The Fontshare and Fontsource APIs are listings, not archives; their files are pinned per face by blob.
        unpinned = [a['url'] for a in self.inventory['archives'] if a['source'] not in ('fontshare', 'fontsource') and len(a.get('sha256', '')) != 64]
        self.assertEqual(unpinned, [])


if __name__ == '__main__':
    unittest.main()
