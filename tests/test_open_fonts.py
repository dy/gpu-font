import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from scripts.open_fonts import MANIFEST, ROOT, colour_letters, debian_licences, decode_webfonts, family_name, is_font, members, name_key, one_format, split_collections, symbol_encoded


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


class ArchiveCache(unittest.TestCase):
    save = Path.write_bytes

    @staticmethod
    def killed(path, data):
        """Path.write_bytes in a process killed mid-write: part of the file lands, then nothing more runs."""
        ArchiveCache.save(path, data[:4]); raise KeyboardInterrupt

    def test_a_run_killed_while_saving_a_download_leaves_no_half_archive_to_trust(self):
        import scripts.open_fonts as open_fonts
        from unittest import mock
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(open_fonts, 'STORE', Path(folder)), mock.patch('time.sleep'), \
             mock.patch.object(open_fonts.corpus, 'request', return_value=b'PK whole archive') as request:
            with mock.patch.object(Path, 'write_bytes', self.killed), self.assertRaises(KeyboardInterrupt): open_fonts.fetch('https://host/a.zip', None)
            self.assertEqual(open_fonts.fetch('https://host/a.zip', None)[0], b'PK whole archive')  # the next run downloads it again, whole
            self.assertEqual(open_fonts.fetch('https://host/a.zip', None)[0], b'PK whole archive')  # and then reads it from the cache
            self.assertEqual(request.call_count, 2)
            self.assertEqual(len(list((Path(folder) / '.archives').iterdir())), 1)  # no partial file left beside it

    def test_a_font_from_an_api_killed_while_saving_is_downloaded_again(self):
        import scripts.open_fonts as open_fonts
        from unittest import mock
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(open_fonts, 'STORE', Path(folder)), mock.patch('time.sleep'), \
             mock.patch.object(open_fonts.corpus, 'request', return_value=b'OTTO whole font') as request:
            with mock.patch.object(Path, 'write_bytes', self.killed), self.assertRaises(KeyboardInterrupt): open_fonts.stored_font('a/A.otf', 'https://cdn/A.otf')
            self.assertEqual(open_fonts.stored_font('a/A.otf', 'https://cdn/A.otf'), b'OTTO whole font')
            self.assertEqual(request.call_count, 2)
            self.assertEqual(sorted(path.name for path in (Path(folder) / 'a').iterdir()), ['A.otf'])


class ManualFile(unittest.TestCase):
    def test_a_file_collected_by_hand_is_read_in_place_and_pinned_by_its_hash(self):
        import hashlib, scripts.open_fonts as open_fonts
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / '.data/manual').mkdir(parents=True); (Path(folder) / '.data/manual/a.zip').write_bytes(b'PK font')
            root, open_fonts.ROOT = open_fonts.ROOT, Path(folder)
            try:
                digest = hashlib.sha256(b'PK font').hexdigest()
                self.assertEqual(open_fonts.fetch('manual/a.zip', None), (b'PK font', digest))
                self.assertEqual(open_fonts.fetch('manual/a.zip', digest), (b'PK font', digest))
                with self.assertRaisesRegex(ValueError, 'changed since it was pinned'): open_fonts.fetch('manual/a.zip', '0' * 64)
                with self.assertRaisesRegex(FileNotFoundError, 'collected by hand'): open_fonts.fetch('manual/gone.zip', None)
                with self.assertRaisesRegex(ValueError, 'must stay inside'): open_fonts.fetch('manual/../secret.txt', None)  # the listing is data, not a path to trust
            finally: open_fonts.ROOT = root


class Members(unittest.TestCase):
    def test_a_loose_file_is_its_own_member_named_without_query(self):
        self.assertEqual(list(members(b'font', 'https://host/fonts/FSEX302.ttf?raw=1', 'file')), [('FSEX302.ttf', b'font')])
        self.assertEqual(list(members(b'f', 'https://raw/x/Sym%5BFILL%2Cwght%5D.ttf', 'file')), [('Sym[FILL,wght].ttf', b'f')])

    def test_the_bytes_decide_the_format_not_the_url(self):
        import io, tarfile, zipfile
        files = {'X/A.ttf': b'\x00\x01\x00\x00a', 'X/OFL.txt': b'licence'}
        zipped = io.BytesIO()
        with zipfile.ZipFile(zipped, 'w') as archive:
            for name, content in files.items(): archive.writestr(name, content)
        tarred = io.BytesIO()
        with tarfile.open(fileobj=tarred, mode='w:gz') as archive:
            for name, content in files.items(): info = tarfile.TarInfo(name); info.size = len(content); archive.addfile(info, io.BytesIO(content))
        for url, format in [('https://www.fontsquirrel.com/fonts/download/Verily-Serif-Mono', None),
                            ('https://tobiasjung.name/downloadfile.php?file=ProFont-Windows-Bold.zip', None), ('https://gitlab.com/a/b.git', 'git')]:
            self.assertEqual(dict(members(zipped.getvalue(), url, format)), files, url)
        self.assertEqual(dict(members(tarred.getvalue(), 'https://host/release.zip', None)), files)  # named zip, but a tar
        self.assertEqual(list(members(b'OTTOa', 'https://host/fonts/A.otf?dl=1')), [('A.otf', b'OTTOa')])
        self.assertEqual(list(members(b'wOF2a', 'https://host/A.woff2')), [('A.woff2', b'wOF2a')])  # decode_webfonts unpacks it next

    def test_rar_and_7zip_archives_are_read_by_bsdtar(self):
        import struct, zlib
        files = {'X/A.ttf': b'\x00\x01\x00\x00a', 'X/OFL.txt': b'licence'}
        # A RAR 4 archive storing the files uncompressed: marker, archive header, one header per file, end block (RAR technote).
        block = lambda kind, flags, body: (lambda head: struct.pack('<H', zlib.crc32(head) & 0xFFFF) + head)(struct.pack('<BHH', kind, flags, 7 + len(body)) + body)
        rar = b'Rar!\x1a\x07\x00' + block(0x73, 0, bytes(6))
        for name, content in files.items():
            rar += block(0x74, 0x8000, struct.pack('<IIBIIBBHI', len(content), len(content), 0, zlib.crc32(content), 0x21, 20, 0x30, len(name), 0x20) + name.encode()) + content
        rar += block(0x7B, 0x4000, b'')
        self.assertEqual(dict(members(rar, 'https://www.behance.net/download/FUNGIS.rar')), files)
        with self.assertRaises(subprocess.CalledProcessError): list(members(rar[:40], 'https://host/cut-short.rar'))  # a listed row reports and skips it
        with tempfile.TemporaryDirectory() as folder:
            for name, content in files.items(): (Path(folder) / name).parent.mkdir(exist_ok=True); (Path(folder) / name).write_bytes(content)
            seven = subprocess.run(['bsdtar', '--format', '7zip', '-cf', '-', '-C', folder, 'X/A.ttf', 'X/OFL.txt'], check=True, capture_output=True).stdout
        self.assertEqual(dict(members(seven, 'https://host/release')), files)

    def test_a_web_page_or_an_empty_download_is_neither_archive_nor_font(self):
        with self.assertRaisesRegex(ValueError, r"cm-web-fonts/: neither an archive nor a font, but b'<!DOCTYPE html>"):
            list(members(b'<!DOCTYPE html><html>', 'http://checkmyworking.com/cm-web-fonts/'))
        with self.assertRaisesRegex(ValueError, r"neither an archive nor a font, but b''"):
            list(members(b'', 'https://host/empty'))


class Listed(unittest.TestCase):
    def test_every_catalogue_row_is_listed_and_skipped_on_failure_while_curated_sources_fail_loudly(self):
        from scripts.open_fonts import SOURCES
        rows = [row for row in json.loads((ROOT / 'bench/open-releases.json').read_text())['releases'] if row.get('include')]
        self.assertEqual(sum(bool(spec.get('listed')) for spec in SOURCES), len(rows))
        self.assertEqual([spec for spec in SOURCES[:len(SOURCES) - len(rows)] if spec.get('listed')], [])  # the curated sources come first

    def test_a_listed_family_links_to_the_page_that_lists_it(self):
        from scripts.open_fonts import SOURCES
        pages = {spec.get('page') for spec in SOURCES if spec['source'] == 'collletttivo'}
        self.assertIn('https://www.collletttivo.it/typefaces/ribes/', pages)  # the link Collletttivo asked for
        self.assertEqual([spec['source'] for spec in SOURCES if not spec.get('listed') and spec.get('page')], [])  # curated sources keep their home


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


class DebianSkip(unittest.TestCase):
    def test_bitmap_noto_and_tex_live_extras_are_skipped_but_its_two_font_packages_are_taken(self):
        from scripts.open_fonts import DEBIAN_SKIP
        taken = ['fonts-dejavu', 'fonts-wqy-zenhei', 'texlive-fonts-extra', 'texlive-fonts-recommended']
        skipped = ['xfonts-base', 'fonts-noto-cjk', 'texlive-fonts-extra-doc', 'texlive-fonts-recommended-doc', 'texlive-latex-base']
        self.assertEqual([name for name in taken if DEBIAN_SKIP.match(name)], [])
        self.assertEqual([name for name in skipped if not DEBIAN_SKIP.match(name)], [])


class ClaimOrder(unittest.TestCase):
    def test_rights_holders_then_fontshare_then_catalogues_then_debian_then_the_long_tail(self):
        from scripts.open_fonts import CATALOGUES, LONG_TAIL, claim_order
        specs = [{'source': 'github'}, {'source': 'fontlibrary'}, {'source': 'velvetyne'}, {'source': 'use-and-modify'}, {'source': 'dejavu'}]
        order = [spec['source'] for spec in claim_order(specs, [{'source': 'debian'}])]
        self.assertEqual(order, ['velvetyne', 'dejavu', 'fontshare', 'fontlibrary', 'use-and-modify', 'debian', 'github'])
        without = [spec['source'] for spec in claim_order(specs, [{'source': 'debian'}], long_tail=False)]
        self.assertEqual(without, order[:-1])  # everything else, in the same order


class LongTailCopies(unittest.TestCase):
    RAW = 'https://raw.githubusercontent.com/a/b/c0ffee/fonts/'
    held = staticmethod(lambda key: key in {'playfairdisplay', 'inter'})

    def test_a_file_named_after_a_held_family_or_patched_by_nerd_fonts_is_a_copy(self):
        from scripts.open_fonts import copied
        for name in ['PlayfairDisplay-BoldItalic.ttf', 'PlayfairDisplay%5Bwght%5D.ttf', 'playfair_display.otf?raw=1', 'inter.ttf',
                     'IosevkaSS10NerdFont-Bold.ttf', 'Hack Regular Nerd Font Complete Mono.ttf', 'JetBrainsMonoNLNerdFontPropo-Italic.ttf']:
            self.assertTrue(copied(self.RAW + name, self.held), name)
        for name in ['Mine-Regular.ttf', 'c.ttf', 'Interstate-Bold.otf', 'PlayfairDisplaySC.ttf']:  # a name only starting like a held one is not it
            self.assertFalse(copied(self.RAW + name, self.held), name)

    def test_a_row_keeps_its_own_fonts_and_their_licence_and_drops_a_row_of_copies_whole(self):
        from scripts.open_fonts import long_tail_files
        row = lambda *names: {'include': r'\.(otf|ttf)$', 'files': [self.RAW + name for name in names]}
        self.assertEqual(long_tail_files(row('PlayfairDisplay-Bold.ttf', 'OFL.txt'), self.held), [])
        self.assertEqual(long_tail_files(row('Mine-Regular.ttf', 'PlayfairDisplay-Bold.ttf', 'OFL.txt'), self.held), [self.RAW + 'Mine-Regular.ttf', self.RAW + 'OFL.txt'])
        self.assertEqual(long_tail_files(row('OFL.txt'), self.held), [])  # a licence alone is nothing to inventory


class SplitCollections(unittest.TestCase):
    def test_a_collection_becomes_its_fonts_and_plain_files_pass_through(self):
        import io
        from fontTools.ttLib import TTCollection, TTFont
        with tempfile.TemporaryDirectory() as folder:
            collection = TTCollection()
            collection.fonts = [TTFont(build(folder, ['A', 'B'])), TTFont(build(folder, ['uni0041', 'uni0042']))]
            data = io.BytesIO(); collection.save(data)
            out = list(split_collections([('x/Pair.ttc', data.getvalue()), ('x/Lone.ttf', b'\x00\x01\x00\x00'), ('x/Bad.ttc', b'ttcf broken')]))
            self.assertEqual([name for name, _ in out], ['x/Pair-0.ttf', 'x/Pair-1.ttf', 'x/Lone.ttf'])
            for index, (name, content) in enumerate(out[:2]):
                path = Path(folder) / f'split-{index}.ttf'; path.write_bytes(content)
                self.assertEqual(family_name(path), 'Test')


class DecodeWebfonts(unittest.TestCase):
    def test_a_web_font_becomes_the_font_it_compresses(self):
        import io
        from fontTools.ttLib import TTFont
        with tempfile.TemporaryDirectory() as folder:
            font = TTFont(build(folder, ['A', 'B'])); font.flavor = 'woff2'
            packed = io.BytesIO(); font.save(packed)
            out = list(decode_webfonts([('web/Icons.woff2', packed.getvalue()), ('x/Plain.ttf', b'\x00\x01\x00\x00'), ('web/Bad.woff2', b'wOF2 broken')]))
            self.assertEqual([name for name, _ in out], ['web/Icons.ttf', 'x/Plain.ttf'])
            path = Path(folder) / 'decoded.ttf'; path.write_bytes(out[0][1])
            self.assertTrue(is_font(out[0][1]))
            self.assertEqual(family_name(path), 'Test')


class OneFormat(unittest.TestCase):
    def test_a_style_in_two_formats_is_one_face_and_opentype_wins(self):
        given = [('x/Foo-Bold.ttf', b't'), ('x/Foo-Bold.otf', b'o'), ('x/Foo-Italic.ttf', b'i'), ('x/OFL.txt', b'l'), ('y/Foo-Bold.ttf', b'y')]
        self.assertEqual(sorted(one_format(given)), [('x/Foo-Bold.otf', b'o'), ('x/Foo-Italic.ttf', b'i'), ('x/OFL.txt', b'l'), ('y/Foo-Bold.ttf', b'y')])
        self.assertEqual(one_format([]), [])


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
