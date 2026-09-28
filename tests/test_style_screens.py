import unittest

from train.style_screens import site_role, script_of, family_of, chosen, normalize


class ScreenViewTests(unittest.TestCase):
    def test_a_site_is_held_out_by_the_hash_of_its_origin_as_the_read_holds_it(self):
        # The same origins tests/screens.test.mjs pins for scripts/screens.mjs's siteRole: one rule in two languages.
        self.assertEqual([site_role(o) for o in ['https://www.chess.com', 'https://m.youtube.com', 'https://site3.test', 'https://site13.test']], ['development', 'held-out', 'development', 'held-out'])
        held = sum(site_role(f'https://site{i}.test') == 'held-out' for i in range(2000)); self.assertTrue(150 < held < 250, held)

    def test_the_script_is_the_one_most_letters_are_in(self):
        self.assertEqual([script_of(t) for t in ['Quiet rivers', 'ЗАГС Северодвинска', 'Αθήνα', '東京都', 'こんにちは', 'مرحبا', 'Tokyo 東京 2026']], ['Latn', 'Cyrl', 'Grek', 'Hani', 'Hira', 'Arab', 'Latn'])
        self.assertIsNone(script_of('123 !')); self.assertIsNone(script_of('')); self.assertIsNone(script_of('ᐃᓄᒃᑎᑐᑦ'))  # syllabics: not a script the views use

    def test_a_drawn_font_belongs_to_its_family_and_a_record_labels_a_face(self):
        families = {normalize('Roboto'): 'roboto', normalize('Roboto Condensed'): 'roboto-condensed'}
        self.assertEqual([family_of(n, families) for n in ['Roboto', 'Roboto Medium', 'Roboto Condensed', 'Roboto 500', 'Helvetica', '', '.\x7f', None]], ['roboto', 'roboto', 'roboto-condensed', 'roboto', None, None, None, None])
        record = lambda origin, **css: {'origin': origin, 'font': {'family': 'Roboto Medium'}, 'text': 'Play chess', 'css': {'fontWeight': '450', 'fontStyle': 'normal', 'fontSize': 16, **css}}
        picked = chosen([record('https://www.chess.com'), record('https://m.youtube.com'), record('https://www.chess.com', fontWeight='bold', fontStyle='italic')], families)
        self.assertEqual([p[1:] for p in picked], [('roboto', 'Latn', 400, False), ('roboto', 'Latn', 400, True)])  # the held-out site's record is left out; 'bold' reads as the default weight
        self.assertEqual(chosen([{**record('https://www.chess.com'), 'text': '!!!'}], families), [])
