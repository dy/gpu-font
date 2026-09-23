import unittest

from scripts.open_releases import choose


class ReleaseFolder(unittest.TestCase):
    def test_the_folder_with_most_font_files_wins(self):
        paths = ['fonts/otf/A-Regular.otf', 'fonts/otf/A-Bold.otf', 'fonts/otf/A-Italic.otf', 'extra/A-Display.otf']
        self.assertEqual(choose(paths), ('fonts/otf', '.otf'))

    def test_opentype_wins_a_tie_with_truetype(self):
        paths = ['fonts/ttf/A-Regular.ttf', 'fonts/ttf/A-Bold.ttf', 'fonts/otf/A-Regular.otf', 'fonts/otf/A-Bold.otf']
        self.assertEqual(choose(paths), ('fonts/otf', '.otf'))

    def test_web_builds_sources_and_old_versions_are_never_the_release(self):
        paths = ['webfonts/A.ttf', 'webfonts/B.ttf', 'webfonts/C.ttf', 'sources/A.ttf', 'old/A.otf', 'old/B.otf', 'fonts/A.otf']
        self.assertEqual(choose(paths), ('fonts', '.otf'))

    def test_a_repository_with_only_sources_or_web_files_has_no_release(self):
        self.assertIsNone(choose(['sources/A.glyphs', 'web/A.woff2', 'web/A.ttf', 'README.md']))
        self.assertIsNone(choose([]))

    def test_files_at_the_root_are_a_release(self):
        self.assertEqual(choose(['A-Regular.otf', 'README.md']), ('.', '.otf'))


if __name__ == '__main__':
    unittest.main()
