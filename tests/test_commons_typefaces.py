import json, tempfile, unittest
from pathlib import Path
from unittest import mock

from scripts import commons_typefaces as commons

# A small Commons: Arial is a typeface filed inside Helvetica, a comparison sits beside the photos,
# one photo is filed under both typefaces, and the deepest category is two levels down.
TREE = {
    'Category:Typefaces by name': {'subcat': ['Category:Helvetica', 'Category:Arial']},
    'Category:Helvetica': {'file': ['File:Sign.jpg', 'File:Specimen.svg'],
                           'subcat': ['Category:Arial', 'Category:Road signs using Helvetica', 'Category:Helvetica vs. Bodoni']},
    'Category:Road signs using Helvetica': {'file': ['File:Road.JPEG', 'File:Shared.jpg'], 'subcat': ['Category:Streets of a town']},
    'Category:Streets of a town': {'file': ['File:Street.jpg']},
    'Category:Helvetica vs. Bodoni': {'file': ['File:Comparison.jpg']},
    'Category:Arial': {'file': ['File:Arial poster.jpg', 'File:Shared.jpg']},
}


class Listing(unittest.TestCase):
    def test_each_photo_takes_its_nearest_typeface_one_level_down(self):
        members = lambda category, kind: TREE.get(category, {}).get(kind, [])
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(commons, 'members', members), \
             mock.patch.object(commons, 'OUT', Path(folder)), mock.patch('builtins.print'):
            commons.listing()
            rows = [json.loads(line) for line in (Path(folder) / 'files.jsonl').open()]
        self.assertEqual({row['title']: row['typeface'] for row in rows},
                         {'File:Sign.jpg': 'Helvetica', 'File:Road.JPEG': 'Helvetica', 'File:Arial poster.jpg': 'Arial'})
        self.assertEqual(next(row['path'] for row in rows if row['title'] == 'File:Road.JPEG'), ['Category:Road signs using Helvetica'])
        # Left out: the SVG specimen, the comparison, the street two levels down, and the photo filed under both typefaces.

    def test_names_comparisons_and_photos(self):
        self.assertEqual([commons.typeface(c) for c in ['Category:Futura (typeface)', 'Category:Jura (font)', 'Category:Helvetica']], ['Futura', 'Jura', 'Helvetica'])
        self.assertTrue(commons.SKIP.search('Category:Helvetica vs. Bodoni'))
        self.assertFalse(commons.SKIP.search('Category:Road signs using Helvetica'))
        self.assertEqual([bool(commons.JPEG.search(t)) for t in ['File:a.JPG', 'File:a.jpeg', 'File:a.svg', 'File:a.jpg.png']], [True, True, False, False])
        camera = [{'name': 'Make', 'value': 'Canon'}]
        self.assertEqual([commons.photo({'mime': 'image/jpeg', 'commonmetadata': camera}), commons.photo({'mime': 'image/jpeg', 'commonmetadata': []}),
                          commons.photo({'mime': 'image/png', 'commonmetadata': camera})], [True, False, False])


if __name__ == '__main__':
    unittest.main()
