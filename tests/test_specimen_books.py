import unittest

from scripts.specimen_books import specimen_book


class SpecimenBook(unittest.TestCase):
    def test_a_specimen_book_before_1929_is_taken(self):
        self.assertTrue(specimen_book({'title': 'American specimen book of type styles', 'year': '1912'}))
        self.assertTrue(specimen_book({'title': 'Le livret typographique; spécimen de caractères', 'year': 1886}))
        self.assertTrue(specimen_book({'title': 'A specimen', 'year': 1734}))  # Caslon's first sheet

    def test_later_books_undated_books_and_fine_printing_are_left_out(self):
        self.assertFalse(specimen_book({'title': 'Specimen book of type faces', 'year': '1929'}))  # still in copyright in the US
        self.assertFalse(specimen_book({'title': 'Specimen book of type faces', 'year': None}))
        self.assertFalse(specimen_book({'title': 'The faerie queene', 'year': 1897}))  # filed under printing specimens


if __name__ == '__main__':
    unittest.main()
