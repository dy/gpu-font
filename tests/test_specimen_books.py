import unittest

from scripts.specimen_books import public_domain, specimen_book


class SpecimenBook(unittest.TestCase):
    def test_a_specimen_book_before_1929_is_taken(self):
        self.assertTrue(specimen_book({'title': 'American specimen book of type styles', 'year': '1912'}))
        self.assertTrue(specimen_book({'title': 'Le livret typographique; spécimen de caractères', 'year': 1886}))
        self.assertTrue(specimen_book({'title': 'A specimen', 'year': 1734}))  # Caslon's first sheet

    def test_later_books_undated_books_and_fine_printing_are_left_out(self):
        self.assertFalse(specimen_book({'title': 'Specimen book of type faces', 'year': '1929'}))  # still in copyright in the US
        self.assertFalse(specimen_book({'title': 'Specimen book of type faces', 'year': None}))
        self.assertFalse(specimen_book({'title': 'The faerie queene', 'year': 1897}))  # filed under printing specimens

class PublicDomain(unittest.TestCase):
    def test_the_archive_s_mark_or_an_unmarked_date_before_1929_decides(self):
        self.assertTrue(public_domain({'possible-copyright-status': 'NOT_IN_COPYRIGHT', 'date': '1950'}))  # the Archive's own judgement
        self.assertTrue(public_domain({'possible-copyright-status': None, 'date': '1923'}))  # American Type Founders' 1923 catalogue
        self.assertTrue(public_domain({'date': '1882-01-01'}))
        self.assertFalse(public_domain({'date': '1929'}))
        self.assertFalse(public_domain({'possible-copyright-status': 'POSSIBLE_COPYRIGHT', 'date': '1900'}))  # a mark against it wins
        self.assertFalse(public_domain({'date': '[19--]'}))  # no year: no claim
        self.assertFalse(public_domain({}))

if __name__ == '__main__':
    unittest.main()
