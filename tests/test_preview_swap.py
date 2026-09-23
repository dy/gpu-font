import unittest

import numpy as np

from scripts.preview_swap import measure, ranked, Grouped


def unit(v):
    v = np.asarray(v, np.float64); return v/np.linalg.norm(v)


class PreviewSwapTests(unittest.TestCase):
    def test_held_out_phrase_and_page_captures_never_serve_as_references(self):
        e0, e1 = np.eye(4)[0], np.eye(4)[1]; query = unit(.8*e0 + .6*e1)
        rows = [('a', 'latin-lower-v1', e0), ('a', 'latin-upper-v1', e0), ('a', 'words-a-v1', query), ('a', 'words-b-v1', e0),
                # B holds the query's very pixels under the same phrase and as a page capture: both must stay out.
                ('b', 'latin-lower-v1', e1), ('b', 'latin-upper-v1', e1), ('b', 'words-a-v1', query), ('b', 'words-b-v1', e1), ('b', 'provided-v1', query)]
        records = [{'familyId': f, 'recipeId': r, 'source': {'key': 'x'}} for f, r, _ in rows]; vectors = np.stack([unit(v) for *_, v in rows])
        everyone = lambda r: True
        out = measure(records, vectors, {}, everyone)
        # A's words-a query finds A (0.8) over B (0.6); B's own words-a query shares A's pixels, so A wins it: 1 of 2 right.
        self.assertEqual((out['queries'], out['top1'], out['top5']), (4, .75, 1.0))
        self.assertEqual(out['misses'], [])
        # Lineage credits a relative ranked first; strict top-1 does not.
        related = measure(records, vectors, {'b': {'a'}, 'a': {'b'}}, everyone)
        self.assertEqual((related['top1'], related['lineage1']), (.75, 1.0))
        # Alphabet references only: B's queries now meet only alphabets.
        self.assertEqual(measure(records, vectors, {}, everyone, references={'latin-lower-v1', 'latin-upper-v1'})['queries'], 4)
        self.assertIsNone(measure(records, vectors, {}, lambda r: False))

    def test_a_merged_catalog_ranks_families_by_their_best_row(self):
        refs = np.stack([unit([1, 0]), unit([0, 1]), unit([1, 1])]); google = Grouped(refs, ['g2', 'g1', 'g2'])
        self.assertEqual(google.names, ['g1', 'g2'])
        self.assertEqual(ranked(unit([1, 0]), np.stack([unit([.9, .1])]), ['own'], google), ['g2', 'own', 'g1'])
        self.assertEqual(ranked(unit([0, 1]), refs[:0], [], google), ['g1', 'g2'])


if __name__ == '__main__': unittest.main()
