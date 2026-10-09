import unittest

from train.style_forum import chosen, crop, line_of, style_of, thread_role
from train.style_screens import normalize


def line(text, box, confidence=1.0):
    return {'text': text, 'box': box, 'confidence': confidence}


def request(thread, name='Lobster', url='https://www.dafont.com/lobster.font', file=None):
    return {'id': thread, 'identified': [{'name': name, 'url': url, 'by': 'x'}], 'files': [{'file': file or f'{thread}-0.png', 'width': 400, 'height': 200}]}


class ForumViewTests(unittest.TestCase):
    def test_a_thread_is_held_out_by_the_hash_of_its_id_one_in_ten(self):
        self.assertEqual([thread_role(t) for t in [603084, 603080, 464694, 384803]], ['development', 'development', 'held-out', 'development'])
        held = sum(thread_role(t) == 'held-out' for t in range(400000, 402000)); self.assertTrue(150 < held < 250, held)

    def test_the_answer_s_style_words_give_the_face(self):
        self.assertEqual([style_of(n) for n in ['Bebas Neue', 'Bebas Neue Bold', 'Lato Light Italic', 'Open Sans ExtraBold', 'Oswald Semibold', 'Roboto 700']],
                         [(400, False), (700, False), (300, True), (800, False), (600, False), (400, False)])

    def test_the_line_is_the_one_the_answerer_previewed_else_the_tallest_confident_horizontal_one(self):
        lines = [line('ADMITS ONE', [43, 42, 58, 126]), line('GIRLS', [96, 74, 255, 139]), line('LOVESICK', [88, 29, 251, 72]), line('Honey', [0, 0, 300, 200], .4), line('', [0, 0, 300, 100])]
        self.assertEqual(line_of(lines, 'Lovesick Girls')['text'], 'GIRLS')  # both share a word; the first of the two, which Vision lists largest first
        self.assertEqual(line_of(lines, 'lovesick+')['text'], 'LOVESICK')
        self.assertEqual(line_of(lines, '')['text'], 'GIRLS')  # no preview text: the tallest line that is confident, horizontal and lettered
        self.assertIsNone(line_of([line('x', [0, 0, 10, 40]), line('y', [0, 0, 40, 10], .2), line('z', [5, 5, 5, 5])]))  # upright, unsure, no size: none is a line

    def test_a_development_thread_with_a_free_font_answer_in_a_training_family_labels_its_line(self):
        families = {normalize('Lobster'): 'lobster'}
        boxes = {f'{t}-0.png': {'lines': [line('Fresh bread', [10, 10, 200, 60])]} for t in [603084, 464694, 603080, 603079, 603078]}
        boxes['603078-0.png'] = {'lines': None}
        picked = chosen([request(603084), request(464694), request(603080, url='https://www.myfonts.com/fonts/lobster'), request(603079, name='Gotham'),
                         request(603078), request(603077), request(603076, name='Lobster Bold')], boxes, families, bench={603076})
        self.assertEqual([(r['id'], l['text'], *rest) for r, l, *rest in picked], [(603084, 'Fresh bread', 'lobster', 'Latn', 400, False, 'www.dafont.com')])
        # Held-out thread, commercial page, unknown family, picture without lines, picture never boxed, benchmark thread: none of them a view.

    def test_the_crop_keeps_a_little_background_around_the_line_within_the_picture(self):
        from PIL import Image
        image = Image.new('L', (400, 200)); self.assertEqual(crop(image, [100, 50, 300, 90]).size, (220, 60)); self.assertEqual(crop(image, [0, 0, 400, 200]).size, (400, 200))


if __name__ == '__main__': unittest.main()
