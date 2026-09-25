import base64
import json
import random
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from scripts.corpus import ROOT
from train.style_data import Preparer, text, tensors, skeleton, sketch, COMMON
from train.style_teacher import distances, twins, enumerate_faces
from train.style_catalog import kinds, LATIN
from train.style import Heads, Setup, losses, architecture_of, benchmark_of, run_seed, train, CATEGORIES
from train.ten_model import Classifier, CORPUS_ARCH, LARGE_ARCH, WIDER_ARCH
from train.style_open import queryable


def line(width=90, height=30, seed=0):
    rng = np.random.default_rng(seed); gray = np.full((height, width), 255, np.uint8)
    for x in range(8, width - 8, 7): gray[6:height - 6, x:x + 3] = rng.integers(0, 60)
    return gray


class StyleDataTests(unittest.TestCase):
    def test_persistent_preparation_matches_batch_preparation_byte_for_byte(self):
        images = [line(), line(140, 44, 1), line(20, 12, 2), np.full((9, 9), 255, np.uint8)]
        batch = json.loads(subprocess.run(['node', str(ROOT/'scripts/corpus-prepare.mjs')], input=json.dumps(
            [{'width': g.shape[1], 'height': g.shape[0], 'pixels': base64.b64encode(g.tobytes()).decode()} for g in images]), text=True, capture_output=True, check=True).stdout)
        prepare = Preparer()
        try:
            for _ in range(2):  # repeated use of one child
                for gray, expected in zip(images, batch):
                    ours = prepare(gray)
                    self.assertEqual([(w.shape[1], w.shape[0], w.tobytes()) for w in ours], [(w['width'], w['height'], base64.b64decode(w['pixels'])) for w in expected])
            self.assertEqual(prepare(np.full((9, 9), 255, np.uint8)), [])  # blank: no windows, child alive
            self.assertRaises(ValueError, prepare, np.zeros((0, 5), np.uint8))
            self.assertTrue(prepare(line()))
        finally: prepare.close()

    def test_text_uses_only_the_face_characters_in_every_case(self):
        pool = COMMON['Latn'][:26] + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789'
        rng = random.Random(4); seen = set()
        for _ in range(400):
            value = text(rng, 'Latn', pool)
            self.assertTrue(1 <= len(value) <= 24); self.assertTrue(set(value) <= set(pool) | {' '})
            seen.add('upper' if value.isupper() else 'lower' if value.islower() else 'mixed')
        self.assertEqual(seen, {'upper', 'lower', 'mixed'})
        self.assertTrue(set(text(random.Random(1), 'Hani', '人体女字小歌谷里')) <= set('人体女字小歌谷里'))
        # A planned case holds for every draw; a script without case ignores it.
        for case, holds in [('upper', lambda v: v == v.upper()), ('lower', lambda v: v == v.lower()), ('title', lambda v: v[1:] == v[1:].lower() and v[:1] == v[:1].upper())]:
            self.assertTrue(all(holds(text(rng, 'Latn', pool, case)) for _ in range(200)), case)
        self.assertTrue(set(text(random.Random(1), 'Hani', '人体女字小歌谷里', 'upper')) <= set('人体女字小歌谷里'))

    def test_paired_views_cross_case_and_unpaired_views_draw_their_own(self):
        pools = {'a': {'Latn': 'abc'}, 'b': {'Latn': 'abc', 'Cyrl': 'абв'}}
        fake = SimpleNamespace(by_family={'a': [0], 'b': [1]}, neighbours=np.array([[1], [0]]), train=[5, 6], pools=pools, faces={5: {'family': 'a'}, 6: {'family': 'b'}})
        paired = Setup.plan(fake, 1, faces=8, pairs=True)(); plain = Setup.plan(fake, 1, faces=8)()
        self.assertEqual(len(paired), 16); self.assertTrue(all(len(spec) == 5 for spec in paired + plain))
        for first, second in zip(paired[::2], paired[1::2]):
            self.assertEqual(first[0], second[0]); self.assertEqual({'lower'} & {first[4], second[4]}, {'lower'}); self.assertTrue({first[4], second[4]} - {'lower'} <= {'upper', 'title'})
        self.assertEqual({spec[4] for spec in plain}, {None})
        full = Setup.plan(fake, 1, pairs=True)()  # a full batch: a two-script face also pairs across scripts
        self.assertTrue(any(a[1] != b[1] for a, b in zip(full[::2], full[1::2]) if a[0] == 6))

    def test_fixed_frame_tensors_drop_filler_rows(self):
        views = [{'windows': [np.zeros((10, 20), np.uint8), np.zeros((48, 128), np.uint8)]}, {'windows': [np.zeros((5, 7), np.uint8)]}]
        pixels, sizes, owners = tensors(views, 'cpu', bucket=4)
        self.assertEqual(tuple(pixels.shape), (4, 1, 48, 128)); self.assertEqual(owners.tolist(), [0, 0, 1, -1])
        self.assertEqual(sizes.tolist()[:3], [[10, 20], [48, 128], [5, 7]]); self.assertEqual(float(pixels[0, 0, 10:, :].min()), 1.0)


class SketchTests(unittest.TestCase):
    def test_thinning_keeps_every_stroke_as_a_one_pixel_line(self):
        mask = np.zeros((40, 60), bool); mask[5:12, 5:55] = True; mask[20:35, 10:17] = True  # a bar and a stem
        thin = skeleton(mask)
        def parts(m):
            seen = np.zeros_like(m); count = 0
            for y, x in zip(*np.nonzero(m)):
                if seen[y, x]: continue
                count += 1; stack = [(y, x)]
                while stack:
                    a, b = stack.pop()
                    if 0 <= a < m.shape[0] and 0 <= b < m.shape[1] and m[a, b] and not seen[a, b]:
                        seen[a, b] = True; stack += [(a + i, b + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            return count
        self.assertEqual(parts(thin), 2); self.assertTrue((thin <= mask).all())
        self.assertLessEqual(thin[5:12, 30].sum(), 2); self.assertGreater(thin[5:12].sum(), 30)  # thin across, long along
        from PIL import Image
        drawn = sketch(Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)), random.Random(1))
        self.assertLess(np.asarray(drawn).min(), 64); self.assertEqual(np.asarray(drawn).max(), 255)


class TeacherTests(unittest.TestCase):
    def test_identical_glyphs_are_zero_apart_and_sparse_overlap_is_unknown(self):
        a = np.zeros((3, 48, 48), np.uint8); a[:, 8:40, 20:26] = 255  # vertical stems
        b = np.zeros((3, 48, 48), np.uint8); b[:, 20:26, 8:40] = 255  # horizontal bars
        glyphs = np.stack([a, a, b, a]); have = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0]], bool)
        d = distances(glyphs, have, device='cpu')
        self.assertAlmostEqual(float(d[0, 1]), 0, places=5); self.assertGreater(float(d[0, 2]), .1)
        self.assertTrue(np.isinf(d[0, 3])); np.testing.assert_allclose(d, d.T, atol=1e-5); self.assertTrue((np.diag(d) == 0).all())

    def test_twins_are_pairwise_not_chained(self):
        near = twins(np.array([[0, .005, .02], [.005, 0, .005], [.02, .005, 0]], np.float32), .008)
        self.assertTrue(near[0, 1] and near[1, 2]); self.assertFalse(near[0, 2]); self.assertFalse(near.diagonal().any())

    def test_variable_faces_follow_their_weight_axis_and_one_default_per_family(self):
        inventory = {'families': [
            {'id': 'v', 'family': 'V', 'excluded': False, 'selected': 'v.ttf', 'trainingAxes': {'wght': 400.0}, 'alphabets': {'Latn': 'ab'},
             'faces': [{'path': 'v.ttf', 'blob': 'x', 'weight': 400, 'italic': False, 'axes': {'wght': {'min': 250, 'default': 400, 'max': 700}}},
                       {'path': 'vi.ttf', 'blob': 'y', 'weight': 400, 'italic': True, 'axes': {'wght': {'min': 250, 'default': 400, 'max': 700}}}]},
            {'id': 's', 'family': 'S', 'excluded': False, 'selected': 's-b.ttf', 'trainingAxes': {}, 'alphabets': {'Latn': 'ab'},
             'faces': [{'path': 's-b.ttf', 'blob': 'z', 'weight': 700, 'italic': False, 'axes': {}}]}]}
        faces = enumerate_faces(inventory)
        self.assertEqual([f['id'] for f in faces if f['family'] == 'v'], ['v/300', 'v/400', 'v/500', 'v/600', 'v/700', 'v/300i', 'v/400i', 'v/500i', 'v/600i', 'v/700i'])
        self.assertEqual([f['id'] for f in faces if f['default']], ['v/400', 's/700'])


class CatalogAndLossTests(unittest.TestCase):
    def test_reference_kinds_cover_both_latin_cases_and_other_scripts(self):
        latin = ''.join(sorted(set(''.join(LATIN['lower'] + LATIN['upper']))))
        self.assertEqual(sorted(kinds({'Latn': latin, 'Cyrl': 'абвгдежзийклмноп'})), ['Cyrl-lower', 'Latn-lower', 'Latn-upper'])
        self.assertEqual(sorted(kinds({'Latn': latin.lower()})), ['Latn-lower'])
        self.assertEqual(kinds({'Grek': 'αβγ'}), {})
        # A cased script gets both cases, each alphabet whole across three lines; digits are not letters.
        cyrillic = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789'
        lines = kinds({'Cyrl': cyrillic})
        self.assertEqual(sorted(lines), ['Cyrl-lower', 'Cyrl-upper']); self.assertEqual([len(l) for l in lines['Cyrl-upper']], [11, 11, 11])
        self.assertEqual(''.join(lines['Cyrl-lower']), cyrillic[:33]); self.assertEqual(''.join(lines['Cyrl-upper']), cyrillic[33:66])
        # Other scripts: four lines of the 32 most shared characters, or fewer when the face has fewer.
        han = ''.join(chr(0x4e00 + i) for i in range(40))
        self.assertEqual(kinds({'Hani': han})['Hani'], [han[i:i + 8] for i in range(0, 32, 8)]); self.assertEqual(kinds({'Hani': han[:12]})['Hani'], [han[:8], han[8:12]])

    def test_twins_share_the_identity_target_and_geometry_follows_the_teacher(self):
        n = 3; distance = torch.tensor([[0, .001, .3], [.001, 0, .3], [.3, .3, 0]])
        near = (distance < .005).float(); setup = SimpleNamespace(position={10: 0, 11: 1, 12: 2}, twins=near/near.sum(1, keepdim=True), distance=distance,
            faces={10: {'weight': 400, 'italic': False, 'family': 'a'}, 11: {'weight': 400, 'italic': False, 'family': 'b'}, 12: {'weight': 700, 'italic': True, 'family': 'c'}},
            scripts=['Latn'], category={'a': 0}, fine_labels={'a': np.array([1.], np.float32)})
        heads = Heads(['Latn'], ['/Sans/Geometric']); proxies = torch.eye(3, 128)
        views = [{'face': 10, 'script': 'Latn'}, {'face': 12, 'script': 'Latn'}]
        on_twin = losses(torch.eye(128)[[1, 2]], heads, proxies, setup, views, 'cpu')[1]['identity']
        on_self = losses(torch.eye(128)[[0, 2]], heads, proxies, setup, views, 'cpu')[1]['identity']
        self.assertAlmostEqual(float(on_twin), float(on_self), places=5)  # a twin's proxy is as correct as its own
        far = losses(torch.eye(128)[[2, 0]], heads, proxies, setup, views, 'cpu')[1]
        self.assertGreater(float(far['geometry']), float(losses(torch.eye(128)[[0, 2]], heads, proxies, setup, views, 'cpu')[1]['geometry']))
        self.assertEqual(len(CATEGORIES), heads.category.out_features)

    def test_each_run_draws_its_own_batches_unless_a_seed_is_pinned(self):
        self.assertEqual(run_seed('wider'), run_seed('wider')); self.assertNotEqual(run_seed('wider'), run_seed('plain'))
        self.assertEqual(run_seed('any', 20260923), 20260923)
        fake = SimpleNamespace(by_family={'a': [0], 'b': [1]}, neighbours=np.array([[1], [0]]), train=[5, 6], pools={'a': {'Latn': 'abc'}, 'b': {'Latn': 'abc'}}, faces={5: {'family': 'a'}, 6: {'family': 'b'}})
        self.assertNotEqual(Setup.plan(fake, run_seed('wider'))(), Setup.plan(fake, run_seed('plain'))())

    def test_a_run_never_selects_on_families_it_trains(self):
        # Refused before any setup: training development families and selecting on them would grade the model on its own homework.
        with self.assertRaisesRegex(ValueError, 'catalog:validation'): train('x', ['train', 'development', 'test'], 1, 'unused.pt', 1)

    def test_the_rejection_threshold_separates_present_from_absent_scores(self):
        from train.style import threshold_of
        # Present scores all above absent ones: the cut sits at the lowest present score and keeps every present query.
        rule = threshold_of([.9, .8, .7], [.6, .5]); self.assertEqual((rule['threshold'], rule['presentCoverage'], rule['absentAcceptance']), (.7, 1.0, 0.0))
        # Overlap: the cut that gains most present coverage per absent acceptance.
        rule = threshold_of([.9, .85, .8, .7, .6], [.5, .55, .6, .65, .4]); self.assertAlmostEqual(rule['threshold'], .7, places=6); self.assertEqual(rule['presentCoverage'], .8)
        # Identical distributions: no cut helps; the rule keeps everything (J = 0 at the lowest score).
        rule = threshold_of([.5, .6], [.5, .6]); self.assertEqual(rule['presentCoverage'], 1.0); self.assertEqual(rule['absentAcceptance'], 1.0)

    def test_a_run_is_measured_where_its_families_are_held_out(self):
        # The frozen benchmark holds out development and test families; a run that trained on either reads the catalog test.
        self.assertEqual(benchmark_of(['train']), 'bench')
        for roles in (['train', 'development', 'test'], ['train', 'test'], ['train', 'development']): self.assertEqual(benchmark_of(roles), 'catalog')

    def test_an_export_names_the_preparation_files_that_run(self):
        from train.style import shipped_preparation
        from train.robustness import sha
        block = shipped_preparation(); shipped = json.load(open(ROOT/'models/encoder/encoder.json'))['preparation']
        self.assertEqual({k: v for k, v in block.items() if k not in ('sha256', 'normalizerSha256', 'lineSha256')}, {k: v for k, v in shipped.items() if k not in ('sha256', 'normalizerSha256', 'lineSha256')})
        self.assertEqual([block['sha256'], block['normalizerSha256'], block['lineSha256']], [sha(ROOT/f'src/{name}.mjs') for name in ['input', 'prepare', 'line']])

    def test_a_folded_head_keeps_the_projected_direction_and_the_checkpoint_its_width(self):
        from train.style import fold_head, dimensions_of
        torch.manual_seed(2); model = Classifier(128, context=True, wide=True, dilations=[1, 1, 2, 2, 1]).eval()
        basis, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(128, 64))); basis = basis.T.astype(np.float32)  # 64 orthonormal rows
        folded = Classifier(64, context=True, wide=True, dilations=[1, 1, 2, 2, 1]).eval(); folded.load_state_dict(fold_head(model.state_dict(), basis))
        x = torch.rand(3, 1, 48, 128)
        with torch.no_grad(): e, f = model(x), folded(x)
        expected = torch.nn.functional.normalize(torch.nn.functional.normalize(e, dim=1) @ torch.from_numpy(basis).T, dim=1)
        self.assertGreater(float((torch.nn.functional.normalize(f, dim=1) * expected).sum(1).min()), .9999)
        self.assertEqual((dimensions_of({'state': model.state_dict()}), dimensions_of({'state': folded.state_dict()})), (128, 64))


        self.assertEqual(architecture_of({'architecture': WIDER_ARCH, 'large': True}), WIDER_ARCH)
        self.assertEqual(architecture_of({'large': True}), LARGE_ARCH); self.assertEqual(architecture_of({}), CORPUS_ARCH)

    def test_view_loss_pulls_a_face_s_views_together_and_is_off_by_default(self):
        near = torch.eye(3); setup = SimpleNamespace(position={10: 0, 11: 1, 12: 2}, twins=near, distance=torch.full((3, 3), .3).fill_diagonal_(0),
            faces={k: {'weight': 400, 'italic': False, 'family': k} for k in (10, 11, 12)}, scripts=['Latn'], category={}, fine_labels={})
        heads = Heads(['Latn'], ['/Sans/Geometric']); proxies = torch.eye(3, 128); views = [{'face': f, 'script': 'Latn'} for f in (10, 10, 12, 12)]
        together = losses(torch.eye(128)[[0, 0, 2, 2]], heads, proxies, setup, views, 'cpu', 1.0)[1]['views']
        apart = losses(torch.eye(128)[[0, 2, 2, 0]], heads, proxies, setup, views, 'cpu', 1.0)[1]['views']  # each view sits on the other face
        self.assertLess(float(together), .01); self.assertGreater(float(apart), 10)
        alone = [{'face': f, 'script': 'Latn'} for f in (10, 11, 12)]  # no face twice: nothing to pull together
        self.assertEqual(float(losses(torch.eye(128)[[0, 1, 2]], heads, proxies, setup, alone, 'cpu', 1.0)[1]['views']), 0)
        drawn = [{'face': 10, 'script': 'Latn', 'drawn': True}, {'face': 10, 'script': 'Latn'}, {'face': 12, 'script': 'Latn'}]
        self.assertEqual(float(losses(torch.eye(128)[[2, 0, 2]], heads, proxies, setup, drawn, 'cpu', 1.0)[1]['views']), 0)  # a drawing is never a positive
        total, parts, _ = losses(torch.eye(128)[[0, 2, 2, 0]], heads, proxies, setup, views, 'cpu')
        self.assertEqual(float(parts['views']), 0)
        self.assertAlmostEqual(float(total), float(parts['identity'] + parts['geometry'] + .2*sum(parts[k] for k in ['weight', 'italic', 'script', 'category', 'fine'])), places=4)
        # A teacher's target: nothing to learn when the views already sit on it, a full cosine distance when orthogonal.
        from train.style import DISTILL
        e = torch.eye(128)[[0, 2, 2, 0]]
        self.assertEqual(float(losses(e, heads, proxies, setup, views, 'cpu', follow=e)[1]['distill']), 0)
        away, parts_away, _ = losses(e, heads, proxies, setup, views, 'cpu', follow=torch.eye(128)[[5, 5, 5, 5]])
        self.assertAlmostEqual(float(parts_away['distill']), 1, places=6); self.assertAlmostEqual(float(away - total), DISTILL, places=4)


if __name__ == '__main__': unittest.main()


class PhotoViewTests(unittest.TestCase):
    def test_photo_views_keep_the_letters_findable_and_differ_by_seed(self):
        from train.style_data import damage, effect
        image = Image.fromarray(line(200, 40, 3)); prepare = Preparer()
        try:
            outs = [damage(image, random.Random(seed), photo=True) for seed in range(12)]
            self.assertTrue(all(o.dtype == np.uint8 and o.ndim == 2 for o in outs))
            self.assertEqual(len({o.tobytes() for o in outs}), 12)
            self.assertEqual(damage(image, random.Random(5), photo=True).tobytes(), outs[5].tobytes())  # one seed, one view
            self.assertGreaterEqual(sum(bool(prepare(o)) for o in outs), 10)  # the surface never hides the line
            self.assertEqual(damage(image, random.Random(5)).tobytes(), damage(image, random.Random(5), photo=False).tobytes())  # plain damage unchanged by the flag
        finally: prepare.close()
        # An outline keeps the letters' interior as surface; a shadow or an extrusion keeps the letters and adds ink behind them.
        mask = np.zeros((30, 60), np.float32); mask[8:22, 10:50] = 1
        outline = effect(mask, random.Random(0), 30, 'outline')
        self.assertEqual(float(outline[15, 30]), 0.0); self.assertGreater(float(outline[7, 30]), 0)
        for kind in ('shadow', 'extrusion'):
            out = effect(mask, random.Random(0), 30, kind)
            self.assertTrue((out >= mask - 1e-6).all()); self.assertGreater(float(out.sum()), float(mask.sum()))

    def test_photo_benchmark_reads_weight_and_posture_from_the_set_s_titles(self):
        from train.style_photos import weight_of, normalize, family_of
        self.assertEqual([weight_of(t) for t in ['Asap 600', 'Lora Bold Italic', 'Alef', 'Roboto Thin', 'Playfair Display 900 Italic']], [(600, False), (700, True), (400, False), (100, False), (900, True)])
        self.assertEqual(normalize('IBM Plex Sans KR'), 'ibmplexsanskr')
        # The set names fonts with a style word or weight after the family; one family keeps a number that is its name.
        self.assertEqual([family_of(n) for n in ['Heebo regular', 'Spartan 500', 'Inter Tight Regular', 'Roboto Medium 500', 'Asap', 'Press Start 2P']], ['Heebo', 'Spartan', 'Inter Tight', 'Roboto', 'Asap', 'Press Start 2P'])


class StyleOpenTests(unittest.TestCase):
    def test_open_plan_queries_only_faces_the_text_recipe_can_draw(self):
        self.assertTrue(queryable('abcdefghijklmnopqrstuvwxyz'))
        self.assertTrue(queryable('etaoinshr'))
        self.assertFalse(queryable('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        self.assertFalse(queryable('eta'))
        self.assertFalse(queryable(''))

