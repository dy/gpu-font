import base64
import json
import random
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts.corpus import ROOT
from train.style_data import Preparer, text, tensors, skeleton, sketch, COMMON
from train.style_teacher import distances, twins, enumerate_faces
from train.style_catalog import kinds, LATIN
from train.style import Heads, Setup, losses, nearest, twin_rows, architecture_of, benchmark_of, run_seed, train, CATEGORIES, StoredViews, adapt_heads
from train.style_data import damage, init as init_stream, sample as sample_view, STATE, Stream
from train.encoder_data import save
from train.ten_model import Classifier, CORPUS_ARCH, LARGE_ARCH, WIDER_ARCH
from train.style_open import queryable
from train.style_fonts import open_role, open_families, HELD_OUT
from train.style_teacher import enumerate_faces, render_job, SIZE
from scripts.corpus import font_path, CACHE, ROOT as REPO


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
        faces = enumerate_faces(inventory['families'])
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
        setup = SimpleNamespace(position={10: 0, 11: 1, 12: 2}, floor=.005, distance=distance.half(),  # as the setup holds it: 16-bit, twins read off the rows
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
        setup = SimpleNamespace(position={10: 0, 11: 1, 12: 2}, floor=.005, distance=torch.full((3, 3), .3).fill_diagonal_(0).half(),
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


class HeadExportTests(unittest.TestCase):
    def test_a_head_is_written_at_four_decimals(self):
        import json
        from train.style import head_layer, HEAD_DECIMALS
        rng = np.random.default_rng(3); weight = rng.normal(size=(5, 64)).astype(np.float32); bias = rng.normal(size=5).astype(np.float32)
        layer = head_layer(weight, bias)
        digits = [len(n.split('.')[1]) for n in json.dumps(layer).replace('[', ' ').replace(']', ' ').replace(',', ' ').split() if '.' in n and n[0] in '-0123456789']
        self.assertEqual(len(digits), weight.size + bias.size); self.assertLessEqual(max(digits), HEAD_DECIMALS)
        self.assertLess(np.abs(np.array(layer['weights']) - weight).max(), .5*10**-HEAD_DECIMALS + 1e-7)
        self.assertLess(np.linalg.norm(np.array(layer['weights']) - weight, axis=1).max(), 4e-4)  # a logit's largest move on a unit embedding


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


class StyleFontsTests(unittest.TestCase):
    def test_open_faces_carry_their_store_and_google_faces_do_not(self):
        google = {'id': 'g', 'family': 'G', 'faces': [{'path': 'ofl/g/G.ttf', 'blob': 'b', 'weight': 400, 'italic': False, 'axes': {}}], 'selected': 'ofl/g/G.ttf', 'trainingAxes': {}, 'alphabets': {'Latn': 'abc'}, 'excluded': None}
        open_ = {**google, 'id': 'o', 'family': 'O', 'store': '.data/fonts-open'}
        faces = enumerate_faces([google, open_])
        self.assertEqual([('store' in f, str(font_path(f))) for f in faces], [(False, str(CACHE/'ofl/g/G.ttf')), (True, str(REPO/'.data/fonts-open/ofl/g/G.ttf'))])

    def test_one_open_family_in_ten_is_held_out_by_its_id_alone(self):
        ids = [f'family-{i}' for i in range(5000)]; held = [i for i in ids if open_role(i) == 'held-out']
        self.assertEqual({open_role(i) for i in ids}, {'train', 'held-out'})  # never a role a training run names, such as test
        self.assertTrue(0.08 < len(held)/len(ids) < 0.12, len(held))
        self.assertEqual([open_role(i) for i in ids[:50]], [open_role(i) for i in ids[:50]])  # the rule, not a draw
        self.assertEqual(open_role('debian-dejavu'), open_role('debian-dejavu'))

    def test_open_families_need_letters_to_draw_and_keep_their_alphabets(self):
        inventory = {'store': '.data/fonts-open', 'families': [
            {'id': 'a', 'family': 'A', 'faces': [], 'excluded': None}, {'id': 'b', 'family': 'B', 'faces': [], 'excluded': 'colour'},
            {'id': 'c', 'family': 'C', 'faces': [], 'excluded': None, 'symbolEncoded': True}, {'id': 'd', 'family': 'D', 'faces': [], 'excluded': None}]}
        alphabets = {'a': {'Latn': 'abcdefghijklmnopqrstuvwxyz', 'Grek': 'αβγ'}, 'b': {'Latn': 'abcdefghijklmnopqrstuvwxyz'}, 'c': {'Latn': 'abcdefghijklmnopqrstuvwxyz'}, 'd': {'Latn': 'abc'}}
        families = open_families(inventory, alphabets)
        self.assertEqual([(f['id'], f['store'], sorted(f['alphabets'])) for f in families], [('a', '.data/fonts-open', ['Latn'])])


class StoredViewTests(unittest.TestCase):
    def test_flat_renders_sometimes_wear_a_display_effect(self):
        image = Image.fromarray(line(120, 40)); base = np.asarray(image, np.float32)
        base_ink = float((np.abs(base - 255) > 60).mean()); heavier = 0
        for seed in range(80):
            a = damage(image, random.Random(seed)).astype(np.float32); paper = np.median(a)
            heavier += float((np.abs(a - paper) > 60).mean()) > 1.3*base_ink
        self.assertTrue(4 <= heavier <= 40, heavier)  # an outline, shadow or extrusion adds ink to about one render in seven

    def test_photographs_are_stored_views_in_the_face_of_their_weight_and_every_image_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            pixels = Path(tmp)/'p.u8'; pixels.write_bytes(bytes(range(24)))
            manifest = Path(tmp)/'m.json'
            save(manifest, {'samples': [{'family': 'a', 'weight': 700, 'italic': False, 'script': 'Latn', 'role': 'development'},
                                        {'family': 'a', 'weight': 100, 'italic': True, 'script': 'Latn', 'role': 'development'},
                                        {'family': 'b', 'weight': 400, 'italic': False, 'script': 'Latn', 'role': 'development'}],
                            'windows': [{'source': 0, 'offset': 0, 'width': 4, 'height': 3}, {'source': 1, 'offset': 12, 'width': 4, 'height': 3}, {'source': 2, 'offset': 0, 'width': 4, 'height': 3}]})
            faces = [{'family': 'a', 'weight': 400, 'italic': False, 'default': True}, {'family': 'a', 'weight': 700, 'italic': False, 'default': False}]
            setup = SimpleNamespace(faces=faces, position={0: 0, 1: 1}, scripts=['Latn'])
            stored = StoredViews(setup, ('train',), sources=[(str(manifest), str(pixels), 'photo')])
            # The 700 photograph takes the 700 face; the 100 italic one, which the family lacks, its default face; family b is not trained.
            self.assertEqual([(face, script) for _, face, script, _ in stored.photos], [(1, 'Latn'), (0, 'Latn')])
            view = stored.sample(random.Random(0), 1)[0]; self.assertEqual(view['windows'][0].shape, (3, 4))
            self.assertEqual((len(stored.photos), len(stored.views)), (2, 0))  # photographs keep their own list, for their share of a step
            chromium = StoredViews(setup, ('train',), sources=[(str(manifest), str(pixels), 'chromium')])
            self.assertEqual(chromium.views, [])  # a Chromium source keeps to the roles asked for; these are development

    def test_a_damaged_font_file_draws_nothing_instead_of_stopping_the_teacher_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'broken.ttf').write_bytes(b'\x00\x01\x00\x00' + bytes(range(256))*4)
            face = {'id': 'x/400', 'family': 'x', 'weight': 400, 'italic': False, 'path': 'broken.ttf', 'axes': {}, 'scripts': ['Latn'], 'store': tmp}
            glyphs, have = render_job((face, 'Latn', 'abc'))
            self.assertEqual((glyphs.shape, have.tolist()), ((3, SIZE, SIZE), [False, False, False]))


class StreamTests(unittest.TestCase):
    def test_a_damaged_font_yields_no_view_and_is_named_once_instead_of_failing_the_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'broken.ttf').write_bytes(b'\x00\x01\x00\x00' + bytes(range(256))*4)
            face = {'id': 'x/400', 'family': 'x', 'weight': 400, 'italic': False, 'path': 'broken.ttf', 'axes': {}, 'scripts': ['Latn'], 'default': True, 'store': tmp}
            init_stream([face], {'x': {'Latn': 'abcdefghijklmnopqrstuvwxyz'}})
            self.assertIsNone(sample_view((0, 'Latn', 1, True, None))); self.assertIn('x/400', STATE['failed'])


class WarmStartTests(unittest.TestCase):
    def test_a_checkpoint_s_heads_follow_their_labels_into_a_setup_with_more_scripts(self):
        torch.manual_seed(1); old = Heads(['Cyrl', 'Latn'], ['a', 'b'], 8); state = {'scripts': ['Cyrl', 'Latn'], 'fine': ['a', 'b'], 'heads': old.state_dict()}
        new = Heads(['Grek', 'Latn', 'Cyrl'], ['b', 'c'], 8); fresh = {k: v.clone() for k, v in new.state_dict().items()}
        new.load_state_dict(adapt_heads(state, ['Grek', 'Latn', 'Cyrl'], ['b', 'c'], new.state_dict())); loaded = new.state_dict()
        torch.testing.assert_close(loaded['script.weight'][1], old.script.weight[1]); torch.testing.assert_close(loaded['script.weight'][2], old.script.weight[0])
        torch.testing.assert_close(loaded['script.bias'][0], fresh['script.bias'][0])  # Grek is new: its row stays as initialized
        torch.testing.assert_close(loaded['fine.weight'][0], old.fine.weight[1]); torch.testing.assert_close(loaded['fine.weight'][1], fresh['fine.weight'][1])
        for key in ('weight.weight', 'italic.bias', 'category.weight'): torch.testing.assert_close(loaded[key], old.state_dict()[key])

    def test_photographs_take_a_fixed_share_of_each_step_s_stored_views(self):
        stored = StoredViews.__new__(StoredViews); stored.pixels = [np.arange(12, dtype=np.uint8)]
        stored.views = [(0, 0, 'Latn', [(0, 4, 3)])]*100; stored.photos = [(0, 1, 'Latn', [(0, 4, 3)])]*10
        faces = [v['face'] for v in stored.sample(random.Random(0), 32)]
        self.assertEqual((faces.count(1), faces.count(0)), (4, 28))
        stored.photos = []; self.assertEqual([v['face'] for v in stored.sample(random.Random(0), 8)], [0]*8)  # no photographs: renders only
        stored.photos = [(0, 1, 'Latn', [(0, 4, 3)])]*3; stored.views = []; self.assertEqual([v['face'] for v in stored.sample(random.Random(0), 8)], [1]*3)  # no renders: the photographs there are

    def test_heads_keep_the_labels_behind_their_outputs(self):
        heads = Heads(['Latn', 'Cyrl'], ['a'], 8); self.assertEqual((heads.script_labels, heads.fine_labels), (['Latn', 'Cyrl'], ['a']))
        self.assertEqual(heads.script.out_features, 2); self.assertEqual(heads.fine.out_features, 1)


class MemoryTests(unittest.TestCase):
    def test_distances_in_row_blocks_from_a_file_equal_the_whole_at_once(self):
        rng = np.random.default_rng(5); glyphs = rng.integers(0, 256, (9, 4, 48, 48), dtype=np.uint8); have = rng.random((9, 4)) < .8
        whole = distances(glyphs, have, device='cpu', block=64)
        with tempfile.TemporaryDirectory() as tmp:
            stored = np.lib.format.open_memmap(Path(tmp)/'g.npy', mode='w+', dtype=np.uint8, shape=glyphs.shape); stored[:] = glyphs
            # Single-precision sums in another order: 1e-6 of noise beside a twin threshold of 8e-3.
            np.testing.assert_allclose(distances(stored, have, device='cpu', block=2), whole, rtol=0, atol=2e-6)
            rows = np.array([7, 0, 3, 4])  # only the faces kept, in the order given
            np.testing.assert_allclose(distances(stored, have, device='cpu', rows=rows, block=3), distances(glyphs[rows], have[rows], device='cpu'), rtol=0, atol=2e-6)
        self.assertEqual((whole.dtype, whole.shape, float(np.diag(whole).max())), (np.float32, (9, 9), 0.0))
        one = distances(glyphs[:1], have[:1], device='cpu'); self.assertEqual(one.tolist(), [[0.0]])  # a single face

    def test_nearest_and_twins_come_from_blocks_of_rows_of_a_16_bit_matrix(self):
        d = torch.tensor([[0, .001, .25, .2], [.001, 0, .3, float('inf')], [.25, .3, 0, .1], [.2, float('inf'), .1, 0]]).half()  # no ties
        neighbours, twinned = nearest(d, .005, count=2, block=3)
        self.assertEqual((neighbours.tolist(), twinned), ([[1, 3], [0, 2], [3, 0], [2, 0]], 2))
        setup = SimpleNamespace(distance=d, floor=.005); rows, near = twin_rows(setup, torch.tensor([1, 3]))
        self.assertEqual((rows.dtype, near.tolist()), (torch.float32, [[True, True, False, False], [False, False, False, True]]))


class StreamRecoveryTests(unittest.TestCase):
    def test_a_batch_that_never_returns_rebuilds_the_pool_and_the_stream_goes_on(self):
        import multiprocessing, queue as queues
        class Lost:
            def get(self, timeout): raise multiprocessing.TimeoutError
        class Done:
            def __init__(self, views): self.views = views
            def get(self, timeout): return self.views
        class Pool:
            built = []
            def __init__(self, dead): self.dead = dead; self.terminated = False; Pool.built.append(self)
            def map_async(self, function, specs, chunksize): return Lost() if self.dead else Done([{'face': specs[0]}, None])
            def terminate(self): self.terminated = True
        stream = Stream.__new__(Stream); stream.PATIENCE = 0; stream.plan = lambda: [7]; stream.queue = queues.Queue(2); stream.stop = False; stream.error = None
        stream.workers = lambda: Pool(dead=False); stream.pool = Pool(dead=True)
        import threading; thread = threading.Thread(target=stream.fill, daemon=True); thread.start()
        views = stream.queue.get(timeout=5); stream.stop = True
        self.assertEqual(views, [{'face': 7}])  # the lost batch is dropped, the next ones arrive, unrenderable views left out
        self.assertEqual([(p.dead, p.terminated) for p in Pool.built], [(True, True), (False, False)]); self.assertIsNone(stream.error)

    def test_a_batch_taken_in_shares_has_the_whole_batch_s_loss(self):
        # What taking a step in parts relies on: every loss is a mean over views, so shares weighted by their size add up.
        torch.manual_seed(3); distance = torch.tensor([[0, .001, .3], [.001, 0, .3], [.3, .3, 0]])
        setup = SimpleNamespace(position={10: 0, 11: 1, 12: 2}, floor=.005, distance=distance.half(),
            faces={k: {'weight': w, 'italic': i, 'family': k} for k, w, i in [(10, 400, False), (11, 700, False), (12, 300, True)]}, scripts=['Latn'], category={10: 0, 11: 1, 12: 2}, fine_labels={})
        heads = Heads(['Latn'], ['/Sans/Geometric'], 16); proxies = torch.randn(3, 16)
        views = [{'face': f, 'script': 'Latn'} for f in (10, 11, 12, 10, 12, 11, 11, 10)]; e = F.normalize(torch.randn(8, 16), dim=1); follow = F.normalize(torch.randn(8, 16), dim=1)
        whole = losses(e, heads, proxies, setup, views, 'cpu', 0.0, follow)[0]
        parts = sum(losses(e[k::2], heads, proxies, setup, views[k::2], 'cpu', 0.0, follow[k::2])[0]*len(views[k::2])/len(views) for k in range(2))
        torch.testing.assert_close(parts, whole, rtol=1e-5, atol=1e-6)

