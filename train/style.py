"""Style encoder: identity per face, letter-by-letter style geometry and typed style heads, on fresh text every step.

One embedding serves swappable catalogs. Training signal:
  identity  margin softmax over face proxies; twins (closer than the renderer noise) share the target
  geometry  the view's similarity profile over all faces matches the letter-by-letter distance profile
  heads     weight, italic, script, Google category and fine class from the embedding
Selection uses the benchmark's development families only; the test families stay held out.
"""
import argparse
import csv
import re
import random
import time
import zlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.corpus import ROOT, CACHE
from train.encoder_data import save, SPLIT
from train.robustness import read, sha
from train.style_data import Stream, pools as make_pools, tensors
from train.style_teacher import glyph_sets, teacher, OUT as STYLE
from train.style_catalog import load as load_references, vectors as reference_vectors
from train.style_bench import load as load_bench
from train.ten import batch
from train.ten_model import Classifier, export, widen, CORPUS_ARCH, LARGE_ARCH, WIDER_ARCH

DIMENSIONS = 128
CATEGORIES = ['SANS_SERIF', 'SERIF', 'DISPLAY', 'HANDWRITING', 'MONOSPACE']
CLASSES = ('/Sans/', '/Serif/', '/Slab/', '/Script/', '/Monospace/')
SCALE, MARGIN, TEACHER_T, STUDENT_T, DRAWN_T, VIEW_T = 30.0, .15, .02, .05, .08, .05
WEIGHTS = {100:'Thin',200:'ExtraLight',300:'Light',400:'Regular',500:'Medium',600:'SemiBold',700:'Bold',800:'ExtraBold',900:'Black'}  # OpenType usWeightClass names


def taxonomy(families):
    """Google's own style labels: METADATA category per family, weighted fine-class tags (multi-label)."""
    folder = {f['id']: f['folder'] for f in families}; names = {f['family']: f['id'] for f in families}
    category = {}
    for fid, path in folder.items():
        m = re.search(r'category:\s*"(\w+)"', (CACHE/path/'METADATA.pb').read_text())
        if m and m.group(1) in CATEGORIES: category[fid] = CATEGORIES.index(m.group(1))
    tags = {}
    for name, _, tag, weight in csv.reader(open(STYLE/'google-tags.csv')):
        if name in names and tag.startswith(CLASSES): tags.setdefault(names[name], {})[tag] = float(weight)/100
    counts = {}
    for t in tags.values():
        for tag in t: counts[tag] = counts.get(tag, 0) + 1
    fine = sorted(tag for tag, n in counts.items() if n >= 20)
    return category, {f: np.array([t.get(tag, 0) for tag in fine], np.float32) for f, t in tags.items()}, fine


def themes(families):
    """Google's theme tags weighted at least 50 (Brush, Blackletter, Pixel…): reported, never trained."""
    names = {f['family']: f['id'] for f in families}; out = {}
    for name, _, tag, weight in csv.reader(open(STYLE/'google-tags.csv')):
        if name in names and tag.startswith('/Theme/') and float(weight) >= 50: out.setdefault(names[name], set()).add(tag)
    return out


class Heads(nn.Module):
    """Typed verdicts read from the normalized embedding; shipped beside the encoder."""
    def __init__(self, scripts, fine):
        super().__init__()
        self.weight = nn.Linear(DIMENSIONS, 1); self.italic = nn.Linear(DIMENSIONS, 1)
        self.script = nn.Linear(DIMENSIONS, len(scripts)); self.category = nn.Linear(DIMENSIONS, len(CATEGORIES)); self.fine = nn.Linear(DIMENSIONS, len(fine))

    def forward(self, e):
        return {'weight': self.weight(e)[:, 0]*300 + 400, 'italic': self.italic(e)[:, 0], 'script': self.script(e), 'category': self.category(e), 'fine': self.fine(e)}


def embed_views(model, pixels, sizes, owners, count):
    w = F.normalize(model(pixels, sizes), dim=1); keep = owners >= 0
    return F.normalize(torch.zeros(count, DIMENSIONS, device=w.device).index_add(0, owners[keep], w[keep]), dim=1)


def references(model, source='pillow', device='mps'):
    """Catalog vectors per (face, kind) from FreeType ('pillow') or Chromium ('browser') reference lines, or both
    averaged ('mixed'): screenshots come mostly from browsers, which draw strokes slightly heavier than FreeType."""
    embed = lambda m, p, s: embed_manifest(model, m, p, s, device)
    sets = [reference_vectors(embed, *load_references(name)) for name in {'pillow': ['references'], 'browser': ['references-browser'], 'mixed': ['references', 'references-browser']}[source]]
    merged = {}
    for vectors, keys in sets:
        for v, k in zip(vectors, keys): merged.setdefault(k, []).append(v)
    keys = sorted(merged); rows = np.stack([np.mean(merged[k], 0) for k in keys])
    return rows/np.linalg.norm(rows, axis=1, keepdims=True), keys


def architecture_of(state):
    """A checkpoint's encoder architecture; runs before architectures were recorded kept a `large` flag."""
    return state.get('architecture') or (LARGE_ARCH if state.get('large') else CORPUS_ARCH)


def embed_manifest(model, manifest, pixels, sources, device='mps'):
    """Stored windows -> one normalized vector per source (normalize windows, average, normalize: the browser rule)."""
    mapping = {s: i for i, s in enumerate(sources)}; ids = [i for i, w in enumerate(manifest['windows']) if w['source'] in mapping]
    out = torch.zeros(len(sources), DIMENSIONS, device=device); model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), 512):
            chunk = ids[start:start + 512]; x, sizes = batch(pixels, manifest['windows'], chunk)
            w = F.normalize(model(x.to(device), sizes.to(device)), dim=1)
            out.index_add_(0, torch.tensor([mapping[manifest['windows'][i]['source']] for i in chunk], device=device), w)
    return F.normalize(out, dim=1).cpu().numpy()


class Setup:
    """Faces, labels, teacher geometry and twins for a chosen set of training families."""
    def __init__(self, roles, device='mps'):
        self.inventory = read(ROOT/'bench/corpus.json'); families = [f for f in self.inventory['families'] if not f['excluded']]
        self.split = read(SPLIT); self.faces = read(STYLE/'faces.json')['faces']
        self.sets = glyph_sets(families); self.scripts = sorted(self.sets); self.pools = make_pools(families, self.sets)
        self.category, self.fine_labels, self.fine = taxonomy(families); self.themes = themes(families)
        self.floor = read(STYLE/'noise-floor.json')['median']
        self.train = [i for i, f in enumerate(self.faces) if self.split['families'][f['family']] in roles and self.pools.get(f['family'])]
        self.position = {f: i for i, f in enumerate(self.train)}
        n = len(self.train); distance = np.full((n, n), np.inf, np.float32)
        # Latin geometry for every face with Latin; faces without Latin use their own script's teacher.
        for script in ['Latn'] + [s for s in self.scripts if s != 'Latn']:
            members, matrix = teacher(script); rows = [self.position.get(int(m), -1) for m in members]; keep = [k for k, r in enumerate(rows) if r >= 0]
            if not keep: continue
            idx = np.array([rows[k] for k in keep]); block = matrix[np.ix_(keep, keep)]
            if script == 'Latn': distance[np.ix_(idx, idx)] = block
            else:
                fresh = np.isinf(distance[idx][:, idx]).all(1)  # rows of faces not yet covered by an earlier script
                for r, ok in zip(idx, fresh):
                    if ok: distance[r, idx] = block[list(idx).index(r)]
        np.fill_diagonal(distance, 0)
        self.distance = torch.from_numpy(distance).to(device)
        near = (distance < self.floor); np.fill_diagonal(near, True)
        self.twins = torch.from_numpy(near / near.sum(1, keepdims=True)).to(device, torch.float32)
        finite = np.where(np.isfinite(distance), distance, np.inf); np.fill_diagonal(finite, np.inf)
        self.neighbours = np.argsort(finite, axis=1)[:, :16]
        self.by_family = {}
        for k, i in enumerate(self.train): self.by_family.setdefault(self.faces[i]['family'], []).append(k)
        print('Training faces', n, 'families', len(self.by_family), 'faces with a twin', int((near.sum(1) > 1).sum()), flush=True)

    def plan(self, seed, views=2, faces=64, pairs=False):
        """Batches of faces and their nearest designs. Paired views cross case (one lowercase, one capitals or title) and
        often script, so the view loss learns what retrieval needs: one face from different letters."""
        rng = random.Random(seed); families = sorted(self.by_family)
        def make():
            anchors = [rng.choice(self.by_family[rng.choice(families)]) for _ in range(faces//2)]
            partners = [int(rng.choice(self.neighbours[a])) for a in anchors]
            specs = []
            for k in anchors + partners:
                i = self.train[k]; own = sorted(self.pools[self.faces[i]['family']]); other = [s for s in own if s != 'Latn'] or own
                # CJK fonts nearly always carry Latin too; without this their Hanzi, kana and Hangul would rarely be seen.
                latin = .4 if any(s in own for s in ('Hani', 'Hira', 'Kana', 'Hang')) else .75
                first = 'Latn' if 'Latn' in own and rng.random() < latin else rng.choice(other)
                cases = rng.sample(['lower', rng.choice(['upper', 'title'])], 2) if pairs else [None]*views
                for v in range(views):
                    script = first if v == 0 or rng.random() > (.5 if pairs else .3) else rng.choice(own)
                    specs.append((i, script, rng.getrandbits(48), rng.random() < .15, cases[v % len(cases)]))
            return specs
        return make


class Chromium:
    """Stored Chromium renders (development and case-diverse data) of the training families: the browser rasterizer,
    which the on-the-fly Pillow views do not reproduce. Labeled with each family's default face, as rendered."""
    SOURCES = [('.data/encoder/development.json', '.data/encoder/development.u8'), ('.data/encoder/case-training/manifest.json', '.data/encoder/case-training/pixels.u8')]

    def __init__(self, setup, roles=('train',)):
        default = {f['family']: i for i, f in enumerate(setup.faces) if f['default']}; self.pixels = []; self.views = []
        for manifest_path, pixels_path in self.SOURCES:
            manifest = read(ROOT/manifest_path); self.pixels.append(np.memmap(ROOT/pixels_path, dtype=np.uint8, mode='r')); owner = len(self.pixels) - 1
            windows = {}
            for w in manifest['windows']: windows.setdefault(w['source'], []).append((w['offset'], w['width'], w['height']))
            for i, s in enumerate(manifest['samples']):
                face = default.get(s['family'])
                if s['renderer'] == 'chromium' and s['role'] in roles and face in setup.position and s['script'] in setup.scripts:
                    self.views.append((owner, face, s['script'], windows[i]))
        print('Chromium training views', len(self.views), flush=True)

    def sample(self, rng, count):
        out = []
        for owner, face, script, windows in rng.sample(self.views, count):
            p = self.pixels[owner]; out.append({'face':face,'script':script,'windows':[np.asarray(p[o:o + w*h]).reshape(h, w) for o, w, h in windows]})
        return out


def losses(e, heads, proxies, setup, views, device, pairs=0.0):
    faces = torch.tensor([setup.position[v['face']] for v in views], device=device)
    P = F.normalize(proxies, dim=1); cos = e @ P.T
    target = setup.twins[faces]; drawn = torch.tensor([bool(v.get('drawn')) for v in views], device=device)
    # A drawing keeps a font's style but not its exact letterforms: it learns the style neighbourhood (a broad
    # letter-by-letter target), never the identity of one face.
    each = -(target*F.log_softmax(SCALE*(cos - MARGIN*(target > 0)), dim=1)).sum(1)
    identity = each[~drawn].mean() if (~drawn).any() else cos.sum()*0
    rows = setup.distance[faces]; valid = torch.isfinite(rows); temperature = torch.where(drawn, DRAWN_T, TEACHER_T)[:, None]
    teacher_p = F.softmax(torch.where(valid, -rows/temperature, torch.full_like(rows, -1e4)), dim=1)
    student = F.log_softmax(torch.where(valid, cos/STUDENT_T, torch.full_like(cos, -1e4)), dim=1)
    geometry = (teacher_p*(torch.log(teacher_p.clamp(min=1e-12)) - student)).sum(1).mean()
    out = heads(e); meta = [setup.faces[v['face']] for v in views]
    weight = F.smooth_l1_loss(out['weight']/300, torch.tensor([m['weight'] for m in meta], device=device, dtype=torch.float32)/300)
    italic = F.binary_cross_entropy_with_logits(out['italic'], torch.tensor([float(m['italic']) for m in meta], device=device))
    # Scripts too rare for a head class (Lao, Myanmar, N'Ko...) still train identity and style, not the script head.
    labelled = [k for k, v in enumerate(views) if v['script'] in setup.scripts]
    script = F.cross_entropy(out['script'][labelled], torch.tensor([setup.scripts.index(views[k]['script']) for k in labelled], device=device)) if labelled else cos.sum()*0
    known = [k for k, m in enumerate(meta) if m['family'] in setup.category]
    category = F.cross_entropy(out['category'][known], torch.tensor([setup.category[meta[k]['family']] for k in known], device=device)) if known else cos.sum()*0
    tagged = [k for k, m in enumerate(meta) if m['family'] in setup.fine_labels]
    fine = F.binary_cross_entropy_with_logits(out['fine'][tagged], torch.tensor(np.stack([setup.fine_labels[meta[k]['family']] for k in tagged]), device=device)) if tagged else cos.sum()*0
    # View to view, as retrieval compares a query with reference lines: a view's positives are the batch's other views of
    # its face or a twin, every other view (its look-alike partner included) a negative.
    view = cos.sum()*0
    if pairs:
        valid = ~drawn[:, None] & ~drawn[None, :] & ~torch.eye(len(views), dtype=torch.bool, device=device)
        positive = (setup.twins[faces][:, faces] > 0) & valid; has = positive.any(1)
        logp = F.log_softmax((e @ e.T/VIEW_T).masked_fill(~valid, -1e4), dim=1)
        if has.any(): view = (-(logp*positive).sum(1)[has]/positive.sum(1)[has]).mean()
    parts = {'identity':identity,'geometry':geometry,'views':view,'weight':weight,'italic':italic,'script':script,'category':category,'fine':fine}
    total = identity + geometry + pairs*view + .2*(weight + italic + script + category + fine)
    accuracy = (cos.argmax(1) == faces)[~drawn].float().mean() if (~drawn).any() else cos.sum()*0
    return total, parts, accuracy


def evaluate(model, heads, setup, roles, device='mps', name='bench', source='pillow', kinds=None):
    """Frozen benchmark against every face's references (all 2,004 families): identity, twins, style, weight, script.
    `kinds` keeps only those reference lines (Latn-lower, Cyrl…); queries whose family then has none are left out."""
    bench, bench_pixels = load_bench(name); vectors, keys = references(model, source, device)
    if kinds: keep = [i for i, (_, kind) in enumerate(keys) if kind in kinds]; vectors, keys = vectors[keep], [keys[i] for i in keep]
    faces = setup.faces; family_ids = sorted({faces[f]['family'] for f, _ in keys}); fpos = {f: i for i, f in enumerate(family_ids)}
    owner = np.array([fpos[faces[f]['family']] for f, _ in keys])
    selected = [i for i, q in enumerate(bench['samples']) if q['role'] in roles and q['family'] in fpos]; queries = [bench['samples'][i] for i in selected]
    q = embed_manifest(model, bench, bench_pixels, selected, device); verdict = None
    if heads is not None:
        with torch.no_grad(): verdict = {k: v.cpu().numpy() for k, v in heads(torch.from_numpy(q).to(device)).items()}
    sims = q @ vectors.T
    scores = np.full((len(q), len(family_ids)), -2, np.float32); best = np.zeros((len(q), len(family_ids)), np.int64)
    order = np.argsort(owner, kind='stable'); starts = np.r_[0, np.flatnonzero(np.diff(owner[order])) + 1]
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        cols = order[a:b]; block = sims[:, cols]; best[:, owner[cols[0]]] = cols[block.argmax(1)]; scores[:, owner[cols[0]]] = block.max(1)
    # Twins and style neighbours between default faces, Latin letter-by-letter distance.
    members, matrix = teacher('Latn'); row = {int(m): i for i, m in enumerate(members)}
    default = {f['family']: i for i, f in enumerate(faces) if f['default']}
    fam_rows = np.array([row.get(default[f], -1) for f in family_ids]); have = fam_rows >= 0
    D = np.full((len(family_ids), len(family_ids)), np.inf, np.float32); D[np.ix_(have, have)] = matrix[np.ix_(fam_rows[have], fam_rows[have])]
    rank_of = np.argsort(np.argsort(np.where(np.isfinite(D), D, 9), axis=1, kind='stable'), axis=1)
    supports = {s: np.array([s in setup.pools.get(f, {}) for f in family_ids]) for s in setup.scripts}
    rows = []
    for n, s in enumerate(queries):
        t = fpos[s['family']]
        for filtered in (False, True) if verdict is not None else (False,):
            sc = scores[n].copy()
            if filtered: sc[~supports[setup.scripts[int(verdict['script'][n].argmax())]]] = -3
            top = np.argsort(-sc, kind='stable')[:5]
            twin = [(j == t) or (D[t, j] < setup.floor) for j in top]
            face = faces[int(keys[best[n, top[0]]][0])]
            rows.append({'filtered':filtered,'role':s['role'],'script':s['script'],'case':s['case'],'condition':s['condition'],'slice':s.get('slice','main'),
                         'long':s['length'] >= 8,'top1':top[0] == t,'top5':t in top,'twin1':twin[0],'twin5':any(twin),
                         'style20':float(np.mean([rank_of[t, j] <= 20 for j in top if j != t])) if np.isfinite(D[t]).sum() > 100 else None,
                         'far':bool(any(rank_of[t, j] > 200 for j in top)) if np.isfinite(D[t]).sum() > 100 else None,
                         'category':float(np.mean([setup.category.get(family_ids[j]) == setup.category.get(s['family']) for j in top])),
                         'scriptOk':float(np.mean([s['script'] in setup.pools.get(family_ids[j], {}) for j in top])),
                         'faceWeightError':abs(face['weight'] - s['weight']) if top[0] == t or twin[0] else None,
                         'faceWeightBias':face['weight'] - s['weight'] if top[0] == t or twin[0] else None,
                         'faceItalic':face['italic'] == s['italic'] if top[0] == t or twin[0] else None,
                         'headWeightError':abs(float(verdict['weight'][n]) - s['weight']) if verdict is not None else None,
                         'headItalic':(verdict['italic'][n] > 0) == s['italic'] if verdict is not None else None,
                         'headScript':setup.scripts[int(verdict['script'][n].argmax())] == s['script'] if verdict is not None else None,
                         'style':CATEGORIES[setup.category[s['family']]] if s['family'] in setup.category else None,
                         'tags':[tag for tag, w in zip(setup.fine, setup.fine_labels.get(s['family'], [])) if w >= .5] + sorted(setup.themes.get(s['family'], ()))})
    def summarize(subset):
        out = {'count': len(subset)}
        for key in ['top1', 'top5', 'twin1', 'twin5', 'style20', 'far', 'category', 'scriptOk', 'faceWeightError', 'faceWeightBias', 'faceItalic', 'headWeightError', 'headItalic', 'headScript']:
            values = [r[key] for r in subset if r[key] is not None]
            if values: out[key] = float(np.mean(values))
        return out
    report = {}
    for filtered in (False, True) if verdict is not None else (False,):
        base = [r for r in rows if r['filtered'] == filtered]; name = 'scriptFiltered' if filtered else 'unfiltered'
        groups = {'all': base}
        for key in ['role', 'script', 'case', 'condition', 'slice', 'long', 'style']:
            for value in sorted({str(r[key]) for r in base}): groups[f'{key}/{value}'] = [r for r in base if str(r[key]) == value]
        for tag in setup.fine + sorted(set().union(*setup.themes.values())): groups['tag' + tag] = [r for r in base if tag in r['tags']]
        for role in sorted({r['role'] for r in base}):
            groups[f'{role}/latin-clean-5to10'] = [r for r in base if r['role'] == role and r['script'] == 'Latn' and r['condition'] == 'clean']
        report[name] = {k: summarize(v) for k, v in groups.items() if v}
    return report


def run_seed(run, seed=None):
    """A run's data seed: given, or derived from its name, so a continuation never replays its parent's batches."""
    return zlib.crc32(run.encode()) if seed is None else seed


def train(run, roles, steps, warm, workers, check=2500, architecture=CORPUS_ARCH, lr=2e-4, drawn=0.0, pairs=0.0, seed=None):
    seed = run_seed(run, seed); device = 'mps'; torch.manual_seed(seed); out = ROOT/'.data/style'/run; out.mkdir(parents=True, exist_ok=True)
    if (out/'progress.json').exists(): raise ValueError('Existing style run: ' + run)
    setup = Setup(roles, device)
    start_state = torch.load(warm, map_location='cpu', weights_only=False); source = architecture_of(start_state)
    model = Classifier(DIMENSIONS, architecture=source, dilations=[1, 1, 2, 2, 1]); model.load_state_dict(start_state['state'])
    # A wider run starts from the narrower model's exact function: duplicated channels, divided weights, a little noise.
    model = (model if source == architecture else widen(model, noise=1e-4, architecture=architecture)).to(device)
    heads = Heads(setup.scripts, setup.fine).to(device)
    if 'heads' in start_state: heads.load_state_dict(start_state['heads'])  # continuing a style run
    if 'proxies' in start_state and tuple(start_state['proxies'].shape) == (len(setup.train), DIMENSIONS):
        init = start_state['proxies'].numpy()  # continuing a style run with the same training faces
    else:
        # Imprint proxies from each training face's catalog references, so identity starts from the warm embedding.
        refs, ref_pixels = load_references(); vectors, keys = reference_vectors(lambda m, p, s: embed_manifest(model, m, p, s, device), refs, ref_pixels)
        init = np.random.default_rng(seed).normal(0, .01, (len(setup.train), DIMENSIONS)).astype(np.float32)
        for (f, _), v in zip(keys, vectors):
            if f in setup.position: init[setup.position[f]] += v
    proxies = nn.Parameter(torch.from_numpy(np.array(init, np.float32)).to(device))
    optimizer = torch.optim.AdamW([{'params':model.parameters(),'lr':lr,'weight_decay':1e-4},
                                   {'params':[proxies],'lr':2e-3,'weight_decay':0},{'params':heads.parameters(),'lr':2e-3,'weight_decay':1e-4}])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1, (s + 1)/500)*(.05 + .95*.5*(1 + np.cos(np.pi*min(s, steps)/steps))))
    stream = Stream(setup.faces, setup.pools, setup.plan(seed, pairs=pairs > 0), workers=workers, drawn=drawn)
    chromium = Chromium(setup, ('train',) if roles == ['train'] else ('train', 'validation', 'reference', 'test')); mix = random.Random(seed + 1)
    pins = {p: sha(ROOT/p) for p in ['train/style.py', 'train/style_data.py', 'train/style_teacher.py', 'train/style_catalog.py', 'train/ten_model.py', 'bench/corpus.json', 'bench/encoder-split.json']}
    history = []; best = -1; start = time.perf_counter(); running = {}
    def checkpoint(step):
        nonlocal best
        report = evaluate(model, heads, setup, ['development'], source='browser')  # the shipped Chromium references
        dev = report['scriptFiltered']['all']; value = dev['twin5']; drawn_result = None
        if drawn:  # a run that learns drawings selects on drawings too: their top-5 category agreement, development queries only
            drawn_result = evaluate(model, heads, setup, ['development'], name='bench-sketch', source='browser')['scriptFiltered']['all']; value = (value + drawn_result['category'])/2
        history.append({'step':step,'selection':value,'development':report['scriptFiltered']['all'],'unfiltered':report['unfiltered']['all'],'sketch':drawn_result,'seconds':time.perf_counter() - start})
        if value > best:
            best = value
            torch.save({'state':{k: v.detach().cpu().clone() for k, v in model.state_dict().items()},'heads':{k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
                        'proxies':proxies.detach().cpu().clone(),'step':step,'pins':pins,'scripts':setup.scripts,'fine':setup.fine,'architecture':architecture}, out/'best.pt')
        save(out/'progress.json', {'run':run,'roles':roles,'steps':steps,'warm':str(warm),'lr':lr,'pairs':pairs,'seed':seed,'architecture':architecture,'pins':pins,'history':history,'best':best})
        print(f'check {step}: dev twin-top5 {dev["twin5"]:.4f} top1 {dev["twin1"]:.4f} style20 {dev.get("style20", 0):.3f} far {dev.get("far", 0):.3f} '
              f'weightErr {dev.get("headWeightError", 0):.0f} italic {dev.get("headItalic", 0):.3f} script {dev.get("headScript", 0):.3f}' + (f' sketch twin-top5 {drawn_result["twin5"]:.4f} category {drawn_result["category"]:.4f} selection {value:.4f}' if drawn_result else '') + f' ({time.perf_counter() - start:.0f}s)', flush=True)
    try:
        if 'development' not in roles: checkpoint(0)
        for step in range(1, steps + 1):
            views = next(stream) + chromium.sample(mix, 32); model.train(); heads.train()
            pixels, sizes, owners = tensors(views, device)
            e = embed_views(model, pixels, sizes, owners, len(views))
            total, parts, accuracy = losses(e, heads, proxies, setup, views, device, pairs)
            optimizer.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_([*model.parameters(), proxies, *heads.parameters()], 5); optimizer.step(); schedule.step()
            for k, v in [*parts.items(), ('accuracy', accuracy)]: running[k] = running.get(k, 0) + v.detach()
            if step % 500 == 0:  # one device sync per report, not per step
                print(f'step {step}: ' + ' '.join(f'{k} {float(v)/500:.3f}' for k, v in running.items()) + f' ({time.perf_counter() - start:.0f}s)', flush=True); running = {}
            if step % check == 0 and 'development' not in roles: checkpoint(step)
        if 'development' in roles:
            torch.save({'state':{k: v.detach().cpu().clone() for k, v in model.state_dict().items()},'heads':{k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
                        'proxies':proxies.detach().cpu().clone(),'step':steps,'pins':pins,'scripts':setup.scripts,'fine':setup.fine,'architecture':architecture}, out/'best.pt')
            save(out/'progress.json', {'run':run,'roles':roles,'steps':steps,'warm':str(warm),'pins':pins,'history':[],'best':None})
    finally:
        stream.close()


def samples(model, heads, setup, device='mps', source='pillow'):
    """The eight reported demo samples ("Quiet rivers flow", 56 px, full resolution): rank and top five under the
    per-face catalog with the script filter, exactly as the demo searches."""
    from train.encoder_quality import OUT as QUALITY
    vectors, keys = references(model, source, device)
    faces = setup.faces; families = sorted({faces[f]['family'] for f, _ in keys}); fpos = {f: i for i, f in enumerate(families)}
    owner = np.array([fpos[faces[f]['family']] for f, _ in keys]); result = []
    for case in read(QUALITY/'demo-before.json'):
        chunks = []; windows = []; offset = 0
        for w in case['inputs']:
            raw = np.round(np.array(w['pixels'])*255).astype(np.uint8); windows.append({'source':0,'offset':offset,'width':w['width'],'height':w['height']}); chunks.append(raw); offset += len(raw)
        q = embed_manifest(model, {'windows': windows}, np.concatenate(chunks), [0], device)
        allowed = np.ones(len(families), bool)
        if heads is not None:
            with torch.no_grad(): script = setup.scripts[int(heads(torch.from_numpy(q).to(device))['script'][0].argmax())]
            covered = np.array([script in setup.pools.get(f, {}) for f in families])
            if covered.any(): allowed = covered
        sims = (q @ vectors.T)[0]; scores = np.full(len(families), -2.0)
        for j in range(len(families)): scores[j] = sims[owner == j].max() if allowed[j] else -2
        order = np.argsort(-scores, kind='stable'); family = case['source']['known'].replace('-', '')
        result.append({'family':family,'rank':int(np.flatnonzero(np.array(families)[order] == family)[0] + 1),'top5':[families[j] for j in order[:5]]})
    return result


def load_run(run, device='mps'):
    state = torch.load(ROOT/'.data/style'/run/'best.pt', map_location='cpu', weights_only=False)
    model = Classifier(DIMENSIONS, architecture=architecture_of(state), dilations=[1, 1, 2, 2, 1]); model.load_state_dict(state['state'])
    heads = Heads(state['scripts'], state['fine']); heads.load_state_dict(state['heads'])
    return model.to(device).eval(), heads.to(device).eval(), state


def export_run(run):
    """Int8 encoder in the browser's existing format; the typed heads travel in the same file as small float layers."""
    model, heads, state = load_run(run, 'cpu')
    artifact, _ = export(model.train(), [str(i) for i in range(DIMENSIONS)], read(ROOT/'models/encoder/encoder.json')['preparation'])
    del artifact['fonts']; artifact.update(kind='font-encoder', dimensions=DIMENSIONS, normalization='l2')
    layers = {k: v.numpy() for k, v in state['heads'].items()}
    layer = lambda name: {'weights': np.round(layers[name + '.weight'], 6).tolist(), 'bias': np.round(layers[name + '.bias'], 6).tolist()}
    artifact['heads'] = {'weight': {**layer('weight'), 'scale': 300, 'offset': 400}, 'italic': layer('italic'),
                         'script': {**layer('script'), 'labels': state['scripts']}, 'category': {**layer('category'), 'labels': CATEGORIES},
                         'fine': {**layer('fine'), 'labels': state['fine']}}
    save(ROOT/'.data/style'/run/'encoder.json', artifact); print('Exported', run, sha(ROOT/'.data/style'/run/'encoder.json'), flush=True)


def export_catalog(run, source='pillow'):
    """Per-face Google Fonts catalog (version 3): every face with its references, weight, style and script coverage.
    Default faces list, per script, the families whose letters there are indistinguishable from their own (closer than 95%
    of one face's re-renders), so the page shows one row per design: IBM Plex Sans KR folds under IBM Plex Sans for Latin,
    while Noto Sans Arabic and Noto Kufi Arabic, which share Latin letters, stay apart for Arabic. Accuracy credits only
    the stricter median twins."""
    import base64
    from train.encoder_catalog import preparation_hash, read_catalog
    encoder = ROOT/'.data/style'/run/'encoder.json'; model, _, _ = load_run(run); vectors, keys = references(model, source)
    inventory = read(ROOT/'bench/corpus.json'); by_id = {f['id']: f for f in inventory['families'] if not f['excluded']}
    faces = read(STYLE/'faces.json')['faces']; noise = read(STYLE/'noise-floor.json')['p95']
    chosen = sorted({f for f, _ in keys}, key=lambda i: (faces[i]['family'], faces[i]['italic'], faces[i]['weight'])); index = {f: n for n, f in enumerate(chosen)}
    twins = {}
    for path in sorted(STYLE.glob('teacher-*.npz')):
        script = path.stem.split('-', 1)[1]; members, matrix = teacher(script); row = {int(m): i for i, m in enumerate(members)}
        defaults = [i for i in chosen if faces[i]['default'] and i in row]
        block = matrix[np.ix_([row[i] for i in defaults], [row[i] for i in defaults])]
        for n, i in enumerate(defaults):
            near = [faces[defaults[j]]['family'] for j in np.argsort(block[n], kind='stable') if j != n and block[n, j] < noise]
            if near: twins.setdefault(i, {})[script] = near
    entries = []
    for i in chosen:
        f = faces[i]; family = by_id[f['family']]
        name = WEIGHTS[min(900, max(100, round(f['weight']/100)*100))]; style_name = ('Italic' if name == 'Regular' else name + ' Italic') if f['italic'] else name
        entry = {'id':f['id'],'familyId':f['family'],'family':family['family'],'styleName':style_name,'weight':f['weight'],'style':'italic' if f['italic'] else 'normal',
                 'scripts':sorted(family['alphabets'])}
        if f['default']: entry.update(default=True, axes=f['axes'], sourcePath=family['selected'], sourceBlob=next(x['blob'] for x in family['faces'] if x['path'] == family['selected']))
        if i in twins: entry['twins'] = twins[i]
        entries.append(entry)
    if sorted(e['familyId'] for e in entries if e.get('default')) != sorted({e['familyId'] for e in entries}): raise ValueError('Every family needs exactly one default face')
    rows = vectors/np.linalg.norm(vectors, axis=1, keepdims=True); scales = np.maximum(np.abs(rows).max(1)/127, 1e-12)
    packed = np.round(rows/scales[:, None]).clip(-127, 127).astype(np.int8)
    catalog = {'version':3,'kind':'font-catalog','encoderSha256':sha(encoder),'preparationSha256':preparation_hash(read(encoder)['preparation']),
               'dimensions':DIMENSIONS,'sourceCommit':inventory['commit'],'referenceMethod':f'faces-cases-scripts-{source}','faces':entries,
               'vectors':{'encoding':'int8-base64','shape':list(packed.shape),'data':base64.b64encode(packed.tobytes()).decode(),
                          'scales':[float(f'{v:.6g}') for v in scales],'owners':[index[f] for f, _ in keys]}}
    target = ROOT/'.data/style'/run/'google-fonts.json'; save(target, catalog)
    decoded, owners, labels = read_catalog(read(target), encoder)
    if labels != [e['id'] for e in entries] or len(decoded) != len(keys): raise ValueError('Catalog round trip failed')
    print('Catalog', len(entries), 'faces', len(keys), 'references', target.stat().st_size, 'bytes', flush=True)
    return target


def compare_references(run):
    """Choose the catalog reference renderer on development families only."""
    setup = Setup(['train']); model, heads, _ = load_run(run); out = {}
    for source in ['pillow', 'browser', 'mixed']:
        r = evaluate(model, heads, setup, ['development'], source=source)['scriptFiltered']['all']
        out[source] = {k: r.get(k) for k in ['count', 'twin5', 'twin1', 'top5', 'category', 'faceWeightError', 'faceWeightBias', 'faceItalic']}
        print(source, out[source], flush=True)
    chosen = max(out, key=lambda k: (out[k]['twin5'], -abs(out[k]['faceWeightBias'])))
    save(ROOT/'bench/style-references.json', {'run': run, 'chosen': chosen, 'results': out,
         'criterion': 'Development top-5 (twins counted); ties prefer the smaller signed weight bias. Test families excluded.'})
    return chosen


def final(runs, source='pillow'):
    """Frozen benchmark: style runs against the interim and the previous encoder, same per-face catalog of all families."""
    from train.encoder import load_encoder
    setup = Setup(['train']); roles = ['seen', 'development', 'test']; reports = {}
    for run in runs:
        model, heads, state = load_run(run)
        reports[run] = {'run': run, 'step': state['step'], 'architecture': architecture_of(state), 'results': evaluate(model, heads, setup, roles, source=source),
                        'samples': samples(model, heads, setup, source=source), 'sketch': evaluate(model, heads, setup, ['development', 'test'], name='bench-sketch', source=source)}
    for name, path in [('interim', ROOT/'.data/encoder/case-refine/encoder.json'), ('previous', ROOT/'.data/encoder/retrieval-refine/encoder.json')]:
        encoder = load_encoder(path).to('mps')
        reports[name] = {'encoderSha256': sha(path), 'results': evaluate(encoder, None, setup, roles, source=source), 'samples': samples(encoder, None, setup, source=source),
                         'sketch': evaluate(encoder, None, setup, ['development', 'test'], name='bench-sketch', source=source)}
    save(ROOT/'bench/style-quality.json', {'benchmarkSha256': sha(STYLE/'bench.json'), 'sketchSha256': sha(STYLE/'bench-sketch.json'), 'referenceSource': source,
         'referencesSha256': {name: sha(STYLE/f'{name}.json') for name in ['references', 'references-browser'] if (STYLE/f'{name}.json').exists()}, 'reports': reports,
         'scope': 'Frozen Chromium benchmark (seen, development-selection and held-out test families) against per-face references of all 2,004 families. Twins: pairwise letter-by-letter distance below the renderer noise median. Synthetic, not screenshots.'})
    return reports


def breakdown(run):
    """Held-out families by requirement: length, Google category and style tag against every reference; capitals searched
    among lowercase references only and the reverse; other scripts among Latin only. The shipped catalog holds every case
    and script, so the cross rows measure what a catalog with fewer references still finds."""
    setup = Setup(['train']); model, heads, _ = load_run(run); keep = lambda r: {k: r[k] for k in ['count', 'twin5', 'twin1', 'top5', 'top1'] if k in r}
    full = evaluate(model, heads, setup, ['test'], source='browser')['scriptFiltered']
    groups = {k: keep(v) for k, v in full.items() if k.split('/')[0] in ('long', 'case', 'script', 'slice', 'style', 'tag')}; cross = {}
    for name, kinds, group in [('capitalsFromLowercase', {'Latn-lower'}, 'case/upper'), ('lowercaseFromCapitals', {'Latn-upper'}, 'case/lower'),
                               ('otherScriptsFromLatin', {'Latn-lower', 'Latn-upper'}, 'case/native'), ('hanziFromLatin', {'Latn-lower', 'Latn-upper'}, 'slice/hanzi')]:
        cross[name] = {'references': sorted(kinds), 'queries': group, **keep(evaluate(model, heads, setup, ['test'], source='browser', kinds=kinds)['scriptFiltered'][group])}
        print(name, cross[name], flush=True)
    save(ROOT/'bench/style-breakdown.json', {'run': run, 'encoderSha256': sha(ROOT/'.data/style'/run/'encoder.json'), 'benchmarkSha256': sha(STYLE/'bench.json'),
         'referencesSha256': sha(STYLE/'references-browser.json'), 'groups': groups, 'cross': cross,
         'scope': 'Held-out test families, frozen Chromium benchmark, script-aware search; twin5/twin1 credit identical designs as the main report does. Tags: Google style tags weighted at least 50.'})
    return groups, cross


def deploy(run):
    """Ship the selected run: encoder (with heads), checkpoint and per-face catalog into models/encoder; the demo's
    accuracy line reads the held-out test slice of the frozen benchmark, bound to these exact bytes."""
    import shutil
    source = ROOT/'.data/style'/run; target = ROOT/'models/encoder'; report = read(ROOT/'bench/style-quality.json')
    if run not in report['reports']: raise ValueError('Run missing from the final report')
    for name in ['encoder.json', 'google-fonts.json', 'best.pt']: shutil.copyfile(source/name, target/name)
    test = report['reports'][run]['results']['scriptFiltered']['role/test']
    # The page quotes these; bench/style-breakdown.json adds held-out case and script figures.
    report['demo'] = {'encoderSha256':sha(target/'encoder.json'),'catalogSha256':sha(target/'google-fonts.json'),
                      'metrics':{'top1':test['twin1'],'top5Accuracy':test['twin5'],'faceWeightError':test.get('faceWeightError'),'faceItalic':test.get('faceItalic'),
                                 'drawnTop5':report['reports'][run]['sketch']['scriptFiltered']['all']['twin5'],
                                 'scope':'5–10 character Chromium crops of 300 unseen Google Fonts families (all conditions and scripts; identical designs counted)'}}
    save(ROOT/'bench/style-quality.json', report); print('Deployed', run, report['demo'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['train', 'export', 'catalog', 'final', 'deploy', 'compare', 'breakdown']); p.add_argument('--run', default='evaluation')
    p.add_argument('--roles', default='train'); p.add_argument('--steps', type=int, default=30000); p.add_argument('--workers', type=int, default=12); p.add_argument('--check', type=int, default=2500); p.add_argument('--large', action='store_true'); p.add_argument('--wider', action='store_true'); p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--drawn', type=float, default=0.0); p.add_argument('--pairs', type=float, default=0.0); p.add_argument('--seed', type=int); p.add_argument('--source', choices=['pillow', 'browser', 'mixed'], default='pillow')
    p.add_argument('--warm', default=str(ROOT/'.data/encoder/case-refine/best.pt')); a = p.parse_args()
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
    if a.command == 'train': train(a.run, a.roles.split(','), a.steps, a.warm, a.workers, a.check, WIDER_ARCH if a.wider else LARGE_ARCH if a.large else CORPUS_ARCH, a.lr, a.drawn, a.pairs, a.seed)
    elif a.command == 'export': export_run(a.run)
    elif a.command == 'catalog': export_catalog(a.run, source=a.source)
    elif a.command == 'compare': compare_references(a.run)
    elif a.command == 'deploy': deploy(a.run)
    elif a.command == 'breakdown': breakdown(a.run)
    else: final(a.run.split(','), a.source)
