"""Letter-by-letter style distance: render the same glyphs in every face and compare them glyph by glyph.

This is the ground truth for "style": computed from font files, independent of any query text. Twins are faces
closer than the renderer's own noise floor, measured by re-rendering the same face with a sub-pixel shift and
without hinting.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import io
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

from scripts.corpus import ROOT, CACHE
from train.corpus_data import without_hints
from train.encoder_data import save
from train.robustness import read, sha

OUT = ROOT/'.data/style'
SIZE = 48; SCALE = 3; MIN_FAMILIES = 10
FIXED = {'Latn':'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
         'Cyrl':'абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
         'Grek':'αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'}
# Reference glyph height and baseline per script; other scripts normalize their median glyph height.
LAYOUT = {'Latn':('H',22,34),'Cyrl':('Н',22,34),'Grek':('Η',22,34),'Hani':('国',30,39),'Hira':('あ',28,38),'Kana':('ロ',26,37),'Hang':('한',30,39),'Bopo':('ㄅ',28,38)}
OTHER = (None,24,34)


def glyph_sets(families):
    counts = {}
    for f in families:
        for script, alphabet in f['alphabets'].items(): counts.setdefault(script, Counter()).update(set(alphabet))
    families_per_script = Counter(s for f in families for s in f['alphabets'])
    sets = {}
    for script, n in families_per_script.items():
        if n < MIN_FAMILIES: continue
        if script in FIXED: sets[script] = FIXED[script]; continue
        size = 200 if script == 'Hani' else 64
        # Hangul style lives in composed syllables; loose jamo are more widely covered but unrepresentative.
        ranked = sorted(((c, k) for c, k in counts[script].items() if script != 'Hang' or '가' <= c <= '힣'), key=lambda v: (-v[1], v[0]))
        sets[script] = ''.join(c for c, _ in ranked[:size] if c.isprintable() and not c.isspace())
    return sets


def enumerate_faces(inventory):
    """Every face: static files at their weight, variable files at each hundred within their weight axis."""
    faces = []; seen = set()
    for family in inventory['families']:
        if family['excluded']: continue
        for face in family['faces']:
            spec = face['axes'].get('wght')
            weights = [w for w in range(100, 1000, 100) if spec['min'] <= w <= spec['max']] if spec else [face['weight']]
            for weight in weights:
                identity = f"{family['id']}/{weight}{'i' if face['italic'] else ''}"
                if identity in seen: continue
                seen.add(identity)
                axes = {tag: float(weight if tag == 'wght' else value['default']) for tag, value in face['axes'].items()}
                default = face['path'] == family['selected'] and axes == {k: float(v) for k, v in family['trainingAxes'].items()}
                faces.append({'id':identity,'family':family['id'],'name':family['family'],'weight':weight,'italic':face['italic'],
                              'path':face['path'],'blob':face['blob'],'axes':axes,'scripts':sorted(family['alphabets']),'default':default})
    defaults = Counter(f['family'] for f in faces if f['default'])
    for family in {f['family'] for f in faces} - set(defaults):
        # Static families: the selected file is the default face.
        candidates = [f for f in faces if f['family'] == family]
        chosen = min(candidates, key=lambda f: (f['italic'], abs(f['weight'] - 400)))
        chosen['default'] = True
    return faces


def load_font(face, size, unhinted=None):
    font = ImageFont.truetype(io.BytesIO(unhinted) if unhinted else str(CACHE/face['path']), size, layout_engine=ImageFont.Layout.RAQM)
    if face['axes']:
        with TTFont(CACHE/face['path'], lazy=True) as tt: order = [a.axisTag for a in tt['fvar'].axes]
        font.set_variation_by_axes([face['axes'].get(tag, 0) for tag in order])
    return font


def draw(face, script, chars, shift=0.0, hinted=True):
    """Glyphs on a SIZE×SIZE canvas, reference height normalized, baseline aligned, centred by advance."""
    reference, target, baseline = LAYOUT.get(script, OTHER)
    with TTFont(CACHE/face['path'], lazy=True) as tt: cmap = tt.getBestCmap() or {}
    have = np.array([ord(c) in cmap for c in chars]); out = np.zeros((len(chars), SIZE, SIZE), np.uint8)
    if not have.any(): return out, have
    unhinted = None
    if not hinted:
        try: unhinted = without_hints(CACHE/face['path'])
        except ValueError: pass  # CFF outlines: no TrueType program to remove
    try:
        probe = load_font(face, 200, unhinted)
        if reference and ord(reference) in cmap:
            _, top, _, bottom = probe.getbbox(reference, anchor='ls'); height = bottom - top
        else:
            heights = [probe.getbbox(c, anchor='ls') for c in chars[:64] if ord(c) in cmap]
            height = float(np.median([b[3] - b[1] for b in heights if b[3] > b[1]] or [0]))
        if height <= 0: return out, np.zeros_like(have)
        font = load_font(face, max(1, round(200 * target * SCALE / height)), unhinted)
        big = SIZE * SCALE
        for i, c in enumerate(chars):
            if not have[i]: continue
            advance = font.getlength(c); canvas = Image.new('L', (big, big), 0)
            ImageDraw.Draw(canvas).text(((big - advance) / 2 + shift * SCALE, baseline * SCALE), c, font=font, fill=255, anchor='ls')
            out[i] = np.asarray(canvas.resize((SIZE, SIZE), Image.Resampling.BOX))
            have[i] = out[i].any()
    except (OSError, ValueError):
        if hinted: return draw(face, script, chars, shift, hinted=False)
        return out, np.zeros_like(have)
    return out, have


def render_job(job):
    face, script, chars = job
    return draw(face, script, chars)


def blur(x, sigma=1.1):
    k = torch.arange(-3, 4, dtype=torch.float32, device=x.device); k = torch.exp(-k**2/(2*sigma**2)); k = k/k.sum()
    x = torch.nn.functional.conv2d(x[:, None], k.view(1, 1, 1, -1), padding=(0, 3))
    return torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1), padding=(3, 0))[:, 0]


def distances(glyphs, have, device='mps'):
    """Mean normalized squared difference over glyphs both faces contain; inf where fewer than 60% are shared."""
    n, count = have.shape; num = torch.zeros(n, n, device=device); shared = torch.zeros(n, n, device=device)
    for i in range(count):
        x = blur(torch.from_numpy(glyphs[:, i].astype(np.float32) / 255).to(device)).flatten(1)
        h = torch.from_numpy(have[:, i]).to(device).float(); sq = (x*x).sum(1)
        d = ((sq[:, None] + sq[None] - 2*x@x.T) / (sq[:, None] + sq[None] + 1e-6)).clamp(min=0)
        mask = h[:, None]*h[None]; num += d*mask; shared += mask
    result = (num/shared.clamp(min=1)).cpu().numpy(); result[shared.cpu().numpy() < .6*count] = np.inf
    np.fill_diagonal(result, 0)
    return result


def pair_distance(a, ha, b, hb):
    both = ha & hb
    if not both.any(): return np.inf
    x = blur(torch.from_numpy(a[both].astype(np.float32)/255)).flatten(1); y = blur(torch.from_numpy(b[both].astype(np.float32)/255)).flatten(1)
    return float((((x-y)**2).sum(1)/((x*x).sum(1)+(y*y).sum(1)+1e-6)).mean())


def noise_floor(faces, chars, count=300):
    """Distance between two renders of the same face: 1/3 px shift and hinting removed. Below this, faces are indistinguishable."""
    rng = random.Random(20260923); sample = rng.sample([f for f in faces if 'Latn' in f['scripts']], count); values = []
    for face in sample:
        a, ha = draw(face, 'Latn', chars); b, hb = draw(face, 'Latn', chars, shift=1/3, hinted=False)
        if ha.sum() >= 40 and hb.sum() >= 40: values.append(pair_distance(a, ha, b, hb))
    values = np.array(values)
    return {'faces':len(values),'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'max':float(values.max())}


def teacher(script):
    """Face indices and their distance matrix for one script."""
    data = np.load(OUT/f'teacher-{script}.npz')
    return data['faces'], data['distances'].astype(np.float32)


def twins(matrix, threshold):
    """Pairwise twins: closer than the renderer noise. Deliberately not transitive: chains of adjacent weights
    or near-clones would otherwise merge distinguishable faces."""
    near = matrix < threshold; np.fill_diagonal(near, False)
    return near


def summarize(floor):
    faces = read(OUT/'faces.json')['faces']; threshold = floor['median']
    summary = {'faces':len(faces),'families':len({f['family'] for f in faces}),'noiseFloor':floor,'twinThreshold':threshold,
               'twinRule':'Pairwise, not transitive: faces closer than the median distance between two renders of one face (1/3 px shift, hinting removed).','scripts':{}}
    for path in sorted(OUT.glob('teacher-*.npz')):
        script = path.stem.split('-', 1)[1]; members, matrix = teacher(script); near = twins(matrix, threshold)
        default = np.array([faces[i]['default'] for i in members])
        summary['scripts'][script] = {'faces':len(members),'families':len({faces[i]['family'] for i in members}),
                                      'facesWithTwin':int(near.any(1).sum()),'familiesWithTwin':int((near[np.ix_(default, default)]).any(1).sum())}
    members, matrix = teacher('Latn'); position = {int(f): i for i, f in enumerate(members)}
    default = [i for i in range(len(members)) if faces[members[i]]['default']]
    names = {faces[members[i]]['family']: i for i in default}; neighbours = {}
    for name in ['roboto', 'montserrat', 'poppins', 'inter', 'lora', 'playfairdisplay', 'merriweather', 'geistmono']:
        row = matrix[names[name], default]; order = np.argsort(row, kind='stable')[1:9]
        neighbours[name] = [[faces[members[default[j]]]['family'], round(float(row[j]), 4)] for j in order]
    summary['nearestDefaultFaces'] = neighbours
    save(ROOT/'bench/style-teacher.json', summary)
    return summary


def build(workers=10):
    inventory = read(ROOT/'bench/corpus.json'); families = [f for f in inventory['families'] if not f['excluded']]
    faces = enumerate_faces(inventory); sets = glyph_sets(families)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT/'faces.json', {'inventorySha256':sha(ROOT/'bench/corpus.json'),'faces':faces})
    floor = noise_floor(faces, sets['Latn']); print('Noise floor', floor, flush=True)
    save(OUT/'noise-floor.json', floor)
    for script, chars in sorted(sets.items(), key=lambda s: s[0] != 'Latn'):
        members = [i for i, f in enumerate(faces) if script in f['scripts']]
        with ProcessPoolExecutor(workers) as pool:
            rendered = list(pool.map(render_job, [(faces[i], script, chars) for i in members], chunksize=16))
        glyphs = np.stack([g for g, _ in rendered]); have = np.stack([h for _, h in rendered])
        keep = have.sum(1) >= .6*len(chars); members = [m for m, k in zip(members, keep) if k]; glyphs, have = glyphs[keep], have[keep]
        matrix = distances(glyphs, have)
        np.savez_compressed(OUT/f'teacher-{script}.npz', faces=np.array(members), distances=matrix.astype(np.float16), chars=np.array(list(chars)))
        print(script, 'faces', len(members), flush=True)
    print(summarize(floor)['scripts'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['build', 'summarize'], nargs='?', default='build'); p.add_argument('--workers', type=int, default=10); a = p.parse_args()
    torch.set_num_threads(4)
    if a.command == 'build': build(a.workers)
    else: print(summarize(read(OUT/'noise-floor.json'))['scripts'])
