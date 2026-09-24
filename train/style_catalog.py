"""Catalog references rendered from font files: per face, lowercase and uppercase Latin plus each other script.

Lines are short so the three preparation windows cover every glyph; each reference averages its lines. The same
references serve checkpoint selection, the frozen evaluation and the shipped per-face catalogs.
"""
import argparse
import base64
import multiprocessing
import random

import numpy as np

from scripts.corpus import ROOT
from train.encoder_data import save, validate_shard
from train.robustness import read, sha
from train.style_data import Fonts, Preparer, pools as make_pools, CASED
from train.style_teacher import glyph_sets

OUT = ROOT/'.data/style'
# Three lines per case cover all 26 letters.
LATIN = {'lower': ['hamburgefonts', 'quickjivexy', 'dwarflpzm'], 'upper': ['HAMBURGEFONTS', 'QUICKJIVEXY', 'DWARFLPZM']}
SIZE = 48
STATE = {}


def kinds(pool):
    """Reference kinds and their lines for one family: each case of a cased script in lines covering its alphabet
    (queries mix cases, so one lowercase line would leave capitals to style transfer), and four lines of the most
    shared characters of every other script."""
    result = {}
    if 'Latn' in pool:
        for case, lines in LATIN.items():
            if all(c in pool['Latn'] for line in lines for c in line): result['Latn-' + case] = lines; continue
            # Faces missing a letter of the fixed lines use their own letters of that case.
            own = ''.join(c for c in pool['Latn'] if c.isascii() and c.isalpha() and (c.islower() if case == 'lower' else c.isupper()))
            if len(own) >= 8: result['Latn-' + case] = [own[i:i + 12] for i in range(0, min(len(own), 36), 12)]
    for script, chars in sorted(pool.items()):
        if script == 'Latn': continue
        if script in CASED:
            for case in ('lower', 'upper'):
                own = ''.join(c for c in chars if c.isalpha() and (c.islower() if case == 'lower' else c.isupper()))
                if len(own) >= 8: step = -(-len(own)//3); result[f'{script}-{case}'] = [own[i:i + step] for i in range(0, len(own), step)]
            continue
        head = chars[:32]
        if len(head) >= 8: result[script] = [head[i:i + 8] for i in range(0, len(head), 8)]
    return result


def init(faces):
    STATE.update(faces=faces, fonts=Fonts(), prepare=Preparer())


def render(job):
    """One reference line: clean Pillow render at 48 px, browser preparation."""
    from PIL import Image, ImageDraw
    index, text = job
    def draw(font):
        left, top, right, bottom = font.getbbox(text)
        if right <= left or bottom <= top: return None
        image = Image.new('L', (right - left + 24, bottom - top + 24), 255)
        ImageDraw.Draw(image).text((12 - left, 12 - top), text, font=font, fill=0)
        return image
    image = STATE['fonts'].render(STATE['faces'][index], SIZE, draw)
    return STATE['prepare'](np.asarray(image, np.uint8)) if image is not None else []


def build(workers=10):
    faces = read(OUT/'faces.json')['faces']; families = [f for f in read(ROOT/'bench/corpus.json')['families'] if not f['excluded']]
    pools = make_pools(families, glyph_sets(families)); jobs = []; samples = []
    for index, face in enumerate(faces):
        for kind, lines in kinds(pools.get(face['family'], {})).items():
            for line in lines: jobs.append((index, line)); samples.append({'face':index,'kind':kind,'text':line})
    with multiprocessing.get_context('spawn').Pool(workers, initializer=init, initargs=(faces,)) as pool:
        rendered = pool.map(render, jobs, chunksize=32)
    windows = []; chunks = []; offset = 0; kept = []
    for sample, items in zip(samples, rendered):
        if not items: continue
        source = len(kept); kept.append(sample)
        for w in items:
            windows.append({'source':source,'offset':offset,'width':w.shape[1],'height':w.shape[0]}); chunks.append(w.tobytes()); offset += w.size
    pixels = b''.join(chunks); (OUT/'references.u8').write_bytes(pixels)
    manifest = {'facesSha256':sha(OUT/'faces.json'),'pins':{p: sha(ROOT/p) for p in ['train/style_catalog.py','scripts/style-prepare.mjs','src/line.mjs','src/input.mjs','src/prepare.mjs']},
                'sha256':sha(OUT/'references.u8'),'samples':kept,'windows':windows}
    validate_shard(manifest, len(pixels)); save(OUT/'references.json', manifest)
    print('References', len(kept), 'lines', len(windows), 'windows', 'faces', len({s['face'] for s in kept}), flush=True)


def load(name='references'):
    manifest = read(OUT/f'{name}.json'); path = OUT/f'{name}.u8'
    if sha(path) != manifest['sha256']: raise ValueError('Changed reference pixels')
    validate_shard(manifest, path.stat().st_size)
    return manifest, np.memmap(path, dtype=np.uint8, mode='r')


def vectors(embed_lines, manifest, pixels):
    """Average line embeddings into one normalized vector per (face, kind). Returns vectors and (face, kind) owners."""
    lines = embed_lines(manifest, pixels, list(range(len(manifest['samples']))))
    groups = {}
    for i, s in enumerate(manifest['samples']): groups.setdefault((s['face'], s['kind']), []).append(i)
    keys = sorted(groups); rows = np.stack([lines[groups[k]].mean(0) for k in keys])
    return rows/np.linalg.norm(rows, axis=1, keepdims=True), keys


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=10); a = p.parse_args(); build(a.workers)
