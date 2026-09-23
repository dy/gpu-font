"""Frozen style benchmark: Chromium-rendered random strings of 5-10 characters, all cases, other scripts, damage.

Families: 300 seen during training, the 301 development families (checkpoint selection) and the 300 final
families (held out: a swapped catalog). Faces: default, bold and italic where they exist. Frozen before any
style training result; later commands refuse changed pins or pixels.
"""
import argparse
import base64
import hashlib
import io
import json
import random
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image

from scripts.corpus import ROOT, CACHE
from train.encoder_data import save, validate_shard, SPLIT
from train.robustness import read, sha
from train.style_data import pools as make_pools, FREQUENT
from train.style_teacher import glyph_sets

OUT = ROOT/'.data/style'
# The plan file freezes every text, size and face; pixels then depend only on the renderer and the preparation.
PINS = ['scripts/style-bench-browser.mjs', 'scripts/corpus-prepare.mjs', 'src/line.mjs', 'src/input.mjs', 'src/prepare.mjs', 'bench/corpus.json', 'bench/encoder-split.json']
SEED = 20260923


def ordered(items, salt):
    return sorted(items, key=lambda v: hashlib.sha256(f'{salt}:{v}'.encode()).hexdigest())


def query_text(rng, script, pool, case):
    if script == 'Latn':
        letters = [c for c in FREQUENT if c in pool]; length = rng.randint(5, 10)
        value = ''.join(rng.choices(letters, k=length))
        if case == 'upper': value = value.upper()
        elif case == 'title': value = value[:1].upper() + value[1:]
        elif case == 'mixed': value = ''.join(c.upper() if rng.random() < .35 else c for c in value)
        return ''.join(c if c in pool else c.lower() for c in value)
    length = rng.randint(3, 6) if script in ('Hani', 'Hira', 'Kana', 'Hang', 'Bopo') else rng.randint(4, 8)
    return ''.join(rng.choices(pool, k=length))


def plan():
    target = OUT/'bench-plan.json'
    split = read(SPLIT); inventory = read(ROOT/'bench/corpus.json'); families = [f for f in inventory['families'] if not f['excluded']]
    faces = read(OUT/'faces.json')['faces']; pools = make_pools(families, glyph_sets(families))
    groups = {'seen':ordered([f for f, r in split['families'].items() if r == 'train'], 'seen')[:300],
              'development':[f for f, r in split['families'].items() if r == 'development'], 'test':[f for f, r in split['families'].items() if r == 'test']}
    by_family = {}
    for i, f in enumerate(faces): by_family.setdefault(f['family'], []).append(i)
    queries = []; rng = random.Random(SEED)
    for role, members in groups.items():
        for family in sorted(members):
            own = pools.get(family, {})
            if not own: continue
            candidates = by_family[family]; default = next(i for i in candidates if faces[i]['default'])
            upright = [i for i in candidates if not faces[i]['italic']]
            bold = min(upright, key=lambda i: abs(faces[i]['weight'] - 700)) if upright else None
            italic = [i for i in candidates if faces[i]['italic'] and faces[i]['weight'] == faces[default]['weight']]
            chosen = [default] + ([bold] if bold is not None and faces[bold]['weight'] >= 600 and bold != default else []) + italic[:1]
            for face in chosen:
                scripts = sorted(own); latin = 'Latn' in own
                items = [('Latn', case) for case in ['lower', 'upper', 'title', 'mixed']] if latin else []
                others = [s for s in scripts if s != 'Latn']
                if others: items.append((rng.choice(others), 'native'))
                for n, (script, case) in enumerate(items):
                    value = query_text(rng, script, own[script], case); size = rng.choice([16, 20, 24, 32, 40, 48, 64])
                    base = {'role':role,'family':family,'face':face,'faceId':faces[face]['id'],'weight':faces[face]['weight'],'italic':faces[face]['italic'],
                            'script':script,'case':case,'text':value,'size':size,'length':len(value)}
                    queries.append({**base,'condition':'clean'})
                    if n == 0: queries.append({**base,'condition':'degraded'})
                    if n == 1: queries.append({**base,'condition':'dark'})
    # Chinese slice: every Hanzi family (most are Noto, kept in training by the lineage split), default face.
    role_of = {'train':'seen','development':'development','test':'test'}
    for family in sorted(f for f, own in pools.items() if 'Hani' in own):
        face = next(i for i in by_family[family] if faces[i]['default'])
        for n in range(3):
            value = query_text(rng, 'Hani', pools[family]['Hani'], 'native'); size = rng.choice([16, 20, 24, 32, 40, 48, 64])
            base = {'role':role_of[split['families'][family]],'family':family,'face':face,'faceId':faces[face]['id'],'weight':faces[face]['weight'],
                    'italic':faces[face]['italic'],'script':'Hani','case':'native','text':value,'size':size,'length':len(value),'slice':'hanzi'}
            queries.append({**base,'condition':'clean'})
            if n == 0: queries.append({**base,'condition':'degraded'})
    data = {'pins':{p: sha(ROOT/p) for p in PINS},'facesSha256':sha(OUT/'faces.json'),'queries':queries}
    if target.exists() and read(target) != data: raise ValueError('Changed frozen benchmark plan')
    save(target, data); print('Benchmark plan', len(queries), 'queries', {r: sum(q['role'] == r for q in queries) for r in groups}, flush=True)


def instance(face):
    """Static instance for the browser, so optical size and other variable defaults cannot change the face."""
    path = OUT/'instances'/(face['id'].replace('/', '--') + '.ttf')
    if path.exists(): return str(path)
    raw = (CACHE/face['path']).read_bytes()
    if face['axes']:
        with TTFont(io.BytesIO(raw), recalcTimestamp=False) as font:
            instantiateVariableFont(font, face['axes'], inplace=True); stream = io.BytesIO(); font.save(stream); raw = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True); path.with_suffix('.part').write_bytes(raw); path.with_suffix('.part').replace(path)
    return str(path)


def instances(workers, references=False):
    """Static instances for the benchmark faces, or for every face with catalog references (browser-rendered references)."""
    faces = read(OUT/'faces.json')['faces']
    needed = sorted({s['face'] for s in read(OUT/'references.json')['samples']} if references else {q['face'] for q in read(OUT/'bench-plan.json')['queries']})
    with ProcessPoolExecutor(workers) as pool: paths = list(pool.map(instance, [faces[i] for i in needed], chunksize=4))
    save(OUT/('reference-instances.json' if references else 'bench-instances.json'), {faces[i]['id']: {'path': p, 'sha256': sha(Path(p))} for i, p in zip(needed, paths)})
    print('Instances', len(paths), flush=True)


def degrade(raw, width, height, condition):
    image = Image.frombytes('L', (width, height), raw)
    if condition == 'degraded':
        image = image.resize((max(1, round(width*.75)), max(1, round(height*.75))), Image.Resampling.BILINEAR)
        stream = io.BytesIO(); image.save(stream, 'JPEG', quality=70); image = Image.open(io.BytesIO(stream.getvalue())).convert('L')
    elif condition == 'dark':
        image = Image.fromarray(255 - np.asarray(image))
    return image


def pack():
    data = read(OUT/'bench-plan.json')
    for path, expected in data['pins'].items():
        if sha(ROOT/path) != expected: raise ValueError('Changed benchmark dependency: ' + path)
    rendered = read(OUT/'bench-browser.json'); raw = (OUT/'bench-browser.u8').read_bytes()
    if rendered['planSha256'] != sha(OUT/'bench-plan.json') or rendered['sha256'] != sha(OUT/'bench-browser.u8'): raise ValueError('Stale browser renders')
    images = []
    for q, r in zip(data['queries'], rendered['images']):
        image = degrade(raw[r['offset']:r['offset'] + r['width']*r['height']], r['width'], r['height'], q['condition'])
        images.append({'width': image.width, 'height': image.height, 'pixels': base64.b64encode(image.tobytes()).decode()})
    samples = []; windows = []; chunks = []; offset = 0
    for start in range(0, len(images), 2048):
        out = subprocess.run(['node', str(ROOT/'scripts/corpus-prepare.mjs')], input=json.dumps(images[start:start + 2048]), text=True, capture_output=True, check=True)
        for items in json.loads(out.stdout):
            if not items: raise ValueError('Blank benchmark query')
            source = len(samples); samples.append(data['queries'][source])
            for w in items:
                raw = base64.b64decode(w['pixels']); windows.append({'source':source,'offset':offset,'width':w['width'],'height':w['height']}); chunks.append(raw); offset += len(raw)
    pixels = b''.join(chunks); (OUT/'bench.u8').write_bytes(pixels)
    manifest = {'pins':data['pins'],'planSha256':sha(OUT/'bench-plan.json'),'browser':rendered['browser'],'sha256':sha(OUT/'bench.u8'),'samples':samples,'windows':windows,
                'scope':'Frozen before style training results. Chromium canvas renders of static instances; random 5-10 character Latin strings in four cases, other-script strings, degraded and dark copies. Synthetic, not screenshots.'}
    validate_shard(manifest, len(pixels)); save(OUT/'bench.json', manifest)
    print('Packed benchmark', len(samples), 'queries', len(windows), 'windows', flush=True)


def sketches():
    """Frozen synthetic hand-drawn slice: clean Latin development and test renders (32 px and larger) redrawn by the
    sketch transform with fixed seeds. The same transform generates training views, so this bounds real drawings from above."""
    from train.style_data import sketch
    target = OUT/'bench-sketch.json'
    if target.exists(): raise ValueError('Sketch slice already frozen')
    data = read(OUT/'bench-plan.json'); rendered = read(OUT/'bench-browser.json'); raw = (OUT/'bench-browser.u8').read_bytes()
    if rendered['planSha256'] != sha(OUT/'bench-plan.json') or rendered['sha256'] != sha(OUT/'bench-browser.u8'): raise ValueError('Stale browser renders')
    chosen = [i for i, q in enumerate(data['queries']) if q['role'] in ('development', 'test') and q['script'] == 'Latn' and q['condition'] == 'clean' and q['size'] >= 32]
    images = []
    for i in chosen:
        r = rendered['images'][i]; image = sketch(Image.frombytes('L', (r['width'], r['height']), raw[r['offset']:r['offset'] + r['width']*r['height']]), random.Random(SEED + i))
        images.append({'width': image.width, 'height': image.height, 'pixels': base64.b64encode(image.tobytes()).decode()})
    out = subprocess.run(['node', str(ROOT/'scripts/corpus-prepare.mjs')], input=json.dumps(images), text=True, capture_output=True, check=True)
    samples = []; windows = []; chunks = []; offset = 0
    for i, items in zip(chosen, json.loads(out.stdout)):
        if not items: continue
        source = len(samples); samples.append({**data['queries'][i], 'condition': 'sketch'})
        for w in items:
            b = base64.b64decode(w['pixels']); windows.append({'source':source,'offset':offset,'width':w['width'],'height':w['height']}); chunks.append(b); offset += len(b)
    pixels = b''.join(chunks); (OUT/'bench-sketch.u8').write_bytes(pixels)
    manifest = {'pins':data['pins'],'planSha256':sha(OUT/'bench-plan.json'),'sha256':sha(OUT/'bench-sketch.u8'),'samples':samples,'windows':windows,
                'scope':'Synthetic hand-drawn slice: skeletonized benchmark renders redrawn with a round pen, wobble and tilt. Not real drawings.'}
    validate_shard(manifest, len(pixels)); save(target, manifest); print('Sketch slice', len(samples), 'queries', flush=True)


def load(name='bench'):
    manifest = read(OUT/f'{name}.json'); path = OUT/f'{name}.u8'
    for file, expected in manifest['pins'].items():
        if sha(ROOT/file) != expected: raise ValueError('Changed benchmark dependency: ' + file)
    if sha(path) != manifest['sha256']: raise ValueError('Changed benchmark pixels')
    validate_shard(manifest, path.stat().st_size)
    return manifest, np.memmap(path, dtype=np.uint8, mode='r')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['plan', 'instances', 'pack', 'sketches']); p.add_argument('--workers', type=int, default=8); p.add_argument('--references', action='store_true'); a = p.parse_args()
    if a.command == 'plan': plan()
    elif a.command == 'instances': instances(a.workers, a.references)
    elif a.command == 'sketches': sketches()
    else: pack()
