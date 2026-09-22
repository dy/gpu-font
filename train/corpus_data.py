"""Bounded, resumable full-catalog rendering; shared JS preparation, split text banks."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import random
import string
import subprocess

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps, features
from fontTools.ttLib import TTFont
from scripts.corpus import ROOT, CACHE, MANIFEST, blob
from train.hundred_data import transform, band
from train.robustness import read, sha

DATA = ROOT / '.data/corpus'
PIN_FILES = ['bench/corpus.json', 'train/corpus_data.py', 'train/hundred_data.py', 'scripts/corpus-prepare.mjs', 'src/line.mjs', 'src/input.mjs', 'src/prepare.mjs', 'requirements.txt']


def pins():
    return {**{p: sha(ROOT / p) for p in PIN_FILES}, 'pillow':PIL.__version__, 'freetype':features.version('freetype2'), 'raqm':features.version('raqm')}


def banks(families):
    counts = {}
    for family in families:
        for script, alphabet in family['alphabets'].items(): counts.setdefault(script, Counter()).update(set(alphabet))
    preferred = {'Latn': string.ascii_letters, 'Hani': '的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经'}
    return {script: preferred.get(script, ''.join(c for c, _ in sorted(count.items(), key=lambda v: (-v[1], v[0]))[:64])) for script, count in sorted(counts.items())}


def texts(alphabet, script, role, avoid=(), revision=1):
    if len(set(alphabet)) < 8 or len(set(alphabet)) != len(alphabet): raise ValueError('Invalid text alphabet')
    seed = int.from_bytes(hashlib.sha256((script + ':' + role + (f':v{revision}' if revision != 1 else '')).encode()).digest()[:8])
    rng = random.Random(seed)
    count = 64 if role == 'train' else 8
    used = {t.casefold() for t in avoid}; result = []
    lengths = [2 if len({c.casefold() for c in alphabet}) >= 6 else 3, 4, 8, 12]
    # No font ID enters text selection. Reject short-string collisions across all splits.
    for _ in range(100000):
        candidate = ''.join(rng.choices(alphabet, k=lengths[len(result) % 4]))
        if candidate.casefold() in used: continue
        result.append(candidate); used.add(candidate.casefold())
        if len(result) == count: return result
    raise ValueError('Text bank exhausted')


def plans(family, pools):
    result = []
    for script, coverage in sorted(family['alphabets'].items()):
        alphabet = ''.join(c for c in pools[script] if c in coverage)
        if len(alphabet) < 8: alphabet = ''.join(sorted(coverage))[:64]
        splits = {}; used = []
        for role in ['train', 'validation', 'test']:
            # The initial test crops informed an axis/rendering audit. Reserve fresh final strings.
            previous = texts(alphabet, script, role, used) if role == 'test' else []
            splits[role] = texts(alphabet, script, role, used + previous, revision=2 if role=='test' else 1)
            used.extend(splits[role])
        for role, values in splits.items():
            for i, text in enumerate(values):
                index = int.from_bytes(script.encode(), 'big') * 80 + {'train':0,'validation':64,'test':72}[role] + i
                result.append({'script': script, 'role': role, 'text': text, 'index': index, 'size': [20, 28, 40, 56][(i // 4 + i) % 4], 'light': bool(i % 2), 'length': band(text)})
    return result


def render_family(job):
    family, pools, expected = job
    dest = DATA / 'shards' / family['id']
    manifest = dest.with_suffix('.json'); packed = dest.with_suffix('.u8')
    if manifest.exists() and packed.exists():
        old = read(manifest)
        if old['pins'] == expected and old['sha256'] == sha(packed): return family['id'], len(old['samples']), old['rejected']
    source = CACHE / family['selected']
    face = next(f for f in family['faces'] if f['path'] == family['selected'])
    if blob(source.read_bytes()) != face['blob']: raise ValueError('Changed source font')
    font_cache = {}; images = []; samples = []; unhinted = None
    def render(plan):
        size = plan['size']
        if size not in font_cache:
            font_cache[size] = ImageFont.truetype(io.BytesIO(unhinted) if unhinted else str(source), size, layout_engine=ImageFont.Layout.RAQM)
            if face['axes']: font_cache[size].set_variation_by_axes([family['trainingAxes'][tag] for tag in face['axes']])
        font = font_cache[size]; left, top, right, bottom = font.getbbox(plan['text'])
        if right <= left or bottom <= top: raise ValueError('Empty supported text in ' + family['id'])
        image = Image.new('L', (right-left+12, bottom-top+12), 255)
        ImageDraw.Draw(image).text((6-left, 6-top), plan['text'], font=font, fill=0)
        return image
    for plan in plans(family, pools):
        try: image = render(plan)
        except OSError as error:
            # Broken TrueType hint programs must not become missing font families.
            if unhinted or 'execution context too long' not in str(error): raise
            unhinted = without_hints(source); font_cache.clear(); image = render(plan)
        conditions = ['clean', 'augmented'] if plan['role'] == 'train' else ['clean', 'rotate', 'small']
        for condition in conditions:
            result = transform(image, plan, condition)
            images.append({'width': result.width, 'height': result.height, 'pixels': base64.b64encode(result.tobytes()).decode()})
            samples.append({**plan, 'condition': condition})
    child = subprocess.run(['node', str(ROOT / 'scripts/corpus-prepare.mjs')], input=json.dumps(images), text=True, capture_output=True, check=True)
    prepared = json.loads(child.stdout)
    if len(prepared) != len(samples): raise ValueError('Missing prepared samples')
    windows = []; rejected = []
    with packed.with_suffix('.part').open('wb') as out:
        for index, (sample, items) in enumerate(zip(samples, prepared)):
            if not items: rejected.append(index)
            for item in items:
                raw = base64.b64decode(item.pop('pixels'), validate=True)
                if len(raw) != item['width'] * item['height']: raise ValueError('Truncated prepared pixels')
                windows.append({**item, 'source': index, 'offset': out.tell()}); out.write(raw)
    packed.with_suffix('.part').replace(packed)
    if rejected: raise ValueError(f'Blank prepared crops: {family["id"]} {rejected}')
    manifest.write_text(json.dumps({'family': family['id'], 'pins': expected, 'sha256': sha(packed), 'samples': samples, 'windows': windows, 'rejected': rejected, 'unhintedFallback':unhinted is not None}, ensure_ascii=False))
    return family['id'], len(samples), rejected


def without_hints(source):
    with TTFont(source, recalcTimestamp=False) as font:
        if 'glyf' not in font: raise ValueError('Expected TrueType outlines')
        font['glyf'].removeHinting()
        for tag in ['fpgm','prep','cvt ','hdmx','VDMX','LTSH']:
            if tag in font: del font[tag]
        result = io.BytesIO(); font.save(result)
        return result.getvalue()


def render(workers=6):
    if not features.check_feature('raqm'): raise ValueError('Complex script shaping requires RAQM')
    (DATA / 'shards').mkdir(parents=True, exist_ok=True)
    families = [f for f in read(MANIFEST)['families'] if not f['excluded']]
    pools = banks(families); expected = pins()
    (DATA / 'banks.json').write_text(json.dumps(pools, ensure_ascii=False))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(render_family, (family, pools, expected)) for family in families]
        failures = []
        for i, job in enumerate(as_completed(jobs), 1):
            try: name, count, rejected = job.result()
            except Exception as error:
                failures.append(str(error)); print(f'Render failed: {error}', flush=True); continue
            if i % 20 == 0 or i == len(jobs): print(f'Rendered {i}/{len(jobs)} families; {name}: {count} crops', flush=True)
    if failures: raise ValueError(f'{len(failures)} failed families: {failures}')


def pack():
    families = [f for f in read(MANIFEST)['families'] if not f['excluded']]
    expected = pins(); samples = []; windows = []; signatures = {}
    with (DATA / 'prepared.u8.part').open('wb') as out:
        for label, family in enumerate(families):
            dest = DATA / 'shards' / family['id']; manifest = read(dest.with_suffix('.json')); raw = dest.with_suffix('.u8').read_bytes()
            if manifest['pins'] != expected or hashlib.sha256(raw).hexdigest() != manifest['sha256']: raise ValueError('Changed shard')
            offset = out.tell(); owner = len(samples); end = 0
            script_hashes = {}
            for window in manifest['windows']:
                if window['offset'] != end or not 0 <= window['source'] < len(manifest['samples']) or not 1 <= window['width'] <= 128 or not 1 <= window['height'] <= 48: raise ValueError('Invalid packed window')
                end += window['width'] * window['height']
                windows.append({**window, 'source': window['source'] + owner, 'offset': window['offset'] + offset})
                sample = manifest['samples'][window['source']]
                if sample['role'] == 'train':
                    digest = script_hashes.setdefault(sample['script'], hashlib.sha256())
                    digest.update(json.dumps([sample['text'], sample['condition'], window['width'], window['height']]).encode())
                    digest.update(raw[window['offset']:end])
            if end != len(raw): raise ValueError('Truncated/trailing shard')
            samples.extend({**s, 'label': label, 'family': family['id']} for s in manifest['samples'])
            for script, digest in script_hashes.items(): signatures.setdefault((script, digest.hexdigest()), []).append(label)
            out.write(raw)
    (DATA / 'prepared.u8.part').replace(DATA / 'prepared.u8')
    aliases = [{'script': script, 'labels': labels, 'sha256': digest} for (script, digest), labels in signatures.items() if len(labels)>1]
    (DATA / 'prepared.json').write_text(json.dumps({'fonts': [f['id'] for f in families], 'pins': expected, 'sha256': sha(DATA / 'prepared.u8'), 'aliases': aliases, 'samples': samples, 'windows': windows}, ensure_ascii=False, separators=(',', ':')))
    print(f'{len(families)} families, {len(samples)} images, {len(windows)} windows', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['render', 'pack']); p.add_argument('--workers', type=int, default=6); args = p.parse_args()
    if args.command == 'render': render(args.workers)
    else: pack()
