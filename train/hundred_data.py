"""100 pinned regular families; balanced text lengths and held-out renderings."""
import argparse
import io
from pathlib import Path
import random
import re
import string

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from scripts.fixtures import render_sample, check_coverage
from train.robustness import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/hundred'
ALPHABET = string.ascii_letters + string.digits


def import_fonts(source):
    original = read(ROOT / 'bench/fonts.json')['fonts']
    entries = {f['family']: f for f in original}
    candidates = {}
    for path in sorted(source.glob('*.ttf')):
        with TTFont(path, recalcTimestamp=False) as face:
            names = face['name']; family = names.getDebugName(16) or names.getDebugName(1)
            if family in entries or family in candidates:
                continue
            axes = {a.axisTag: a.defaultValue for a in face['fvar'].axes} if 'fvar' in face else {}
            if 'wght' in axes: axes['wght'] = 400
            if 'wdth' in axes: axes['wdth'] = 100
            if face['post'].italicAngle or (400 if 'wght' in axes else face['OS/2'].usWeightClass) != 400:
                continue
            if 'Open Font License' not in (names.getDebugName(13) or ''):
                continue
            check_coverage(face, [ALPHABET])
            ident = re.sub('[^a-z0-9]+', '-', family.lower()).strip('-')
            entry = dict(id=ident, family=family, file=path.name, sourceSha256=sha(path), axes=axes,
                         license=names.getDebugName(13), licenseUrl=names.getDebugName(14),
                         copyright=names.getDebugName(0), sourceVersion=names.getDebugName(5))
            if axes: instantiateVariableFont(face, axes, inplace=True)
            if face['OS/2'].usWeightClass != 400 or face['post'].italicAngle:
                raise ValueError(f'Not an upright regular instance: {path}')
            dest = ROOT / f'.data/fonts/{ident}.ttf'; face.save(dest)
            entry['instanceSha256'] = sha(dest); candidates[family] = entry
    # Preserve the original collection, then a seeded, reproducible family selection.
    extra = sorted(candidates.values(), key=lambda f: f['id'])
    random.Random(20260923).shuffle(extra)
    all_fonts = list(entries.values()) + extra
    if len(all_fonts) < 110: raise ValueError('Need 100 classes plus disjoint unknown families')
    for i, f in enumerate(all_fonts):
        f['split'] = 'train' if i < 100 else 'unknown-validation' if i % 2 == 0 else 'unknown-test'
    write(ROOT / 'bench/fonts-100.json', {'version': 1, 'source': 'Local fontr Google collection; upstream revisions unknown', 'fonts': all_fonts})
    config = {'data': {'seed': 20260923, 'fonts': [f['id'] for f in all_fonts[:100]],
                      'input': {'width': 128, 'height': 48, 'windows': 3},
                      'trainViews': ['clean', 'augmented'],
                      'conditions': ['clean', 'edge', 'small', 'rotate', 'shadow', 'margin']},
              'training': {'seed': 20260923, 'steps': 40000, 'batch': 64, 'learningRate': .002,
                           'threads': 4, 'device': 'mps', 'dilations': [1, 1, 2, 2]}}
    write(ROOT / 'bench/hundred.json', config)
    print(f'{len(all_fonts[:100])} known families; {len(all_fonts[100:])} unknown families', flush=True)


def band(text):
    n = len(text.replace(' ', ''))
    return '1' if n == 1 else '2-3' if n <= 3 else '4-7' if n <= 7 else '8+'


def plans():
    rng = random.Random(20260923)
    texts = read(ROOT / 'bench/hundred-texts.json') if (ROOT / 'bench/hundred-texts.json').exists() else None
    if texts is None:
        words = sorted({w.lower() for w in Path('/usr/share/dict/words').read_text().splitlines() if re.fullmatch('[a-zA-Z]{4,7}', w)})
        rng.shuffle(words)
        texts = {'train': list(ALPHABET), 'validation': list('agRi'), 'test': list('bepqH7')}
        used = set(ALPHABET)
        for role, count in [('train', 64), ('validation', 6), ('test', 10)]:
            short = []
            while len(short) < count:
                s = ''.join(rng.choices(ALPHABET, k=rng.choice([2, 3])))
                if s not in used: short.append(s); used.add(s)
            if role == 'validation': short[0:2] = ['ri', 'riv']
            texts[role] += short
            for i in range(count):
                w = words.pop(); used.add(w)
                texts[role].append(w.upper() if i % 4 == 0 else w)
            texts[role] += [' '.join(words.pop() for _ in range(rng.choice([2, 3]))) for _ in range(count)]
        write(ROOT / 'bench/hundred-texts.json', texts)
    # Single characters intentionally overlap: evaluate unseen renderings, not an unseen alphabet.
    sets = [{t.lower() for t in values if len(t) > 1} for values in texts.values()]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]): raise ValueError('Multi-character text leakage')
    rng = random.Random(20260923)  # Rendering plans do not depend on whether pools were just created.
    result = []
    for role, values in texts.items():
        for index, text in enumerate(values):
            result.append({'role': role, 'index': index, 'text': text, 'length': band(text),
                           'size': rng.choice([18, 24, 32, 44, 60]), 'dpr': rng.choice([1, 1, 2]),
                           'light': bool(rng.randrange(2))})
    DATA.mkdir(parents=True, exist_ok=True); write(DATA / 'plans.json', result)
    return result


def transform(base, plan, condition):
    image = base.convert('L')
    rng = random.Random(8917 + plan['index'] * 31)
    if condition == 'augmented':
        condition = rng.choice(['edge', 'small', 'rotate', 'shadow', 'margin', 'blur', 'jpeg'])
    if condition == 'edge':
        # A partial last glyph, with enough ink retained even for a single letter.
        box = ImageOps.invert(image).getbbox()
        if box:
            left, top, right, bottom = box
            fraction = .92 if plan['length'] == '1' else .82
            image = image.crop((max(0, left - 2), max(0, top - 2), max(left + 1, round(left + (right - left) * fraction)), min(image.height, bottom + 2)))
    elif condition == 'small':
        image = image.resize((max(1, round(image.width * .45)), max(1, round(image.height * .45))), Image.Resampling.LANCZOS)
    elif condition == 'rotate':
        image = image.rotate(-6 if plan['index'] % 2 else 6, Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    elif condition == 'shadow':
        ink = ImageOps.invert(image)
        shadow = Image.new('L', image.size, 0); shadow.paste(ink.filter(ImageFilter.GaussianBlur(1)), (1, 2))
        image = Image.fromarray(np.minimum(np.asarray(image), 255 - (np.asarray(shadow).astype(np.float32) * .2).astype(np.uint8)))
    elif condition == 'margin':
        image = ImageOps.expand(image, border=(7, image.height, 13, image.height // 2), fill=255)
    elif condition == 'blur': image = image.filter(ImageFilter.GaussianBlur(.5))
    elif condition == 'jpeg':
        stream = io.BytesIO(); image.save(stream, 'JPEG', quality=60); image = Image.open(io.BytesIO(stream.getvalue())).convert('L')
    elif condition != 'clean': raise ValueError('Unknown augmentation')
    return ImageOps.invert(image) if plan['light'] else image


def render():
    config = read(ROOT / 'bench/hundred.json')['data']; items = read(DATA / 'plans.json')
    fonts = read(ROOT / 'bench/fonts-100.json')['fonts']; browser = read(DATA / 'browser.json')
    if browser['dataConfig'] != config or browser['rawSha256'] != sha(DATA / 'browser.gray'): raise ValueError('Stale browser data')
    samples = []
    with (DATA / 'source.gray').open('wb') as stream:
        def append(base, plan, family, renderer, condition):
            image = transform(base, plan, condition)
            samples.append({**plan, 'family': family, 'renderer': renderer, 'condition': condition, 'offset': stream.tell(), 'width': image.width, 'height': image.height})
            stream.write(image.tobytes())
        for face in fonts[:100]:
            path = ROOT / f'.data/fonts/{face["id"]}.ttf'
            if sha(path) != face['instanceSha256']: raise ValueError('Font checksum mismatch')
            for plan in items:
                if plan['role'] != 'train': continue
                base = render_sample(path, plan['text'], plan['size'], plan['dpr'], 'dark')
                for condition in config['trainViews']: append(base, plan, face['id'], 'pillow', condition)
            print(f'Pillow {face["id"]}', flush=True)
        with (DATA / 'browser.gray').open('rb') as raw:
            for sample in browser['samples']:
                if sample['offset'] != raw.tell(): raise ValueError('Invalid offset')
                pixels = raw.read(sample['width'] * sample['height'])
                base = Image.frombytes('L', (sample['width'], sample['height']), pixels)
                for condition in config['trainViews'] if sample['role'] == 'train' else config['conditions']:
                    append(base, sample, sample['family'], 'chromium', condition)
            if raw.read(1): raise ValueError('Trailing pixels')
    write(DATA / 'source.json', {'samples': samples, 'dataConfig': config, 'rawSha256': sha(DATA / 'source.gray'),
          'fontsSha256': sha(ROOT / 'bench/fonts-100.json'), 'textsSha256': sha(ROOT / 'bench/hundred-texts.json'),
          'generatorSha256': sha(Path(__file__)), 'browserSha256': sha(DATA / 'browser.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['import', 'plans', 'render']); parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.command == 'import':
        if not args.source: parser.error('--source required')
        import_fonts(args.source.expanduser())
    elif args.command == 'plans': plans()
    else: render()
