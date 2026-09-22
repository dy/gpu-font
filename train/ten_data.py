"""Pinned natural-word pools, labeled synthetic views; no final-test training access."""
import json
from pathlib import Path
import random
import re
import sys

from PIL import Image, ImageOps
from fontTools.ttLib import TTFont
from scripts.fixtures import render_sample, check_coverage
from train.robustness import augment, sha, read, write

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/ten'


def plans():
    config = read(ROOT / 'bench/ten.json')['data']
    path = ROOT / 'bench/ten-texts.json'
    if path.exists():
        pools = read(path)
    else:
        words = sorted({w.lower() for w in Path('/usr/share/dict/words').read_text().splitlines() if re.fullmatch('[a-zA-Z]{3,9}', w)})
        rng = random.Random(config['seed'])
        rng.shuffle(words)
        pools = {}
        offset = 0
        for role, count in config['texts'].items():
            bank = words[offset:offset + count * 4]
            offset += len(bank)
            texts = []
            for i in range(count):
                text = ' '.join(bank[i * 4:i * 4 + (1 if i % 4 == 0 else rng.randint(2, 4))])
                texts.append(text.upper() if i % 5 == 0 else text.title() if i % 3 == 0 else text)
            pools[role] = texts
        write(path, pools)
    word_sets = [{word.lower() for text in pool for word in text.split()} for pool in pools.values()]
    if any(a & b for i, a in enumerate(word_sets) for b in word_sets[i + 1:]):
        raise ValueError('Word leakage between splits')
    if any(len(pools[role]) != count for role, count in config['texts'].items()):
        raise ValueError('Text count mismatch')
    rng = random.Random(config['seed'])
    result = []
    for role, texts in pools.items():
        for index, text in enumerate(texts):
            result.append({'role': role, 'text': text, 'index': index, 'size': rng.choice([20, 28, 36, 48, 64]),
                           'dpr': rng.choice([1, 1, 2]), 'angle': rng.uniform(-6, 6), 'position': rng.random(),
                           'light': bool(rng.randrange(2))})
    DATA.mkdir(parents=True, exist_ok=True)
    write(DATA / 'plans.json', result)
    return result


def transform(base, plan, name):
    options = {'clean': {'pad': 4}, 'crop': {'window': .5, 'position': plan['position']},
               'small': {'scale': .6}, 'rotate': {'angle': -6 if plan['index'] % 2 else 6},
               'blur': {'blur': .45}, 'jpeg': {'quality': 65}}
    if name == 'train':
        rng = random.Random(plan['index'] + 481)
        name = rng.choice(['clean', 'clean', 'crop', 'small', 'rotate', 'blur', 'jpeg'])
    return augment(base, {**options[name], 'light': plan['light']}).convert('L')


def render():
    config = read(ROOT / 'bench/ten.json')['data']
    items = read(DATA / 'plans.json')
    fonts = {f['id']: f for f in read(ROOT / 'bench/fonts.json')['fonts']}
    browser = read(DATA / 'browser.json')
    if browser['dataConfig'] != config or browser['rawSha256'] != sha(DATA / 'browser.gray'):
        raise ValueError('Stale browser data')
    samples = []
    with (DATA / 'source.gray').open('wb') as stream:
        def append(image, meta):
            samples.append({**meta, 'width': image.width, 'height': image.height, 'offset': stream.tell()})
            stream.write(image.tobytes())
        for family in config['fonts']:
            path = ROOT / f'.data/fonts/{family}.ttf'
            if sha(path) != fonts[family]['instanceSha256']:
                raise ValueError('Font checksum mismatch')
            with TTFont(path) as face:
                check_coverage(face, [p['text'] for p in items])
            for plan in items:
                if plan['role'] == 'test':
                    continue
                base = render_sample(path, plan['text'], plan['size'], plan['dpr'], 'dark')
                for condition in config.get('trainViews', ['train']) if plan['role'] == 'train' else config['conditions']:
                    append(transform(base, plan, condition), {**plan, 'family': family, 'condition': condition, 'renderer': 'pillow'})
            print(f'Rendered {family}', flush=True)
        with (DATA / 'browser.gray').open('rb') as raw:
            offset = 0
            for sample in browser['samples']:
                if sample['offset'] != offset:
                    raise ValueError('Invalid browser offsets')
                length = sample['width'] * sample['height']
                pixels = raw.read(length)
                if len(pixels) != length:
                    raise ValueError('Truncated browser pixels')
                offset += length
                base = Image.frombytes('L', (sample['width'], sample['height']), pixels)
                for condition in config.get('trainViews', ['train']) if sample['role'] == 'train' else config['conditions']:
                    append(transform(base, sample, condition), {**sample, 'condition': condition, 'renderer': 'chromium'})
            if raw.read(1):
                raise ValueError('Trailing browser pixels')
    write(DATA / 'source.json', {'samples': samples, 'dataConfig': config, 'rawSha256': sha(DATA / 'source.gray'),
          'fontsSha256': sha(ROOT / 'bench/fonts.json'), 'textsSha256': sha(ROOT / 'bench/ten-texts.json'),
          'generatorSha256': sha(Path(__file__)), 'augmentSha256': sha(ROOT / 'train/robustness.py'),
          'browserSha256': sha(DATA / 'browser.json')})


if __name__ == '__main__':
    if sys.argv[1] == 'plans':
        plans()
    elif sys.argv[1] == 'render':
        render()
    else:
        raise ValueError('Unknown command')
