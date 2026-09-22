"""Controlled input-policy study. No OCR, descriptor, or demo artifact changes."""
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from fontTools.ttLib import TTFont
import torch
from torch import nn
from torch.nn import functional as F

from scripts.fixtures import render_sample, check_coverage
from train.model import Encoder, export_model, load_export
from train.robustness import augment, condition, sha, read, write, threshold_at_fpr, metrics, validate_samples

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/focus'


def text_pools(counts, seed):
    """Split unique underlying strings before any font rendering or augmentation."""
    if set(counts) != {'train', 'calibration', 'query'} or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('Expected three nonempty text pools')
    rng = random.Random(seed)
    seen, pools = set(), {}
    for role, count in counts.items():
        pool = []
        while len(pool) < count:
            # Random letters prevent memorizing 24 sentences; spaces/case vary independently of font.
            words = [''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=rng.randint(3, 8)))
                     for _ in range(rng.randint(1, 4))]
            text = ' '.join(words)
            style = rng.randrange(4)
            text = text.upper() if style == 0 else text.title() if style == 1 else text
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            pool.append(text)
        pools[role] = pool
    return pools


def source_plan(index, seed):
    rng = random.Random(seed + index)
    return {'size': rng.choice([20, 28, 36, 48]), 'dpr': 1, 'pad': rng.choice([2, 8]),
            'light': bool(rng.randrange(2))}


def window(image, settings):
    """Take one central height-preserving window, with no letter recognition."""
    if not isinstance(settings, dict):
        raise ValueError('Invalid window geometry')
    width, height, padding = (settings.get(key) for key in ['width', 'height', 'padding'])
    if (any(type(value) is not int for value in [width, height, padding])
            or not 4 <= width <= 1024 or not 4 <= height <= 1024
            or not 0 <= padding <= (min(width, height) - 1) // 2):
        raise ValueError('Invalid window geometry')
    aspect = (width - 2 * padding) / (height - 2 * padding)
    gray = image.convert('L')
    # Source inputs are known synthetic black-on-white renderings, not arbitrary photos.
    box = ImageOps.invert(gray).point(lambda p: 255 if p > 8 else 0).getbbox()
    if box is None:
        raise ValueError('Empty source image')
    image = image.crop(box)
    width = min(image.width, max(1, round(image.height * aspect)))
    left = (image.width - width) // 2
    return image.crop((left, 0, left + width, image.height))


def render():
    config = read(ROOT / 'bench/focus.json')
    fonts = read(ROOT / 'bench/fonts.json')['fonts']
    pools = text_pools(config['texts'], config['seed'])
    browser = read(DATA / 'browser.json')
    if browser['configSha256'] != sha(ROOT / 'bench/focus.json') or browser['fontsSha256'] != sha(ROOT / 'bench/fonts.json') or browser['rawSha256'] != sha(DATA / 'browser.rgba'):
        raise ValueError('Stale browser fixtures')
    DATA.mkdir(parents=True, exist_ok=True)
    samples = []
    with (DATA / 'source.rgba').open('wb') as stream:
        for face in fonts:
            path = ROOT / f'.data/fonts/{face["id"]}.ttf'
            if sha(path) != face['instanceSha256']:
                raise ValueError('Font checksum mismatch')
            with TTFont(path) as font:
                check_coverage(font, [text for pool in pools.values() for text in pool])
            for role, texts in pools.items():
                if role != 'query' and face['split'] != 'train':
                    continue
                for index, text in enumerate(texts):
                    plan = source_plan(index, config['seed'])
                    base = render_sample(path, text, plan['size'], 1, 'dark')
                    # Train/calibrate on clean crops first; distortions are a separate stress test.
                    conditions = config['conditions'] if role == 'query' else ['clean']
                    for name in conditions:
                        for policy in config['policies']:
                            image = base if policy == 'line' else window(base, config['input'])
                            image = augment(image, {**condition(name), 'light': plan['light']})
                            samples.append({'family': face['id'], 'split': face['split'], 'role': role,
                                            'text': text, 'plan': index, 'renderer': 'pillow', 'condition': name,
                                            'policy': policy, 'width': image.width, 'height': image.height,
                                            'offset': stream.tell()})
                            stream.write(image.tobytes())
            print(f'Rendered {face["id"]}', flush=True)
        with (DATA / 'browser.rgba').open('rb') as source:
            offset = 0
            for item in browser['samples']:
                if item['offset'] != offset:
                    raise ValueError('Invalid browser offset')
                raw = source.read(item['width'] * item['height'] * 4)
                if len(raw) != item['width'] * item['height'] * 4:
                    raise ValueError('Truncated browser image')
                offset += len(raw)
                base = Image.frombytes('RGBA', (item['width'], item['height']), raw)
                for name in config['conditions']:
                    for policy in config['policies']:
                        image = base if policy == 'line' else window(base, config['input'])
                        image = augment(image, {**condition(name), 'light': source_plan(item['plan'], config['seed'])['light']})
                        samples.append({**item, 'role': 'query', 'renderer': 'chromium', 'condition': name,
                                        'policy': policy, 'width': image.width, 'height': image.height, 'offset': stream.tell()})
                        stream.write(image.tobytes())
            if source.read(1):
                raise ValueError('Trailing browser bytes')
    write(DATA / 'samples.json', {'samples': samples, 'pools': pools,
          'configSha256': sha(ROOT / 'bench/focus.json'), 'fontsSha256': sha(ROOT / 'bench/fonts.json'),
          'generatorSha256': sha(Path(__file__)), 'encoderSha256': sha(ROOT / 'train/model.py'),
          'augmentationSha256': sha(ROOT / 'train/robustness.py'), 'rendererSha256': sha(ROOT / 'scripts/fixtures.py'),
          'browserSha256': sha(DATA / 'browser.json'), 'rawSha256': sha(DATA / 'source.rgba')})


def load_data():
    config = read(ROOT / 'bench/focus.json')
    manifest = read(DATA / 'prepared.json')
    for key, path in [('configSha256', ROOT / 'bench/focus.json'), ('fontsSha256', ROOT / 'bench/fonts.json'),
                      ('generatorSha256', Path(__file__)), ('encoderSha256', ROOT / 'train/model.py'),
                      ('augmentationSha256', ROOT / 'train/robustness.py'), ('rendererSha256', ROOT / 'scripts/fixtures.py'),
                      ('preparationSha256', ROOT / 'src/prepare.mjs'), ('runnerSha256', ROOT / 'scripts/focus.mjs'),
                      ('tensorSha256', DATA / 'prepared.f32')]:
        if manifest[key] != sha(path):
            raise ValueError(f'Stale input: {path}')
    if manifest['input'] != config['input']:
        raise ValueError('Input geometry mismatch')
    samples = manifest['samples']
    validate_samples(samples, config['target'])
    shape = (len(samples), 1, config['input']['height'], config['input']['width'])
    if (DATA / 'prepared.f32').stat().st_size != math.prod(shape) * 4:
        raise ValueError('Tensor length mismatch')
    array = np.memmap(DATA / 'prepared.f32', mode='r', dtype='<f4', shape=shape)
    for batch in np.array_split(array, max(1, len(array) // 128)):
        if not np.isfinite(batch).all() or np.any((batch < 0) | (batch > 1)):
            raise ValueError('Invalid tensor pixels')
    return config, manifest, array


def train(policy):
    config, manifest, array = load_data()
    if policy not in config['policies']:
        raise ValueError('Unknown policy')
    selected = [i for i, s in enumerate(manifest['samples']) if s['policy'] == policy]
    samples = [manifest['samples'][i] for i in selected]
    pixels = torch.from_numpy(array[selected])
    labels = torch.tensor([int(s['family'] == config['target']) for s in samples])
    by_family = {}
    for i, sample in enumerate(samples):
        if sample['role'] == 'train':
            by_family.setdefault(sample['family'], {})[sample['plan']] = i
    target = by_family.pop(config['target'])
    plans = sorted(target)
    if any(set(pool) != set(plans) for pool in by_family.values()):
        raise ValueError('Missing matched negatives')
    negatives = list(by_family.values())
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config['seed'])
    model = nn.Sequential(Encoder(), nn.Linear(32, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learningRate'])
    losses = []
    start = time.perf_counter()
    for step in range(config['steps']):
        chosen = rng.choice(plans, config['batchPairs'])
        batch = [target[p] for p in chosen] + [negatives[rng.integers(len(negatives))][p] for p in chosen]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(pixels[batch]), labels[batch])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 400 == 0:
            print(f'{policy} {step + 1}: loss {np.mean(losses[-100:]):.4f}', flush=True)
    seconds = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        probabilities = torch.cat([model(batch).softmax(1)[:, 1] for batch in pixels.split(128)]).numpy()
    labels = labels.numpy().astype(bool)
    cal = np.array([s['role'] == 'calibration' for s in samples])
    threshold = threshold_at_fpr(probabilities[cal], labels[cal], config['calibrationFpr'])
    groups = {}
    for renderer in ['pillow', 'chromium']:
        for name in config['conditions']:
            mask = np.array([s['role'] == 'query' and s['renderer'] == renderer and s['condition'] == name for s in samples])
            groups[f'{renderer}/{name}'] = metrics(probabilities[mask], labels[mask], threshold)
    artifact = {**export_model(model[0]), 'task': 'binary-font-verification', 'target': config['target'],
                'policy': policy, 'threshold': threshold, 'preparation': config['input'],
                'head': {name: value.detach().tolist() for name, value in model[1].state_dict().items()}}
    write(DATA / f'{policy}.json', artifact)
    restored = nn.Sequential(load_export(artifact), nn.Linear(32, 2))
    restored[1].load_state_dict({name: torch.tensor(value) for name, value in artifact['head'].items()})
    with torch.no_grad():
        error = float((restored(pixels[:8]) - model(pixels[:8])).abs().max())
    if error > 1e-6:
        raise ValueError('Export mismatch')
    report = {'policy': policy, 'seconds': seconds, 'steps': config['steps'], 'seed': config['seed'],
              'parameters': sum(p.numel() for p in model.parameters()), 'threshold': threshold,
              'lossFirst100': float(np.mean(losses[:100])), 'lossLast100': float(np.mean(losses[-100:])),
              'calibration': metrics(probabilities[cal], labels[cal], threshold), 'conditions': groups,
              'training': metrics(probabilities[np.array([s['role'] == 'train' for s in samples])], labels[np.array([s['role'] == 'train' for s in samples])], threshold),
              'manifestSha256': sha(DATA / 'prepared.json'), 'modelSha256': sha(DATA / f'{policy}.json'),
              'exportMaxAbsoluteError': error, 'torch': torch.__version__}
    write(ROOT / f'bench/focus-{policy}.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'plans':
        config = read(ROOT / 'bench/focus.json')
        DATA.mkdir(parents=True, exist_ok=True)
        write(DATA / 'plans.json', [{'text': text, 'plan': i, **source_plan(i, config['seed'])}
                                  for i, text in enumerate(text_pools(config['texts'], config['seed'])['query'])])
    elif sys.argv[1] == 'render':
        render()
    elif sys.argv[1] == 'train':
        train(sys.argv[2])
    else:
        raise ValueError('Unknown command')
