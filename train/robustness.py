"""One target versus other fonts; disjoint texts and a Chromium development set."""
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from scripts.fixtures import render_sample, check_coverage
from fontTools.ttLib import TTFont
from train.model import Encoder, export_model, load_export

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/robustness'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def augment(image, plan):
    """White-background glyph image → bounded screenshot variations, no stretch."""
    image = image.convert('L')
    bbox = ImageOps.invert(image).point(lambda x: 255 if x > 8 else 0).getbbox()
    if bbox is None:
        raise ValueError('Empty source image')
    image = image.crop(bbox)
    fraction = plan.get('window', 1)
    position = plan.get('position', .5)
    scale = plan.get('scale', 1)
    pad = plan.get('pad', 0)
    if not 0 < fraction <= 1 or not 0 <= position <= 1 or not .25 <= scale <= 2 or not isinstance(pad, int) or pad < 0:
        raise ValueError('Invalid augmentation geometry')
    width = max(1, round(image.width * fraction))
    left = round((image.width - width) * position)
    image = image.crop((left, 0, left + width, image.height))
    if scale != 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    image = ImageOps.expand(image, pad, 255)
    angle = plan.get('angle', 0)
    if angle:
        image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    if plan.get('blur', 0):
        image = image.filter(ImageFilter.GaussianBlur(plan['blur']))
    if plan.get('quality'):
        buffer = io.BytesIO()
        image.save(buffer, 'JPEG', quality=plan['quality'])
        image = Image.open(io.BytesIO(buffer.getvalue())).convert('L')
    if plan.get('light', False):
        image = ImageOps.invert(image)
    return image.convert('RGBA')


def condition(name):
    plans = {
        'clean': {'pad': 8}, 'tight': {},
        'short': {'window': .45, 'position': .35}, 'small': {'scale': .5},
        'rotate-left': {'angle': -8}, 'rotate-right': {'angle': 8},
        'blur': {'blur': .65}, 'jpeg': {'quality': 55},
        'mixed': {'window': .6, 'position': .2, 'scale': .75, 'angle': 5, 'blur': .3, 'quality': 70},
    }
    return plans[name].copy()


def random_plan(rng, texts, index):
    return {'text': texts[index % len(texts)], 'size': rng.randint(14, 48), 'dpr': rng.choice([1, 2]),
            'window': rng.uniform(.4, 1), 'position': rng.random(), 'scale': rng.uniform(.6, 1.25),
            'pad': rng.choice([0, 0, 2, 8, 16]), 'angle': rng.choice([0, 0, rng.uniform(-8, 8)]),
            'blur': rng.choice([0, 0, .3, .6]), 'quality': rng.choice([None, None, 60, 85]),
            'light': bool(rng.randrange(2))}


def validate_samples(samples, target):
    pools = {role: set() for role in ['train', 'calibration', 'query']}
    splits = {}
    for sample in samples:
        role = sample['role']
        if role not in pools or sample['split'] not in ['train', 'unseen-dev']:
            raise ValueError('Unknown role or family split')
        if role != 'query' and sample['split'] != 'train':
            raise ValueError('Held-out family leaked into fitting or calibration')
        if splits.setdefault(sample['family'], sample['split']) != sample['split']:
            raise ValueError('Family crosses training/development boundary')
        pools[role].add(sample['text'])
    for a, b in [('train', 'calibration'), ('train', 'query'), ('calibration', 'query')]:
        if not pools[a] or not pools[b] or pools[a] & pools[b]:
            raise ValueError('Missing text pool or text leakage')
    for role in pools:
        labels = {s['family'] == target for s in samples if s['role'] == role}
        if labels != {False, True}:
            raise ValueError('Every role needs target and negative samples')


def render():
    config = read(ROOT / 'bench/robustness.json')
    pilot = read(ROOT / 'bench/pilot.json')
    fonts = read(ROOT / 'bench/fonts.json')['fonts']
    browser = read(ROOT / '.data/browser/samples.json')
    pillow = read(ROOT / '.data/pillow/samples.json')
    for source in [browser, pillow]:
        if source['fontsSha256'] != sha(ROOT / 'bench/fonts.json') or source['configSha256'] != sha(ROOT / 'bench/pilot.json'):
            raise ValueError('Stale rendering fixtures')
    training_fonts = [f for f in fonts if f['split'] == 'train']
    if config['target'] not in [f['id'] for f in training_fonts]:
        raise ValueError('Target must be a training family')
    rng = random.Random(config['seed'])
    plans = {role: [random_plan(rng, texts, i) for i in range(count)] for role, texts, count in [
        ('train', pilot['training']['texts'], config['samplesPerFamily']),
        ('calibration', config['calibrationTexts'], config['calibrationPerFamily'])]}
    DATA.mkdir(parents=True, exist_ok=True)
    samples = []
    with (DATA / 'source.rgba').open('wb') as stream:
        def append(image, meta):
            samples.append({**meta, 'width': image.width, 'height': image.height, 'offset': stream.tell()})
            stream.write(image.tobytes())
        for face in training_fonts:
            path = ROOT / f'.data/fonts/{face["id"]}.ttf'
            if sha(path) != face['instanceSha256']:
                raise ValueError(f'Font changed: {face["id"]}')
            with TTFont(path) as font:
                check_coverage(font, pilot['training']['texts'] + config['calibrationTexts'])
            # Identical text/augmentation plans across families provide matched negatives.
            for role, items in plans.items():
                for index, plan in enumerate(items):
                    base = render_sample(path, plan['text'], plan['size'], plan['dpr'], 'dark')
                    append(augment(base, plan), {'family': face['id'], 'split': 'train', 'role': role,
                                               'renderer': 'pillow', 'text': plan['text'], 'plan': index, 'augmentation': plan})
        query_sources = browser['samples'] + [s for s in pillow['samples'] if s['role'] == 'query']
        for source in query_sources:
            raw = (ROOT / source['raw']).read_bytes()
            if len(raw) != source['width'] * source['height'] * 4:
                raise ValueError('Invalid Chromium RGBA length')
            base = Image.frombytes('RGBA', (source['width'], source['height']), raw).convert('L')
            if source['polarity'] == 'light':
                base = ImageOps.invert(base)
            for name in config['conditions']:
                plan = {**condition(name), 'light': source['polarity'] == 'light'}
                append(augment(base, plan), {**{k: source[k] for k in ['family', 'split', 'text', 'size', 'dpr', 'polarity']},
                                           'role': 'query', 'renderer': source['renderer'], 'condition': name, 'augmentation': plan})
    validate_samples(samples, config['target'])
    write(DATA / 'samples.json', {'samples': samples, 'configSha256': sha(ROOT / 'bench/robustness.json'),
          'fontsSha256': sha(ROOT / 'bench/fonts.json'), 'generatorSha256': sha(Path(__file__)),
          'encoderSha256': sha(ROOT / 'train/model.py'), 'rawSha256': sha(DATA / 'source.rgba'),
          'browserManifestSha256': sha(ROOT / '.data/browser/samples.json'), 'pillowManifestSha256': sha(ROOT / '.data/pillow/samples.json')})
    print(f'Rendered {len(samples)} images: {sum(s["role"] == "train" for s in samples)} training; shared plans across families.', flush=True)


def threshold_at_fpr(probabilities, labels, fpr):
    probabilities, labels = np.asarray(probabilities), np.asarray(labels, dtype=bool)
    if probabilities.shape != labels.shape or probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid calibration probabilities')
    if not 0 <= fpr < 1 or not labels.any() or labels.all():
        raise ValueError('Calibration needs both classes and FPR in [0, 1)')
    negatives = np.sort(probabilities[~labels])
    allowed = math.floor(fpr * len(negatives))
    # Strictly above the boundary handles ties without exceeding the FPR budget.
    return float(np.nextafter(np.float32(negatives[-allowed - 1]), np.float32(math.inf)))


def metrics(probabilities, labels, threshold):
    labels = np.asarray(labels, dtype=bool)
    accepted = np.asarray(probabilities) >= threshold
    positives, negatives = int(labels.sum()), int((~labels).sum())
    tp, fp = int((accepted & labels).sum()), int((accepted & ~labels).sum())
    return {'positives': positives, 'negatives': negatives, 'truePositives': tp, 'falsePositives': fp,
            'recall': tp / positives if positives else None, 'falsePositiveRate': fp / negatives if negatives else None,
            'precision': tp / (tp + fp) if tp + fp else None}


def train(name):
    config = read(ROOT / 'bench/robustness.json')
    settings = config['inputs'][name]
    folder = DATA / name
    manifest = read(folder / 'prepared.json')
    for key, path in [('configSha256', ROOT / 'bench/robustness.json'), ('fontsSha256', ROOT / 'bench/fonts.json'),
                      ('generatorSha256', Path(__file__)), ('preparationSha256', ROOT / 'src/prepare.mjs'),
                      ('encoderSha256', ROOT / 'train/model.py'),
                      ('tensorSha256', folder / 'prepared.f32')]:
        if sha(path) != manifest[key]:
            raise ValueError(f'Stale or corrupt experiment: {path}')
    if any(manifest.get(key) != value for key, value in settings.items()):
        raise ValueError('Prepared geometry mismatch')
    samples = manifest['samples']
    validate_samples(samples, config['target'])
    shape = (len(samples), 1, settings['height'], settings['width'])
    if (folder / 'prepared.f32').stat().st_size != math.prod(shape) * 4:
        raise ValueError('Prepared tensor length mismatch')
    array = np.fromfile(folder / 'prepared.f32', dtype='<f4').reshape(shape)
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError('Invalid prepared pixels')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config['seed'])
    pixels = torch.from_numpy(array)
    labels = torch.tensor([int(s['family'] == config['target']) for s in samples])
    by_family = {}
    for i, sample in enumerate(samples):
        if sample['role'] == 'train':
            by_family.setdefault(sample['family'], {})[sample['plan']] = i
    target = by_family.pop(config['target'])
    plans = sorted(target)
    if any(set(pool) != set(plans) for pool in by_family.values()):
        raise ValueError('Missing matched negative plans')
    negatives = list(by_family.values())
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
        if (step + 1) % 200 == 0:
            print(f'{name} verification {step + 1}/{config["steps"]}: loss {np.mean(losses[-50:]):.4f}', flush=True)
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        probabilities = torch.cat([model(batch).softmax(1)[:, 1] for batch in pixels.split(128)]).numpy()
    labels = labels.numpy().astype(bool)
    calibration = np.array([s['role'] == 'calibration' for s in samples])
    threshold = threshold_at_fpr(probabilities[calibration], labels[calibration], config['calibrationFpr'])
    query = np.array([s['role'] == 'query' for s in samples])
    groups = {}
    renderers = {}
    for renderer in ['pillow', 'chromium']:
        rendered = query & np.array([s['renderer'] == renderer for s in samples])
        renderers[renderer] = metrics(probabilities[rendered], labels[rendered], threshold)
        for variation in config['conditions']:
            selected = rendered & np.array([s.get('condition') == variation for s in samples])
            groups[f'{renderer}/{variation}'] = metrics(probabilities[selected], labels[selected], threshold)
    per_family = {}
    for family in sorted({s['family'] for s in samples if s['role'] == 'query'}):
        selected = query & np.array([s['family'] == family for s in samples])
        per_family[family] = metrics(probabilities[selected], labels[selected], threshold)
    artifact = {**export_model(model[0]), 'task': 'binary-font-verification', 'target': config['target'], 'threshold': threshold,
                'head': {name: value.detach().tolist() for name, value in model[1].state_dict().items()},
                'preparation': {**settings, 'sha256': manifest['preparationSha256']}}
    write(folder / 'model.json', artifact)
    saved = read(folder / 'model.json')
    restored = nn.Sequential(load_export(saved), nn.Linear(32, 2))
    restored[1].load_state_dict({name: torch.tensor(value) for name, value in saved['head'].items()})
    with torch.no_grad():
        parity = float((restored(pixels[:8]) - model(pixels[:8])).abs().max())
    if parity > 1e-6:
        raise ValueError('Binary model export mismatch')
    torch.save(model.state_dict(), folder / 'model.pt')
    report = {'description': 'Inter versus 13 training negatives, with 6 additional unseen negative families at development evaluation. Synthetic only; not a 20-family retrieval model.',
              'target': config['target'], 'seed': config['seed'], 'steps': config['steps'], 'seconds': elapsed,
              'parameters': sum(p.numel() for p in model.parameters()), 'input': settings,
              'samples': {role: sum(s['role'] == role for s in samples) for role in ['train', 'calibration', 'query']},
              'lossFirst50': float(np.mean(losses[:50])), 'lossLast50': float(np.mean(losses[-50:])),
              'calibrationFprBudget': config['calibrationFpr'], 'threshold': threshold,
              'calibration': metrics(probabilities[calibration], labels[calibration], threshold),
              'training': metrics(probabilities[~(calibration | query)], labels[~(calibration | query)], threshold),
              'development': metrics(probabilities[query], labels[query], threshold), 'renderers': renderers, 'conditions': groups, 'families': per_family,
              'manifestSha256': sha(folder / 'prepared.json'), 'modelSha256': sha(folder / 'model.json'),
              'exportMaxAbsoluteError': parity, 'torch': torch.__version__}
    write(ROOT / f'bench/robustness-{name}.json', report)
    print(json.dumps({'seconds': elapsed, 'calibration': report['calibration'], 'development': report['development'], 'renderers': renderers}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'render':
        render()
    elif sys.argv[1] == 'train':
        train(sys.argv[2])
    else:
        raise ValueError('Unknown experiment command')
