"""Ten-way classification with held-out words and image-level multi-window evaluation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from train.robustness import sha, read, write
from train.ten_model import Classifier, export

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/ten'


def load_data():
    config = read(ROOT / 'bench/ten.json')
    manifest = read(DATA / 'prepared.json')
    if manifest['dataConfig'] != config['data']:
        raise ValueError('Data configuration changed')
    for key, path in [('fontsSha256', ROOT / 'bench/fonts.json'), ('textsSha256', ROOT / 'bench/ten-texts.json'),
                      ('generatorSha256', ROOT / 'train/ten_data.py'), ('augmentSha256', ROOT / 'train/robustness.py'),
                      ('inputSha256', ROOT / 'src/input.mjs'), ('preparationSha256', ROOT / 'src/prepare.mjs'),
                      ('runnerSha256', ROOT / 'scripts/ten.mjs'), ('tensorSha256', DATA / 'prepared.u8')]:
        if manifest[key] != sha(path):
            raise ValueError(f'Stale data: {path}')
    samples, windows = manifest['samples'], manifest['windows']
    pools = {role: set() for role in ['train', 'validation', 'test']}
    for s in samples:
        if s['family'] not in config['data']['fonts'] or s['role'] not in pools:
            raise ValueError('Unknown family or split')
        pools[s['role']].update(s['text'].lower().split())
    if any(not a or not b or a & b for i, a in enumerate(pools.values()) for b in list(pools.values())[i + 1:]):
        raise ValueError('Empty or overlapping word pools')
    offset = 0
    coverage = set()
    for w in windows:
        if not isinstance(w, dict) or any(type(w.get(k)) is not int for k in ['source', 'offset', 'width', 'height']) or w['offset'] != offset or not 1 <= w['width'] <= config['data']['input']['width'] or not 1 <= w['height'] <= config['data']['input']['height'] or not 0 <= w['source'] < len(samples):
            raise ValueError('Invalid prepared window')
        offset += w['width'] * w['height']; coverage.add(w['source'])
    if coverage != set(range(len(samples))) or (DATA / 'prepared.u8').stat().st_size != offset:
        raise ValueError('Missing windows or invalid tensor length')
    pixels = np.memmap(DATA / 'prepared.u8', dtype=np.uint8, mode='r')
    return config, manifest, pixels


def batch(pixels, windows, indexes):
    sizes = np.array([[windows[i]['height'], windows[i]['width']] for i in indexes], dtype=np.int64)
    array = np.ones((len(indexes), 1, sizes[:, 0].max(), sizes[:, 1].max()), dtype=np.float32)
    for n, i in enumerate(indexes):
        w = windows[i]
        array[n, 0, :w['height'], :w['width']] = pixels[w['offset']:w['offset'] + w['width'] * w['height']].reshape(w['height'], w['width']) / np.float32(255)
    return torch.from_numpy(array), torch.from_numpy(sizes)


def evaluate(model, manifest, pixels, role):
    samples, windows = manifest['samples'], manifest['windows']
    fonts = manifest['dataConfig']['fonts']
    indexes = [i for i, w in enumerate(windows) if samples[w['source']]['role'] == role]
    totals, counts = {}, {}
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(indexes), 96):
            chosen = indexes[start:start + 96]
            scores = model(*(t.to(device) for t in batch(pixels, windows, chosen))).softmax(1).cpu().numpy()
            for i, score in zip(chosen, scores):
                source = windows[i]['source']
                totals[source] = totals.get(source, np.zeros(len(fonts))) + score
                counts[source] = counts.get(source, 0) + 1
    groups = {}
    confusion = np.zeros((len(fonts), len(fonts)), dtype=np.int64)
    predictions = []
    for source, score in totals.items():
        s = samples[source]; score /= counts[source]
        label, predicted = fonts.index(s['family']), int(score.argmax())
        confusion[label, predicted] += 1
        for key in ['all', s['renderer'], f'{s["renderer"]}/{s["condition"]}', s['family']]:
            entry = groups.setdefault(key, {'count': 0, 'correct': 0, 'top3': 0})
            entry['count'] += 1; entry['correct'] += int(label == predicted)
            entry['top3'] += int(label in score.argsort()[-3:])
        predictions.append({'source': source, 'label': label, 'predicted': predicted, 'scores': score.tolist()})
    for g in groups.values():
        g['accuracy'] = g['correct'] / g['count']; g['top3Accuracy'] = g['top3'] / g['count']
    return {'groups': groups, 'confusion': confusion.tolist()}, predictions


def train(steps=None, resume=False):
    config, manifest, pixels = load_data()
    settings = config['training']
    steps = settings['steps'] if steps is None else steps
    if type(steps) is not int or steps < 1:
        raise ValueError('Training steps must be a positive integer')
    torch.set_num_threads(settings['threads']); torch.manual_seed(settings['seed'])
    device = settings.get('device', 'cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; use CPU explicitly or run with GPU access')
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(settings['seed'])
    fonts = config['data']['fonts']; samples, windows = manifest['samples'], manifest['windows']
    families = [[[i for i, w in enumerate(windows) if samples[w['source']]['role'] == 'train' and samples[w['source']]['family'] == f and samples[w['source']]['renderer'] == renderer] for renderer in ['pillow', 'chromium']] for f in fonts]
    if any(not group for family in families for group in family):
        raise ValueError('Missing training family')
    model = Classifier(len(fonts), dilations=settings.get('dilations'))
    parent = read(ROOT / 'bench/ten-training.json') if resume else None
    if parent:
        previous = read(DATA / 'model.json')
        if parent['modelSha256'] != sha(DATA / 'model.json') or previous['fonts'] != fonts or previous['preparation']['sha256'] != manifest['inputSha256'] or parent['architectureSha256'] != sha(ROOT / 'train/ten_model.py'):
            raise ValueError('Cannot resume incompatible model')
        model.load_state_dict(torch.load(DATA / 'best.pt', weights_only=True, map_location='cpu')['state'])
        if export(model, fonts, previous['preparation'])[0] != previous:
            raise ValueError('Resume checkpoint does not match exported model')
    model.to(device)
    rate = settings['learningRate'] / 4 if resume else settings['learningRate']
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=rate / 20)
    best, history, losses = -1, [], []
    if resume:
        initial, _ = evaluate(model, manifest, pixels, 'validation')
        best = initial['groups']['all']['accuracy']
    start = time.perf_counter()
    for step in range(steps):
        model.train()
        labels = rng.integers(len(fonts), size=settings['batch'])
        indexes = [int(rng.choice(families[label][rng.integers(2)])) for label in labels]
        x, sizes = (t.to(device) for t in batch(pixels, windows, indexes))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, sizes), torch.from_numpy(labels).to(device), label_smoothing=.03)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward(); optimizer.step(); scheduler.step(); losses.append(loss.item())
        if (step + 1) % 250 == 0:
            print(f'ten {step + 1}/{steps}: loss {np.mean(losses[-100:]):.4f}; {time.perf_counter() - start:.0f}s', flush=True)
        if (step + 1) % 1000 == 0 or step == steps - 1:
            report, _ = evaluate(model, manifest, pixels, 'validation')
            score = report['groups']['all']['accuracy']
            history.append({'step': step + 1, 'validation': report})
            print(f'Validation {step + 1}: {score:.3%}; Chromium {report["groups"]["chromium"]["accuracy"]:.3%}', flush=True)
            if score > best:
                best = score
                torch.save({'state': model.state_dict(), 'step': step + 1}, DATA / 'best.pt')
    elapsed = time.perf_counter() - start
    model.cpu().load_state_dict(torch.load(DATA / 'best.pt', weights_only=True, map_location='cpu')['state'])
    artifact, quantized = export(model, fonts, {**config['data']['input'], 'sha256': sha(ROOT / 'src/input.mjs')})
    floating, _ = evaluate(model, manifest, pixels, 'validation')
    quant, _ = evaluate(quantized, manifest, pixels, 'validation')
    if quant['groups']['all']['accuracy'] < floating['groups']['all']['accuracy'] - .01:
        raise ValueError('Int8 loses more than one accuracy point')
    artifact['preparation']['normalizerSha256'] = sha(ROOT / 'src/prepare.mjs')
    write(DATA / 'model.json', artifact)
    summary = {'task': 'ten-font-classification', 'fonts': fonts, 'training': settings, 'steps': steps, 'seconds': elapsed,
               'selectedStep': torch.load(DATA / 'best.pt', weights_only=True, map_location='cpu')['step'],
               'parameters': sum(p.numel() for p in quantized.parameters()), 'validationFloat': floating,
               'validationInt8': quant, 'history': history, 'manifestSha256': sha(DATA / 'prepared.json'),
               'modelSha256': sha(DATA / 'model.json'), 'trainerSha256': sha(Path(__file__)),
               'architectureSha256': sha(ROOT / 'train/ten_model.py'), 'torch': torch.__version__,
               'sampling': 'uniform family and renderer', 'learningRate': rate,
               'parentModelSha256': parent['modelSha256'] if parent else None,
               'parentManifestSha256': parent['manifestSha256'] if parent else None,
               'dataConfig': config['data'],
               'totalSeconds': elapsed + (parent.get('totalSeconds', parent['seconds']) if parent else 0)}
    write(ROOT / 'bench/ten-training.json', summary)
    print(json.dumps({k: summary[k] for k in ['seconds', 'parameters', 'selectedStep', 'modelSha256']}, indent=2), flush=True)


def final_evaluation():
    config, manifest, pixels = load_data()
    report = read(ROOT / 'bench/ten-training.json')
    if report['manifestSha256'] != sha(DATA / 'prepared.json') or report['modelSha256'] != sha(DATA / 'model.json'):
        raise ValueError('Stale trained model')
    torch.set_num_threads(config['training']['threads'])
    model = Classifier(len(config['data']['fonts']), dilations=read(DATA / 'model.json').get('dilations'))
    model.load_state_dict(torch.load(DATA / 'best.pt', weights_only=True, map_location='cpu')['state'])
    artifact, quantized = export(model, config['data']['fonts'], read(DATA / 'model.json')['preparation'])
    if artifact != read(DATA / 'model.json'):
        raise ValueError('Checkpoint does not match deployed weights')
    metrics, predictions = evaluate(quantized, manifest, pixels, 'test')
    write(ROOT / 'bench/ten-test.json', {**metrics, 'fonts': config['data']['fonts'], 'modelSha256': report['modelSha256'],
          'manifestSha256': report['manifestSha256'], 'description': 'Frozen Chromium word-disjoint synthetic test. No real-image accuracy claim.'})
    write(DATA / 'test-predictions.json', predictions)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'evaluate']); parser.add_argument('--steps', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'train':
        train(args.steps, args.resume)
    else:
        final_evaluation()
