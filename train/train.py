"""Bounded 20-family development experiment, not a release model."""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from .model import Encoder, contrastive_loss, export_model, load_export
else:
    from model import Encoder, contrastive_loss, export_model, load_export

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_samples(samples):
    if not samples:
        raise ValueError('Empty dataset')
    texts = {role: set() for role in ('train', 'prototype', 'query')}
    splits = {}
    for sample in samples:
        if sample['role'] not in texts or sample['split'] not in ('train', 'unseen-dev'):
            raise ValueError('Unknown role or split')
        if sample['role'] == 'train' and sample['split'] != 'train':
            raise ValueError('Unseen family leaked into training')
        if sample['family'] in splits and splits[sample['family']] != sample['split']:
            raise ValueError('Family crosses training/development boundary')
        splits[sample['family']] = sample['split']
        texts[sample['role']].add(sample['text'])
    if not all(texts.values()) or any(texts[a] & texts[b] for a, b in [('train', 'query'), ('train', 'prototype'), ('prototype', 'query')]):
        raise ValueError('Missing text pool or text leakage between roles')
    prototypes = {s['family'] for s in samples if s['role'] == 'prototype'}
    if any(s['family'] not in prototypes for s in samples if s['role'] == 'query'):
        raise ValueError('Query family missing from catalog')


def evaluate(vectors, samples):
    families = sorted({s['family'] for s in samples if s['role'] == 'prototype'})
    prototypes = torch.stack([vectors[[i for i, s in enumerate(samples) if s['role'] == 'prototype' and s['family'] == family]].mean(0) for family in families])
    prototypes = F.normalize(prototypes, dim=1)
    groups = {}
    for i, sample in enumerate(samples):
        if sample['role'] != 'query':
            continue
        # Stable sorting on sorted family IDs gives deterministic ties.
        order = torch.argsort(vectors[i] @ prototypes.T, descending=True, stable=True).tolist()
        matches = [families[j] for j in order]
        group = groups.setdefault(f"{sample['renderer']}/{sample['split']}", {})
        counts = group.setdefault(sample['family'], {'queries': 0, 'top1': 0, 'top5': 0})
        counts['queries'] += 1
        counts['top1'] += matches[0] == sample['family']
        counts['top5'] += sample['family'] in matches[:5]
    result = {}
    for key, families_stats in groups.items():
        for counts in families_stats.values():
            for k in ['top1', 'top5']:
                counts[k] /= counts['queries']
        result[key] = {'queries': sum(c['queries'] for c in families_stats.values()),
                       'top1': sum(c['top1'] for c in families_stats.values()) / len(families_stats),
                       'top5': sum(c['top5'] for c in families_stats.values()) / len(families_stats),
                       'families': families_stats}
    return result, [{'family': f, 'vector': p.tolist()} for f, p in zip(families, prototypes)]


def run():
    config = json.loads((ROOT / 'bench/pilot.json').read_text())
    settings = config['training']
    torch.set_num_threads(settings['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config['seed'])
    manifest_path = ROOT / '.data/prepared.json'
    manifest = json.loads(manifest_path.read_text())
    for key, path in [('configSha256', ROOT / 'bench/pilot.json'), ('fontsSha256', ROOT / 'bench/fonts.json'), ('preparationSha256', ROOT / 'src/prepare.mjs'), ('tensorSha256', ROOT / '.data/prepared.f32')]:
        if digest(path) != manifest[key]:
            raise ValueError(f'Stale or corrupt preparation: {path}')
    samples = manifest['samples']
    validate_samples(samples)
    if (ROOT / '.data/prepared.f32').stat().st_size != len(samples) * manifest['height'] * manifest['width'] * 4:
        raise ValueError('Prepared tensor length mismatch')
    array = np.fromfile(ROOT / '.data/prepared.f32', dtype='<f4').reshape(len(samples), 1, manifest['height'], manifest['width'])
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError('Invalid prepared pixels')
    pixels = torch.from_numpy(array)
    train_families = sorted({s['family'] for s in samples if s['role'] == 'train'})
    pools = [[i for i, s in enumerate(samples) if s['role'] == 'train' and s['family'] == family] for family in train_families]
    n = settings['batchPerFamily']
    if len(pools) < 2 or any(len(p) < n for p in pools):
        raise ValueError('Need two families with enough distinct training samples')
    labels = torch.arange(len(pools)).repeat_interleave(n)
    model = Encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learningRate'])
    losses = []
    start = time.perf_counter()
    first_gradient = None
    for step in range(settings['steps']):
        model.train()
        batch = np.concatenate([rng.choice(pool, n, replace=False) for pool in pools])
        optimizer.zero_grad(set_to_none=True)
        loss = contrastive_loss(model(pixels[batch]), labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        gradient = sum(float(p.grad.square().sum()) for p in model.parameters()) ** .5
        if not np.isfinite(gradient) or gradient == 0:
            raise ValueError('Invalid gradients')
        if first_gradient is None:
            first_gradient = gradient
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f'Step {step + 1}/{settings["steps"]}: loss {np.mean(losses[-20:]):.4f}', flush=True)
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        vectors = torch.cat([model(batch) for batch in pixels.split(128)])
    groups, catalog = evaluate(vectors, samples)
    artifact = export_model(model)
    artifact.update({'version': 1, 'preparation': {**config['input'], 'sha256': manifest['preparationSha256']},
                     'catalog': catalog, 'note': 'Synthetic pilot, no calibrated rejection; requires a future inference implementation.'})
    model_path = ROOT / '.data/pilot-model.json'
    model_path.write_text(json.dumps(artifact, separators=(',', ':')) + '\n')
    restored = load_export(json.loads(model_path.read_text()))
    with torch.no_grad():
        parity = float((restored(pixels[:8]) - model(pixels[:8])).abs().max())
    if parity > 1e-6:
        raise ValueError(f'Export parity failed: {parity}')
    torch.save(model.state_dict(), ROOT / '.data/pilot-model.pt')
    report = {'description': 'One fixed-seed synthetic development run. Unseen-dev families enter only through prototypes; no final-test or real-screenshot claim.',
              'seed': config['seed'], 'steps': settings['steps'], 'trainingFamilies': train_families,
              'unseenFamilies': sorted({s['family'] for s in samples if s['split'] == 'unseen-dev'}),
              'trainingSamples': sum(s['role'] == 'train' for s in samples), 'parameters': sum(p.numel() for p in model.parameters()),
              'embeddingDimensions': 32, 'seconds': elapsed, 'device': 'cpu', 'threads': settings['threads'],
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'machine': platform.machine(), 'platform': platform.platform()},
              'lossFirst20': float(np.mean(losses[:20])), 'lossLast20': float(np.mean(losses[-20:])),
              'firstGradientNorm': first_gradient, 'exportMaxAbsoluteError': parity, 'exportJsonBytes': model_path.stat().st_size,
              'manifestSha256': digest(manifest_path), 'modelSha256': digest(model_path), 'tensorSha256': manifest['tensorSha256'],
              'groups': groups}
    (ROOT / 'bench/training.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Trained {report["parameters"]:,} parameters in {elapsed:.1f}s; export error {parity}')
    for key, stats in groups.items():
        print(f'{key}: top1 {stats["top1"]:.1%}, top5 {stats["top5"]:.1%} ({stats["queries"]} queries)')


if __name__ == '__main__':
    run()
