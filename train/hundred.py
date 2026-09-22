"""Train 100 families; report length/condition slices and validation-only rejection."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from train.robustness import read, write, sha
from train.ten import batch
from train.ten_model import Classifier, export, load_export

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/hundred'


def load_data():
    config = read(ROOT / 'bench/hundred.json'); manifest = read(DATA / 'prepared.json')
    if manifest['dataConfig'] != config['data']: raise ValueError('Changed data configuration')
    for key, path in [('fontsSha256', ROOT / 'bench/fonts-100.json'), ('textsSha256', ROOT / 'bench/hundred-texts.json'),
                      ('generatorSha256', ROOT / 'train/hundred_data.py'), ('runnerSha256', ROOT / 'scripts/hundred.mjs'),
                      ('inputSha256', ROOT / 'src/input.mjs'), ('preparationSha256', ROOT / 'src/prepare.mjs'), ('tensorSha256', DATA / 'prepared.u8')]:
        if manifest[key] != sha(path): raise ValueError(f'Stale data: {path}')
    sources = manifest['samples']; offset = 0; coverage = set(); texts = {}
    known = set(config['data']['fonts'])
    families = {f['id']: f['split'] for f in read(ROOT / 'bench/fonts-100.json')['fonts']}
    for s in sources:
        expected = families.get(s['family'])
        if not expected or (expected == 'train' and (s['family'] not in known or s['role'] not in ['train', 'validation', 'test'])) or (expected != 'train' and (s['family'] in known or s['role'] != expected)):
            raise ValueError('Invalid family split')
        if len(s['text']) > 1: texts.setdefault(s['role'].removeprefix('unknown-'), set()).add(s['text'].lower())
    if set(texts) != {'train', 'validation', 'test'}: raise ValueError('Missing text split')
    if any(texts[a] & texts[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]): raise ValueError('Text leakage')
    for w in manifest['windows']:
        if not isinstance(w, dict) or any(type(w.get(k)) is not int for k in ['source', 'offset', 'width', 'height']) or w['offset'] != offset or not 1 <= w['width'] <= 128 or not 1 <= w['height'] <= 48 or not 0 <= w['source'] < len(sources): raise ValueError('Invalid packed window')
        offset += w['width'] * w['height']; coverage.add(w['source'])
    if len(coverage) != len(sources) or (DATA / 'prepared.u8').stat().st_size != offset: raise ValueError('Incomplete tensor')
    return config, manifest, np.memmap(DATA / 'prepared.u8', dtype=np.uint8, mode='r')


def predict(model, manifest, pixels, roles):
    windows, samples = manifest['windows'], manifest['samples']
    indexes = [i for i, w in enumerate(windows) if samples[w['source']]['role'] in roles]
    if not indexes: raise ValueError('No windows for requested split')
    sources = sorted({windows[i]['source'] for i in indexes}); mapping = {s: i for i, s in enumerate(sources)}
    logits, owners = [], []
    model.eval(); device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(indexes), 128):
            chosen = indexes[start:start + 128]
            logits.append(model(*(t.to(device) for t in batch(pixels, windows, chosen))).cpu().numpy())
            owners.extend(mapping[windows[i]['source']] for i in chosen)
    return sources, np.concatenate(logits), np.array(owners)


def probabilities(logits, owners, count, temperature=1):
    if not np.isfinite(temperature) or temperature <= 0: raise ValueError('Invalid temperature')
    values = logits.astype(np.float64) / temperature; values -= values.max(1, keepdims=True)
    values = np.exp(values); values /= values.sum(1, keepdims=True)
    result = np.zeros((count, logits.shape[1])); np.add.at(result, owners, values)
    return result / np.bincount(owners, minlength=count)[:, None]


def metrics(manifest, sources, probs, threshold=1.01):
    fonts = manifest['dataConfig']['fonts']; groups = {}; accepted = correct = unknown = false_accept = 0
    confusion = np.zeros((len(fonts), len(fonts)), dtype=np.int64)
    for i, source in enumerate(sources):
        s = manifest['samples'][source]; label = fonts.index(s['family']) if s['family'] in fonts else -1
        predicted = int(probs[i].argmax()); accept = probs[i, predicted] >= threshold
        if label < 0:
            unknown += 1; false_accept += int(accept); continue
        hit = label == predicted; accepted += int(accept); correct += int(accept and hit)
        confusion[label, predicted] += 1
        for key in ['all', f'length/{s["length"]}', f'condition/{s["condition"]}', f'{s["length"]}/{s["condition"]}', f'font/{s["family"]}']:
            entry = groups.setdefault(key, {'count': 0, 'correct': 0, 'top3': 0})
            entry['count'] += 1; entry['correct'] += int(hit); entry['top3'] += int(label in probs[i].argsort()[-3:])
    for g in groups.values():
        g['accuracy'] = g['correct'] / g['count']; g['top3Accuracy'] = g['top3'] / g['count']
    return {'groups': groups, 'confusion': confusion.tolist(), 'rejection': {'threshold': threshold, 'accepted': accepted,
            'acceptedCorrect': correct, 'acceptedAccuracy': correct / accepted if accepted else None,
            'coverage': accepted / groups['all']['count'], 'unknownCount': unknown, 'unknownAccepted': false_accept,
            'unknownFalseAcceptRate': false_accept / unknown if unknown else None}}


def calibrate(manifest, sources, logits, owners):
    fonts = manifest['dataConfig']['fonts']
    labels = np.array([fonts.index(manifest['samples'][s]['family']) if manifest['samples'][s]['family'] in fonts else -1 for s in sources])
    known = labels >= 0
    if not known.any() or known.all(): raise ValueError('Calibration requires known and unknown fonts')
    best = None
    for t in np.geomspace(.5, 5, 41):
        p = probabilities(logits, owners, len(sources), t)
        loss = -np.log(np.maximum(p[known, labels[known]], 1e-12)).mean()
        if best is None or loss < best[0]: best = (loss, t, p)
    loss, temperature, p = best; score = p.max(1); correct = p.argmax(1) == labels
    # Choose the most coverage meeting both requirements on validation only.
    threshold = 1.01; coverage = 0
    for value in np.unique(np.quantile(score, np.linspace(0, 1, 1001))):
        accept = score >= value; count = int((accept & known).sum())
        if count >= 100 and correct[accept & known].mean() >= .95 and accept[~known].mean() <= .05 and count > coverage:
            threshold, coverage = float(value), count
    return {'temperature': float(temperature), 'threshold': threshold, 'validationNLL': float(loss),
            'targetAcceptedAccuracy': .95, 'targetUnknownFalseAcceptRate': .05, 'validation': metrics(manifest, sources, p, threshold)}


def train(steps=None, resume=False, context=False):
    config, manifest, pixels = load_data(); settings = dict(config['training'])
    steps = settings['steps'] if steps is None else steps
    if type(steps) is not int or steps < 1: raise ValueError('Training steps must be a positive integer')
    context = context or settings.get('context', False) or (resume and len(read(DATA / 'model.json')['layers']) == 6)
    if context:
        settings['context'] = True; settings['dilations'] = [*settings['dilations'][:4], 1]
    torch.set_num_threads(settings['threads']); torch.manual_seed(settings['seed']); torch.use_deterministic_algorithms(True)
    device = settings['device']
    if device == 'mps' and not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
    fonts = config['data']['fonts']; model = Classifier(len(fonts), dilations=settings['dilations'], context=context)
    parent = read(ROOT / 'bench/hundred-training.json') if resume else None
    if resume:
        previous = read(DATA / 'model.json')
        if parent['modelSha256'] != sha(DATA / 'model.json') or previous['preparation'] != {**config['data']['input'], 'sha256': manifest['inputSha256'], 'normalizerSha256': manifest['preparationSha256']}:
            raise ValueError('Stale resume model')
        original = Classifier(len(fonts), dilations=previous['dilations'], context=len(previous['layers']) == 6)
        original.load_state_dict(torch.load(DATA / 'best.pt', weights_only=True, map_location='cpu')['state'])
        if export(original, fonts, previous['preparation'])[0] != previous: raise ValueError('Checkpoint mismatch')
        if context and not original.context:
            model.convs.load_state_dict(original.convs.state_dict(), strict=False)
            model.norms.load_state_dict(original.norms.state_dict(), strict=False)
            model.head.load_state_dict(original.head.state_dict())
            torch.nn.init.dirac_(model.convs[-1].weight); torch.nn.init.zeros_(model.convs[-1].bias)
        else: model.load_state_dict(original.state_dict())
    else:
        # Reuse the existing ten-font visual features; all layers remain trainable.
        old = read(ROOT / '.data/ten/model.json')
        state = torch.load(ROOT / '.data/ten/best.pt', weights_only=True, map_location='cpu')['state']
        original = Classifier(len(old['fonts']), dilations=old['dilations']); original.load_state_dict(state)
        if export(original, old['fonts'], old['preparation'])[0] != old: raise ValueError('Invalid initialization checkpoint')
        model.convs.load_state_dict(original.convs.state_dict(), strict=False); model.norms.load_state_dict(original.norms.state_dict(), strict=False)
        if context:
            torch.nn.init.dirac_(model.convs[-1].weight); torch.nn.init.zeros_(model.convs[-1].bias)
        with torch.no_grad():
            for i, family in enumerate(old['fonts']):
                model.head.weight[fonts.index(family)].copy_(original.head.weight[i]); model.head.bias[fonts.index(family)].copy_(original.head.bias[i])
    model.to(device); rng = np.random.default_rng(settings['seed'])
    pools = {(family, renderer): [] for family in fonts for renderer in ['pillow', 'chromium']}
    for i, w in enumerate(manifest['windows']):
        s = manifest['samples'][w['source']]
        if s['role'] == 'train': pools[s['family'], s['renderer']].append(i)
    if any(not v for v in pools.values()): raise ValueError('Missing training pool')
    rate = settings['learningRate'] / (4 if resume else 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=rate / 20)
    best = -1; history = []; losses = []; start = time.perf_counter()
    if resume:
        sources, logits, owners = predict(model, manifest, pixels, ['validation'])
        report = metrics(manifest, sources, probabilities(logits, owners, len(sources)))
        best = report['groups']['all']['accuracy']
        history.append({'step': 0, 'groups': report['groups']})
        # The starting weights remain eligible, including an initialized context layer.
        torch.save({'state': model.state_dict(), 'step': 0}, DATA / 'best.pt')
    for step in range(steps):
        model.train(); labels = rng.integers(len(fonts), size=settings['batch'])
        chosen = [int(rng.choice(pools[fonts[label], 'pillow' if rng.integers(2) else 'chromium'])) for label in labels]
        x, sizes = (t.to(device) for t in batch(pixels, manifest['windows'], chosen))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, sizes), torch.tensor(labels, device=device), label_smoothing=.03)
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward(); optimizer.step(); scheduler.step(); losses.append(loss.item())
        if (step + 1) % 1000 == 0: print(f'{step+1}/{steps}: loss {np.mean(losses[-200:]):.4f}; {time.perf_counter()-start:.0f}s', flush=True)
        if (step + 1) % 5000 == 0 or step + 1 == steps:
            sources, logits, owners = predict(model, manifest, pixels, ['validation'])
            report = metrics(manifest, sources, probabilities(logits, owners, len(sources)))
            score = report['groups']['all']['accuracy']; history.append({'step': step + 1, 'groups': report['groups']})
            print(f'Validation {step+1}: {score:.2%}; by length '+str({k: round(v['accuracy'], 3) for k,v in report['groups'].items() if k.startswith('length/')}), flush=True)
            if score > best:
                best = score; torch.save({'state': model.state_dict(), 'step': step+1}, DATA / 'best.pt')
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(DATA / 'best.pt', weights_only=True, map_location='cpu'); model.cpu().load_state_dict(checkpoint['state'])
    preparation = {**config['data']['input'], 'sha256': manifest['inputSha256'], 'normalizerSha256': manifest['preparationSha256']}
    artifact, quantized = export(model, fonts, preparation)
    sources, logits, owners = predict(quantized.to(device), manifest, pixels, ['validation', 'unknown-validation'])
    calibration = calibrate(manifest, sources, logits, owners)
    validation = metrics(manifest, sources, probabilities(logits, owners, len(sources)))
    if validation['groups']['all']['accuracy'] < best - .01: raise ValueError('Int8 regression exceeds one point')
    write(DATA / 'model.json', artifact); write(DATA / 'calibration.json', calibration)
    write(ROOT / 'bench/hundred-training.json', {'task': '100-font-classification', 'fonts': fonts, 'training': settings, 'steps': steps,
          'seconds': elapsed, 'totalSeconds': elapsed + (parent['totalSeconds'] if parent else 0), 'selectedStep': checkpoint['step'],
          'parameters': sum(p.numel() for p in quantized.parameters()), 'modelSha256': sha(DATA / 'model.json'),
          'manifestSha256': sha(DATA / 'prepared.json'), 'trainerSha256': sha(Path(__file__)), 'architectureSha256': sha(ROOT / 'train/ten_model.py'),
          'initialModelSha256': parent['modelSha256'] if parent else sha(ROOT / '.data/ten/model.json'), 'torch': torch.__version__,
          'validationInt8': validation, 'calibration': calibration, 'history': history})
    print(f'Exported {best:.2%} validation, {elapsed:.1f}s; rejection: {calibration["validation"]["rejection"]}', flush=True)


def final_evaluation():
    config, manifest, pixels = load_data(); report = read(ROOT / 'bench/hundred-training.json')
    if report['manifestSha256'] != sha(DATA / 'prepared.json') or report['modelSha256'] != sha(DATA / 'model.json'): raise ValueError('Stale model')
    torch.set_num_threads(4)
    sources, logits, owners = predict(load_export(read(DATA / 'model.json')).to(config['training']['device']), manifest, pixels, ['test', 'unknown-test'])
    calibration = read(DATA / 'calibration.json')
    if calibration != report['calibration']: raise ValueError('Stale calibration')
    probabilities_ = probabilities(logits, owners, len(sources), calibration['temperature'])
    result = metrics(manifest, sources, probabilities_, calibration['threshold'])
    write(ROOT / 'bench/hundred-test.json', {**result, 'modelSha256': report['modelSha256'], 'manifestSha256': report['manifestSha256'],
          'calibrationSha256': sha(DATA / 'calibration.json'),
          'description': 'New Chromium test strings; single characters share the alphabet but not rendering plans. Unknown test families disjoint from calibration. Synthetic only.'})
    write(DATA / 'test-predictions.json', [{'source': int(s), 'family': manifest['samples'][s]['family'], 'top': [config['data']['fonts'][k] for k in p.argsort()[-3:][::-1]], 'score': float(p.max())} for s,p in zip(sources,probabilities_)])
    print({k:v for k,v in result['groups'].items() if k.startswith(('length/', 'condition/'))}); print(result['rejection'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['train', 'evaluate']); parser.add_argument('--steps', type=int); parser.add_argument('--resume', action='store_true'); parser.add_argument('--context', action='store_true')
    args = parser.parse_args()
    if args.command == 'train': train(args.steps, args.resume, args.context)
    else: final_evaluation()
