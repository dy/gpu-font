"""Widen the proven encoder, train all eligible families, export a measured int8 candidate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import gzip

import numpy as np
import torch
from torch.nn import functional as F

from train.corpus_data import DATA, ROOT, pins
from train.ten import batch
from train.ten_model import Classifier, export, load_export
from train.robustness import read, write, sha


def widen(source, classes):
    """Duplicate channels, halve their downstream weights: preserve the initial encoder exactly."""
    if not source.context or source.wide: raise ValueError('Expected the compact context encoder')
    result = Classifier(classes, context=True, wide=True, dilations=source.dilations)
    with torch.no_grad():
        for i, (a, b) in enumerate(zip(source.convs, result.convs)):
            weights = a.weight.repeat_interleave(2, 0)
            if i: weights = weights.repeat_interleave(2, 1) / 2
            b.weight.copy_(weights); b.bias.copy_(a.bias.repeat_interleave(2))
        for a, b in zip(source.norms, result.norms):
            for key, value in a.state_dict().items():
                b.state_dict()[key].copy_(value.repeat_interleave(2) if value.ndim else value)
    return result


def load_data():
    manifest = read(DATA / 'prepared.json'); path = DATA / 'prepared.u8'
    if manifest['pins'] != pins() or manifest['sha256'] != sha(path): raise ValueError('Changed corpus tensors or preparation')
    fonts = manifest['fonts']; samples = manifest['samples']; windows = manifest['windows']
    if not fonts or len(set(fonts)) != len(fonts): raise ValueError('Invalid corpus labels')
    end = 0; owners = set()
    for window in windows:
        if any(type(window.get(k)) is not int for k in ['width', 'height', 'source', 'offset']) or window['offset'] != end or not 0 <= window['source'] < len(samples) or not 1 <= window['width'] <= 128 or not 1 <= window['height'] <= 48: raise ValueError('Invalid corpus window')
        end += window['width'] * window['height']; owners.add(window['source'])
    if end != path.stat().st_size or len(owners) != len(samples): raise ValueError('Incomplete corpus tensors')
    coverage = Counter()
    for sample in samples:
        if sample['role'] not in ['train', 'validation', 'test'] or type(sample['label']) is not int or not 0 <= sample['label'] < len(fonts) or sample['family'] != fonts[sample['label']]: raise ValueError('Invalid corpus sample')
        coverage[sample['label'], sample['role']] += 1
    if any(not coverage[i, role] for i in range(len(fonts)) for role in ['train', 'validation', 'test']): raise ValueError('Incomplete family split coverage')
    return manifest, np.memmap(path, dtype=np.uint8, mode='r')


def evaluate(model, manifest, pixels, role, quick=False):
    samples = manifest['samples']; windows = manifest['windows']; n = len(manifest['fonts'])
    owners = [i for i,s in enumerate(samples) if s['role'] == role and (not quick or (s['condition'] == 'clean' and s['length'] == '8+'))]
    mapping = {owner:i for i,owner in enumerate(owners)}
    selected = [i for i,w in enumerate(windows) if w['source'] in mapping]
    sums = np.zeros((len(owners), n), dtype=np.float32); counts = np.zeros(len(owners), dtype=np.int32)
    model.eval(); device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(selected), 128):
            indexes = selected[start:start+128]
            probabilities = model(*(t.to(device) for t in batch(pixels, windows, indexes))).softmax(1).cpu().numpy()
            ids = np.array([mapping[windows[i]['source']] for i in indexes]); np.add.at(sums, ids, probabilities); np.add.at(counts, ids, 1)
    if not len(owners) or np.any(counts == 0): raise ValueError('Missing evaluation queries')
    sums /= counts[:,None]; groups = {}; confusions = Counter(); family_counts=np.zeros(n); family_hits=np.zeros(n)
    seed_fonts = {name.replace('-', '') for name in read(ROOT/'models/hundred/model.json')['fonts']}
    for owner, probabilities in zip(owners, sums):
        sample = samples[owner]; label = sample['label']; prediction = int(probabilities.argmax())
        hits = int(prediction == label), int(label in np.argpartition(probabilities, -min(5,n))[-min(5,n):])
        family_counts[label]+=1; family_hits[label]+=hits[0]
        if prediction != label: confusions[(label, prediction)] += 1
        keys = ['all', 'condition/'+sample['condition'], 'length/'+sample['length'], 'script/'+sample['script'], 'seed-100' if sample['family'] in seed_fonts else 'added-families']
        for key in keys:
            group = groups.setdefault(key, {'count':0, 'correct':0, 'top5':0}); group['count'] += 1; group['correct'] += hits[0]; group['top5'] += hits[1]
    for group in groups.values(): group['accuracy'] = group['correct']/group['count']; group['top5Accuracy'] = group['top5']/group['count']
    groups['all']['macroAccuracy'] = float(np.mean(family_hits[family_counts>0]/family_counts[family_counts>0]))
    return {'groups':groups, 'confusions':[{'true':manifest['fonts'][a], 'predicted':manifest['fonts'][b], 'count':count} for (a,b),count in confusions.most_common(30)]}


def train(steps, resume=False):
    if steps < 1: raise ValueError('Positive training steps required')
    torch.set_num_threads(4); torch.manual_seed(20260930); torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
    manifest, pixels = load_data(); fonts = manifest['fonts']
    original = read(ROOT / 'models/deskew/model.json')
    source = Classifier(len(original['fonts']), context=True, dilations=original['dilations'])
    source.load_state_dict(torch.load(ROOT / 'models/deskew/best.pt', weights_only=True, map_location='cpu')['state'])
    if export(source, original['fonts'], original['preparation'])[0] != original: raise ValueError('Initial checkpoint/export mismatch')
    model = widen(source, len(fonts))
    with torch.no_grad():
        for i, name in enumerate(original['fonts']):
            if name.replace('-', '') not in fonts: continue
            j = fonts.index(name.replace('-', ''))
            model.head.weight[j].copy_(source.head.weight[i].repeat_interleave(2) / 2)
            model.head.bias[j].copy_(source.head.bias[i])
    initial_checkpoint = sha(DATA/'best.pt' if resume else ROOT/'models/deskew/best.pt')
    if resume:
        saved = torch.load(DATA / 'best.pt', weights_only=True, map_location='cpu')
        if saved['fonts'] != fonts or saved['dataSha256'] != sha(DATA/'prepared.json'): raise ValueError('Changed resume labels/data')
        model.load_state_dict(saved['state'])
    model.to('mps'); windows = manifest['windows']; samples = manifest['samples']
    pools = [[] for _ in fonts]
    for i,w in enumerate(windows):
        if samples[w['source']]['role'] == 'train': pools[samples[w['source']]['label']].append(i)
    if any(not pool for pool in pools): raise ValueError('Missing family training windows')
    aliases = {(label, group['script']): group['labels'] for group in manifest['aliases'] for label in group['labels']}
    rng = np.random.default_rng(20260930); rate = .0003 if resume else .001
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=rate/30)
    best = -1; history = []; start = time.perf_counter(); losses = []; data_sha = sha(DATA/'prepared.json')
    def retain(step):
        nonlocal best
        report = evaluate(model, manifest, pixels, 'validation', quick=True)
        score = report['groups']['all']['macroAccuracy']; history.append({'step':step, 'seconds':time.perf_counter()-start, 'validation':report})
        print(f'Corpus {step}: clean long-text validation (family mean) {score:.2%}, top5 {report["groups"]["all"]["top5Accuracy"]:.2%}', flush=True)
        if score > best:
            best = score
            torch.save({'state':{k:v.cpu().clone() for k,v in model.state_dict().items()}, 'step':step, 'fonts':fonts, 'dataSha256':data_sha}, DATA / 'best.pt')
        write(DATA / 'progress.json', {'steps':steps, 'history':history, 'best':best})
    retain(0)
    for step in range(1, steps+1):
        model.train(); labels = rng.integers(len(fonts), size=96)
        indexes = [int(rng.choice(pools[label])) for label in labels]
        optimizer.zero_grad(set_to_none=True)
        targets = np.zeros((len(labels), len(fonts)), dtype=np.float32)
        for i,(label,index) in enumerate(zip(labels,indexes)):
            choices = aliases.get((int(label), samples[windows[index]['source']]['script']), [label])
            targets[i, choices] = 1/len(choices)
        loss = F.cross_entropy(model(*(t.to('mps') for t in batch(pixels, windows, indexes))), torch.from_numpy(targets).to('mps'), label_smoothing=.02)
        if not torch.isfinite(loss): raise ValueError('Nonfinite corpus loss')
        loss.backward(); optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        if step % 500 == 0:
            print(f'Corpus {step}/{steps}: loss {np.mean(losses):.3f}, {time.perf_counter()-start:.0f}s', flush=True); losses=[]
        if step % 4000 == 0 or step == steps: retain(step)
    checkpoint = torch.load(DATA/'best.pt', weights_only=True, map_location='cpu')
    model.cpu().load_state_dict(checkpoint['state'])
    float_report = evaluate(model.to('mps'), manifest, pixels, 'validation'); model.cpu()
    preparation = {**original['preparation'], 'method':'deskew-windows', 'lineSha256':sha(ROOT/'src/line.mjs')}
    artifact, restored = export(model, fonts, preparation)
    artifact['ambiguities'] = [{'script': g['script'], 'fonts':[fonts[i] for i in g['labels']], 'basis':'identical training renderings'} for g in manifest['aliases']]
    catalog = read(ROOT/'bench/corpus.json')
    artifact['sourceCommit'] = catalog['commit']
    artifact['catalog'] = []
    for family in catalog['families']:
        if family['id'] not in fonts: continue
        face = next(f for f in family['faces'] if f['path'] == family['selected'])
        artifact['catalog'].append({'id':family['id'], 'family':family['family'], 'trainingFace':{'weight':face['axes'].get('wght',{}).get('default',face['weight']), 'style':'italic' if face['italic'] else 'normal'}, 'scripts':sorted(family['alphabets'])})
    write(DATA/'model.json', artifact)
    quant_report = evaluate(restored.to('mps'), manifest, pixels, 'validation')
    encoded = (DATA/'model.json').read_bytes()
    if len(encoded) > 10_000_000: raise ValueError('Model exceeds 10 MB budget')
    report = {'seed':20260930, 'steps':steps, 'selectedStep':checkpoint['step'], 'seconds':time.perf_counter()-start,
              'families':len(fonts), 'parameters':sum(p.numel() for p in model.parameters()), 'modelBytes':len(encoded), 'gzipBytes':len(gzip.compress(encoded,compresslevel=9,mtime=0)),
              'modelSha256':sha(DATA/'model.json'), 'dataSha256':sha(DATA/'prepared.json'), 'catalogSha256':sha(ROOT/'bench/corpus.json'),
              'trainerSha256':sha(Path(__file__)), 'encoderSha256':sha(ROOT/'train/ten_model.py'), 'initialModelSha256':sha(ROOT/'models/deskew/model.json'), 'initialCheckpointSha256':initial_checkpoint, 'continuedFromCorpus':resume,
              'history':history, 'validationFloat':float_report, 'validationInt8':quant_report,
              'quantizationGatePassed':quant_report['groups']['all']['accuracy'] >= float_report['groups']['all']['accuracy']-.01,
              'scope':'Default-face closed-set recognition; synthetic RAQM/Pillow crops, disjoint text banks. No real-screenshot, unknown-rejection, weight/style or universal-script accuracy claim.'}
    write(ROOT/'bench/corpus-training.json', report)
    print({k:report[k] for k in ['families','modelBytes','quantizationGatePassed']}, quant_report['groups']['all'], flush=True)


def final_evaluation(browser=False):
    torch.set_num_threads(4)
    manifest, pixels = load_data(); report = read(ROOT/'bench/corpus-training.json')
    if report['modelSha256'] != sha(DATA/'model.json') or report['dataSha256'] != sha(DATA/'prepared.json'): raise ValueError('Changed selected model/data')
    model = load_export(read(DATA/'model.json')).to('mps')
    if browser:
        manifest = read(DATA/'browser.json')
        if manifest['fonts'] != read(DATA/'model.json')['fonts'] or manifest['sha256'] != sha(DATA/'browser.u8'): raise ValueError('Changed browser corpus')
        for key,path in [('catalog','bench/corpus.json'),('runner','scripts/corpus-browser.mjs'),('preparation','scripts/corpus-prepare.mjs'),('line','src/line.mjs'),('input','src/input.mjs'),('normalizer','src/prepare.mjs')]:
            if manifest['pins'][key] != sha(ROOT/path): raise ValueError('Changed browser preparation')
        pixels = np.memmap(DATA/'browser.u8', dtype=np.uint8, mode='r')
    result = evaluate(model, manifest, pixels, 'test')
    write(ROOT/('bench/corpus-browser.json' if browser else 'bench/corpus-test.json'), {'modelSha256':report['modelSha256'], 'dataSha256':sha(DATA/('browser.json' if browser else 'prepared.json')), 'scope':'Held-out Chromium queries. Chromium is absent from corpus fine-tuning but was used in the 100-family seed model.' if browser else report['scope'], 'test':result})
    print(result['groups']['all'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['train','test','browser']); p.add_argument('--steps', type=int, default=40000); p.add_argument('--resume', action='store_true'); args=p.parse_args()
    if args.command == 'train': train(args.steps, args.resume)
    else: final_evaluation(args.command=='browser')
