"""Fonts outside Google Fonts as queries: the default face of every open family (bench/open-fonts.json) that a shipped
catalog indexes, Latin lower and title case at the catalog benchmark's sizes with one degraded copy, rendered by the
frozen Chromium renderer. None of them trained the model, so this is the held-out read, and it is searched as the page's
All: every shipped catalog's rows at once, the Google catalog included, so the figure is what a user of those catalogs gets.

    python -m train.style_open plan                      # .data/style/open-plan.json and open-instances.json
    python -m train.style_bench render --name open && python -m train.style_bench pack --name open
    python -m train.style_open evaluate --run student-30k # against models/encoder/*.json as shipped
"""
import argparse
import base64
import hashlib
import random

import numpy as np

from scripts.corpus import ROOT
from train.encoder_data import save
from train.robustness import read, sha
from train.style_bench import OUT, PINS, query_text
from train.style_data import FREQUENT
from train.ten_model import unpack_bits

SEED = 20260925
SIZES = [16, 20, 24, 32, 40, 48, 64]


def shipped_faces():
    """The face ids every shipped catalog indexes: the files Chromium loads, since the catalog build skips the rest."""
    index = read(ROOT/'models/encoder/catalogs/index.json')
    return {face['id'] for c in index for face in read(ROOT/'models/encoder/catalogs'/f"{c['id']}.json")['faces']}


def queryable(alphabet):
    """Whether the frozen text recipe can draw a Latin query from this alphabet: eight of the frequent lowercase letters (a capitals-only face cannot)."""
    return sum(c in alphabet for c in set(FREQUENT)) >= 8


def plan():
    inventory = read(ROOT/'bench/open-fonts.json'); alphabets = read(ROOT/inventory['store']/'alphabets.json'); indexed = shipped_faces()
    families = [f for f in inventory['families'] if not f.get('excluded') and queryable(alphabets.get(f['id'], {}).get('Latn', ''))]
    rng = random.Random(SEED); queries = []; instances = {}
    for family in sorted(families, key=lambda f: f['id']):
        face = next((x for x in family['faces'] if x['path'] == family['selected']), None)
        if face is None: continue
        path = ROOT/inventory['store']/face['path']; face_id = f"{family['id']}/{path.stem}"
        if face_id not in indexed or not path.exists(): continue
        instances[face_id] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for n, case in enumerate(['lower', 'title']):
            value = query_text(rng, 'Latn', alphabets[family['id']]['Latn'], case); size = rng.choice(SIZES)
            base = {'role': 'test', 'family': family['id'], 'source': family['source'], 'face': face_id, 'faceId': face_id, 'weight': face['weight'], 'italic': face['italic'],
                    'script': 'Latn', 'case': case, 'text': value, 'size': size, 'length': len(value)}
            queries.append({**base, 'condition': 'clean'})
            if n == 0: queries.append({**base, 'condition': 'degraded'})
    pins = {p: sha(ROOT/p) for p in PINS} | {'train/style_open.py': sha(ROOT/'train/style_open.py'), 'bench/open-fonts.json': sha(ROOT/'bench/open-fonts.json')}
    save(OUT/'open-plan.json', {'pins': pins, 'seed': SEED, 'queries': queries}); save(OUT/'open-instances.json', instances)
    print('Open benchmark plan:', len(queries), 'queries of', len(instances), 'families', flush=True)


def shipped_rows():
    """Every shipped catalog's rows and the family and scripts each belongs to: the page's All."""
    encoder = read(ROOT/'models/encoder/encoder.json'); dims = encoder['dimensions']; bound = sha(ROOT/'models/encoder/encoder.json')
    index = read(ROOT/'models/encoder/catalogs/index.json'); files = [ROOT/'models/encoder/google-fonts.json'] + [ROOT/'models/encoder/catalogs'/f"{c['id']}.json" for c in index]
    rows = []; owners = []; families = []; scripts = []; catalogs = []
    for path in files:
        data = read(path); v = data['vectors']; bits = {'int8-base64': 8, 'int4-base64': 4}[v['encoding']]
        if data['dimensions'] != dims or data['encoderSha256'] != bound: raise ValueError(f'{path.name} was built for another encoder')
        values = unpack_bits(base64.b64decode(v['data']), bits, v['shape'][0] * dims).reshape(v['shape'][0], dims).astype(np.float32) * np.array(v['scales'], np.float32)[:, None]
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        for r, owner in enumerate(v['owners']):
            face = data['faces'][owner]; key = face['familyId']
            if key not in families: families.append(key); scripts.append(set(face.get('scripts') or ['Latn'])); catalogs.append(path.stem)
            owners.append(families.index(key))
        rows.append(values)
    return np.concatenate(rows), np.array(owners), families, scripts, catalogs


def evaluate(run):
    """The run as exported, on the open benchmark and on the catalog validation queries, against every shipped catalog."""
    from train.style import Setup, load_run, load_bench, embed_manifest
    setup = Setup(['train']); model, heads, _ = load_run(run, exported=True)
    vectors, owner, families, family_scripts, catalogs = shipped_rows(); fpos = {f: i for i, f in enumerate(families)}
    print(f'{len(vectors)} rows of {len(families)} families from {len(set(catalogs))} catalogs', flush=True)
    order = np.argsort(owner, kind='stable'); starts = np.r_[0, np.flatnonzero(np.diff(owner[order])) + 1]
    latin = np.array(['Latn' in s for s in family_scripts])
    report = {}
    for name, roles, key in [('open', ['test'], 'family'), ('catalog', ['validation'], 'family')]:
        bench, pixels = load_bench(name); selected = [i for i, s in enumerate(bench['samples']) if s['role'] in roles and s['script'] == 'Latn']
        q = embed_manifest(model, bench, pixels, selected, 'mps'); sims = q @ vectors.T
        scores = np.full((len(q), len(families)), -2, np.float32)
        for a, b in zip(starts, np.r_[starts[1:], len(order)]): cols = order[a:b]; scores[:, owner[cols[0]]] = sims[:, cols].max(1)
        scores[:, ~latin] = -3
        rows = []
        for n, i in enumerate(selected):
            s = bench['samples'][i]; t = fpos.get(s[key]); top = np.argsort(-scores[n], kind='stable')[:5]
            rows.append({'found': t is not None, 'top1': t == top[0], 'top5': t is not None and t in top, 'condition': s['condition'], 'source': s.get('source', 'google-fonts'), 'size': s['size']})
        summary = lambda subset: {'queries': len(subset), 'inCatalogs': sum(r['found'] for r in subset), 'top1': float(np.mean([r['top1'] for r in subset])), 'top5': float(np.mean([r['top5'] for r in subset]))}
        groups = {'all': summary(rows)}
        for k in ['condition', 'source']:
            for value in sorted({r[k] for r in rows}): groups[f'{k}/{value}'] = summary([r for r in rows if r[k] == value])
        report[name] = groups
        for k, g in groups.items(): print(f"{name:8} {k:26} n {g['queries']:5} in catalogs {g['inCatalogs']:5} top5 {100*g['top5']:.1f} top1 {100*g['top1']:.1f}", flush=True)
    save(ROOT/'bench/style-open.json', {'run': run, 'encoderSha256': sha(ROOT/'models/encoder/encoder.json'), 'families': len(families), 'rows': len(vectors), 'report': report,
         'scope': 'Latin queries of the open families (never trained on) and of catalog validation, searched against every shipped catalog as one, exact family, no twin credit.'})
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['plan', 'evaluate']); p.add_argument('--run', default='student-30k'); a = p.parse_args()
    plan() if a.command == 'plan' else evaluate(a.run)
