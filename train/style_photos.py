"""Photographs as a style benchmark: WhatFontIs-Bench (CC BY 4.0), one painted or printed word per picture, cut at the set's
own crop box and prepared as the page prepares a crop. Only Google Fonts families count here, since the Python evaluation
searches the Google references. Two roles, as bench/photos.md reads them: development (training and development families,
for tuning) and final (held-out families, read once per released model).
"""
import argparse
import base64
import json
import random
import subprocess

import numpy as np
from PIL import Image

from scripts.corpus import ROOT
from train.encoder_data import save, validate_shard
from train.robustness import read, sha
from train.style_bench import OUT, PINS

SET = ROOT/'.data/whatfontis/WhatFontIs-Bench/v1'
WEIGHTS = {'Thin': 100, 'ExtraLight': 200, 'Light': 300, 'Regular': 400, 'Medium': 500, 'SemiBold': 600, 'Bold': 700, 'ExtraBold': 800, 'Black': 900}


def normalize(name):
    return ''.join(c for c in name.lower() if c.isalnum())


def family_of(name):
    """The set names a font with its style ('Heebo regular', 'Spartan 500'); the family is what is left without them."""
    words = name.split()
    while words and (words[-1].lower() in ('regular', 'book', 'medium', 'roman') or (words[-1].isdigit() and len(words[-1]) == 3)): words.pop()
    return ' '.join(words)


def weight_of(title):
    """'Asap 600', 'Lora Bold Italic' or 'Alef' → the CSS weight and posture."""
    words = title.split(); weight = 400; italic = 'Italic' in words
    for word in words:
        if word.isdigit(): weight = int(word)
        elif word in WEIGHTS: weight = WEIGHTS[word]
    return weight, italic


ABSENT = 3000  # pictures of fonts no shipped catalog holds: what a rejection rule must turn away


def pack(role):
    """`development` and `final`: pictures of Google families by the split. `absent`: a fixed sample of pictures whose
    font is in no shipped catalog (bench/whatfontis.json lists the indexed pictures), for calibrating rejection."""
    families = {normalize(f['family']): f['id'] for f in read(ROOT/'bench/corpus.json')['families'] if not f['excluded']}
    split = read(ROOT/'bench/encoder-split.json')['families']
    labels = [json.loads(line) for line in (SET/'labels.jsonl').read_text().splitlines() if line.strip()]
    chosen = []
    if role == 'absent':
        indexed = {image['id'] for image in read(ROOT/'bench/whatfontis.json')['images']}
        absent = [label for label in labels if label['id'] not in indexed]
        chosen = [(label, None) for label in random.Random(20260924).sample(absent, ABSENT)]
    for label in labels if role != 'absent' else []:
        family = families.get(normalize(family_of(label['family'])))
        if family is None or split.get(family) is None: continue
        if (split[family] == 'test') != (role == 'final'): continue
        chosen.append((label, family))
    images = []
    for label, _ in chosen:
        x0, y0, x1, y1 = label['crop_box']
        image = Image.open(SET/label['image']).convert('L').crop((x0, y0, x1, y1))
        images.append({'width': image.width, 'height': image.height, 'pixels': base64.b64encode(image.tobytes()).decode()})
    samples = []; windows = []; chunks = []; offset = 0
    for start in range(0, len(images), 512):
        out = subprocess.run(['node', str(ROOT/'scripts/corpus-prepare.mjs')], input=json.dumps(images[start:start + 512]), text=True, capture_output=True, check=True)
        for k, items in enumerate(json.loads(out.stdout)):
            if not items: continue
            label, family = chosen[start + k]; weight, italic = weight_of(label['title'])
            source = len(samples)
            samples.append({'role': role, 'family': family, 'font': label['family'], 'weight': weight, 'italic': italic, 'script': 'Latn', 'case': 'upper' if label['text'].isupper() else 'lower',
                            'text': label['text'], 'length': len(label['text']), 'size': label['font_px'], 'condition': 'photo', 'kind': label['type'],
                            'difficulty': label['difficulty'], 'image': label['id']})
            for w in items:
                raw = base64.b64decode(w['pixels']); windows.append({'source': source, 'offset': offset, 'width': w['width'], 'height': w['height']}); chunks.append(raw); offset += len(raw)
    pixels = b''.join(chunks); (OUT/f'photos-{role}.u8').write_bytes(pixels)
    scope = {'absent': f'WhatFontIs-Bench photographs of {ABSENT} fonts in no shipped catalog: what rejection must turn away.'}.get(role, f'WhatFontIs-Bench photographs of Google Fonts families, {role} role (bench/photos.md), cut at the crop box the set gives. Real surfaces and light; one word each.')
    manifest = {'pins': {p: sha(ROOT/p) for p in PINS}, 'set': {'name': 'WhatFontIs-Bench v1.0', 'labelsSha256': sha(SET/'labels.jsonl')}, 'sha256': sha(OUT/f'photos-{role}.u8'),
                'samples': samples, 'windows': windows, 'scope': scope}
    validate_shard(manifest, len(pixels)); save(OUT/f'photos-{role}.json', manifest)
    print('Photos', role, len(samples), 'of', len(chosen), 'pictures prepared,', len(windows), 'windows', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('role', choices=['development', 'final', 'absent'], nargs='*', default=['development', 'final', 'absent']); a = p.parse_args()
    for role in a.role: pack(role)
