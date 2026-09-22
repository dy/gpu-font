"""Independent PyTorch outputs for the deployed variable-shape int8 network."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from fontTools.ttLib import TTFont
from train.ten_model import load_export
from train import ten, hundred

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
model_bytes = (root / 'dist/assets/model.json').read_bytes()
artifact = json.loads(model_bytes)
for family in artifact['fonts']:
    with TTFont(root / f'.data/fonts/{family}.ttf') as face:
        if not (set(range(32, 127)) - {ord('^'), ord('~')}) <= face.getBestCmap().keys():
            raise ValueError(f'Incomplete preview glyph coverage: {family}')
model = load_export(artifact)
cases = [
    ('white-min', np.ones((1, 1), dtype=np.float32)),
    ('black-max', np.zeros((48, 128), dtype=np.float32)),
    ('ramp-odd', (np.arange(47 * 127).reshape(47, 127) % 113 / 112).astype(np.float32)),
    ('stripes-small', (np.arange(21).reshape(3, 7) % 2).astype(np.float32)),
]
_, manifest, pixels = (hundred if len(artifact['fonts']) == 100 else ten).load_data()
for renderer in ['pillow', 'chromium']:
    w = next(w for w in manifest['windows'] if (s := manifest['samples'][w['source']])['renderer'] == renderer and s['role'] == ('train' if renderer == 'pillow' else 'validation') and s['family'] == 'inter')
    cases.append((renderer, (pixels[w['offset']:w['offset'] + w['width'] * w['height']].astype(np.float32) / 255).reshape(w['height'], w['width'])))
result = []
for name, pixels in cases:
    with torch.no_grad():
        logits = model(torch.from_numpy(pixels[None, None]))[0].tolist()
    result.append({'name': name, 'width': pixels.shape[1], 'height': pixels.shape[0], 'pixels': pixels.flatten().tolist(), 'logits': logits})
(root / '.data/demo-reference.json').write_text(json.dumps({'modelSha256': hashlib.sha256(model_bytes).hexdigest(), 'cases': result}))
print(f'Exported {len(result)} PyTorch parity cases')
