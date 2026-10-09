"""A line reader: the characters of one line of Latin text, for the text the style match can use (a second stage that
renders the shortlist in the crop's own words) and for the person to correct. Tiny by design: `read-24-48-96-160-v1`,
about 190,000 weights, trained on the style stream's own views, words rendered in every collected font on paper and on
photographed surfaces (train.style_data), each prepared as one strip 48 pixels tall (src/line.mjs, sampler `line`).
Connectionist temporal classification over the strip's columns, eight pixels each, decoded greedily. The classes are
space, digits and the Latin letters; an accented letter reads as its base letter. Faces of held-out and test families
never train it: they are its development set, so the character error is that on fonts it has not seen.

    node scripts/python.mjs -m train.read train --run reader-20k --steps 20000 --workers 6 --photo 0.5
    node scripts/python.mjs -m train.read export --run reader-20k --bits 6      # .data/read/<run>/reader.json
"""
import argparse
import base64
import json
import random
import time
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.corpus import ROOT
from train.encoder_data import save, SPLIT
from train.robustness import read
from train.style_data import Stream, pools as make_pools
from train.style_fonts import training_families, open_role, designs
from train.style_teacher import glyph_sets, bound, OUT as STYLE
from train.ten_model import pack_bits, shortest

OUT = ROOT/'.data/read'
ARCH = 'read-24-48-96-160-v1'
CLASSES = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # index 0 is blank; a class is index + 1
HEIGHT = 48
STRIDE = 8        # strip pixels per output column: three stride-2 blocks
MAX_WIDTH = 1024  # as prepareStrip caps a strip
BUCKET = 64       # strip widths are padded to a multiple of this: stable shapes for the device
DEVELOPMENT = ('test', 'held-out')  # roles whose faces the reader never trains on


def fold(text):
    """The label of a rendered text: letters without their accents, None when a character has no class (ß, a symbol)."""
    out = []
    for c in text:
        base = next((b for b in unicodedata.normalize('NFKD', c) if not unicodedata.combining(b)), c)
        if base not in CLASSES: return None
        out.append(base)
    return ''.join(out)


def encode(text):
    return [CLASSES.index(c) + 1 for c in text]


def decode(indices):
    """Greedy CTC: repeats collapse, blanks drop."""
    out = []; last = 0
    for i in indices:
        if i and i != last: out.append(CLASSES[i - 1])
        last = i
    return ''.join(out)


def distance(a, b):
    """Levenshtein distance between two strings."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1): new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


class Reader(nn.Module):
    """Three stride-2 blocks (a dense 3×3, then two depthwise 3×3 + pointwise), the six remaining rows folded into
    channels, a pointwise mix, three dilated context taps along the line (1×3, dilation 1, 2, 4) each with a pointwise,
    and a pointwise head per column. Columns past a strip's own width are masked to zero after every layer, as the
    browser never computes them."""
    CHANNELS = [1, 24, 48, 96]; CONTEXT = 160; DILATIONS = [1, 2, 4]

    def __init__(self, training=True):
        super().__init__()
        c = self.CHANNELS; rows = HEIGHT//STRIDE
        self.blocks = nn.ModuleList([nn.Conv2d(c[0], c[1], 3, stride=2, padding=1),
                                     nn.Conv2d(c[1], c[1], 3, stride=2, padding=1, groups=c[1]), nn.Conv2d(c[1], c[2], 1),
                                     nn.Conv2d(c[2], c[2], 3, stride=2, padding=1, groups=c[2]), nn.Conv2d(c[2], c[3], 1)])
        self.mix = nn.Conv2d(c[3]*rows, self.CONTEXT, 1)
        self.context = nn.ModuleList(sum([[nn.Conv2d(self.CONTEXT, self.CONTEXT, (1, 3), padding=(0, d), dilation=(1, d), groups=self.CONTEXT), nn.Conv2d(self.CONTEXT, self.CONTEXT, 1)] for d in self.DILATIONS], []))
        self.head = nn.Conv2d(self.CONTEXT, len(CLASSES) + 1, 1)
        norm = lambda layer: nn.BatchNorm2d(layer.out_channels) if training else nn.Identity()
        self.norms = nn.ModuleList([norm(l) for l in [*self.blocks, self.mix, *self.context]])

    def layers(self):
        return [*self.blocks, self.mix, *self.context]

    def forward(self, strips, widths):
        """strips: (B, 1, 48, W) with ink as 1; widths: (B,) the strips' own widths. Returns (B, T, classes) logits, T = ceil(W/8), and each strip's column count."""
        x = strips; columns = widths
        for i, (layer, norm) in enumerate(zip(self.layers(), self.norms)):
            if layer is self.mix: x = x.flatten(1, 2)[:, :, None]  # (B, C·rows, 1, T): the rows become channels
            x = F.relu(norm(layer(x)))
            if i < 5 and layer.stride[1] == 2: columns = (columns + 1)//2
            mask = torch.arange(x.shape[3], device=x.device)[None] < columns[:, None]
            x = x*mask[:, None, None]
        return self.head(x)[:, :, 0].transpose(1, 2), columns

    def folded(self):
        """The same function with batch norms fused into their layers, for export."""
        self.eval(); result = Reader(training=False).eval()
        for source, norm, target in zip(self.layers(), self.norms, result.layers()):
            fused = nn.utils.fuse_conv_bn_eval(source, norm) if isinstance(norm, nn.BatchNorm2d) else source
            target.load_state_dict(fused.state_dict())
        result.head.load_state_dict(self.head.state_dict())
        return result


def strips(views, device):
    """The views' strips in one array, ink as 1, widths padded to a bucket; with each strip's width and label."""
    widths = np.array([v['windows'][0].shape[1] for v in views]); padded = min(MAX_WIDTH, -(-int(widths.max())//BUCKET)*BUCKET)
    array = np.zeros((len(views), 1, HEIGHT, padded), np.float32)
    for n, v in enumerate(views): s = v['windows'][0]; array[n, 0, :s.shape[0], :s.shape[1]] = 1 - s/np.float32(255)
    return torch.from_numpy(array).to(device), torch.from_numpy(widths).to(device)


def faces_by_role():
    """The teacher's face list split for the reader: faces of training families, and faces of the development ones."""
    families = training_families(); split = read(SPLIT)['families']; design = designs(families)
    role = {f['id']: split.get(f['id']) or open_role(f['id'], design.get(f['id'])) for f in families}
    faces = read(STYLE/'faces.json')['faces']; pools = make_pools(families, glyph_sets(families))
    latin = [i for i, f in enumerate(faces) if 'Latn' in pools.get(f['family'], {})]
    train = [i for i in latin if role[faces[i]['family']] not in DEVELOPMENT]; development = [i for i in latin if role[faces[i]['family']] in DEVELOPMENT]
    return faces, pools, train, development


def plan_for(indices, seed, batch, clean=.15):
    rng = random.Random(seed)
    return lambda: [(rng.choice(indices), 'Latn', rng.getrandbits(31), rng.random() < clean, None) for _ in range(batch)]


def labelled(views):
    out = []
    for v in views:
        label = fold(v['text'])
        if label and v['windows'] and v['windows'][0].shape[1] <= MAX_WIDTH: out.append({**v, 'label': label})
    return out


def development_views(faces, pools, indices, workers, photo, count=768, seed=7):
    """A fixed development set: `count` views of the development faces, half clean, half damaged as training views are."""
    stream = Stream(faces, pools, plan_for(indices, seed, 64, clean=.5), workers=workers, prefetch=2, photo=photo, sampler='line'); views = []
    while len(views) < count: views += labelled(next(stream))
    stream.close(); return views[:count]


def evaluate(model, views, device, batch=64):
    """Character error rate and whole-line accuracy, over clean and damaged views apart."""
    model.eval(); results = {'clean': [], 'damaged': []}
    with torch.no_grad():
        for start in range(0, len(views), batch):
            chunk = views[start:start + batch]; x, widths = strips(chunk, device); logits, columns = model(x, widths)
            best = logits.argmax(2).cpu().numpy(); columns = columns.cpu().numpy()
            for v, row, t in zip(chunk, best, columns):
                guess = decode(row[:t]); results['damaged' if v['damaged'] else 'clean'].append((distance(guess, v['label']), len(v['label']), guess == v['label']))
    model.train()
    return {kind: {'cer': round(sum(d for d, _, _ in r)/max(1, sum(n for _, n, _ in r)), 4), 'exact': round(sum(e for _, _, e in r)/max(1, len(r)), 4), 'lines': len(r)} for kind, r in results.items()}


def train(run, steps, workers, check, lr, photo, batch=48, seed=0, device='mps'):
    bound(8); torch.manual_seed(seed)
    faces, pools, train_faces, development_faces = faces_by_role()
    print('Reader faces', len(train_faces), 'training,', len(development_faces), 'development', flush=True)
    development = development_views(faces, pools, development_faces, workers, photo)
    print('Development views', len(development), flush=True)
    stream = Stream(faces, pools, plan_for(train_faces, seed, batch), workers=workers, photo=photo, sampler='line')
    model = Reader().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=.05, final_div_factor=20)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True); folder = OUT/run; folder.mkdir(parents=True, exist_ok=True)
    best = None; progress = []; started = time.time(); recent = []
    for step in range(1, steps + 1):
        views = labelled(next(stream))
        if not views: continue
        x, widths = strips(views, device); logits, columns = model(x, widths)
        targets = torch.tensor(sum((encode(v['label']) for v in views), []), dtype=torch.long)
        lengths = torch.tensor([len(v['label']) for v in views], dtype=torch.long)
        loss = ctc(F.log_softmax(logits, 2).transpose(0, 1).cpu(), targets, columns.cpu(), lengths)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step(); schedule.step()
        recent.append(loss.item())
        if step % check == 0 or step == steps:
            result = evaluate(model, development, device); mean = sum(recent)/len(recent); recent = []
            entry = {'step': step, 'loss': round(mean, 4), **result, 'seconds': round(time.time() - started)}; progress.append(entry)
            print(f"check {step}: loss {mean:.3f} clean cer {result['clean']['cer']} exact {result['clean']['exact']} damaged cer {result['damaged']['cer']} exact {result['damaged']['exact']} ({entry['seconds']}s)", flush=True)
            score = result['damaged']['cer'] + result['clean']['cer']
            if best is None or score < best: best = score; torch.save(model.state_dict(), folder/'best.pt')
            save(folder/'progress.json', {'architecture': ARCH, 'steps': steps, 'lr': lr, 'photo': photo, 'batch': batch, 'checks': progress, 'best': best})
    stream.close()


def artifact_of(model, bits=6):
    """The folded reader as the browser reads it (src/reader.mjs): every weighted layer with its geometry in order, weights
    rounded per output row to `bits`; `fold` marks the layer before which the rows become channels, `head` the last. Returns
    the artifact and the folded model with the rounded weights, the function the browser computes."""
    model = model.folded(); levels = 2**(bits - 1) - 1; layers = []
    for layer in [*model.layers(), model.head]:
        weight = layer.weight.detach().numpy(); row = weight.reshape(len(weight), -1)
        scale = np.maximum(np.max(np.abs(row), axis=1)/levels, 1e-12).astype(np.float32); quant = np.clip(np.round(row/scale[:, None]), -levels, levels).astype(np.int8)
        with torch.no_grad(): layer.weight.copy_(torch.from_numpy((quant.astype(np.float32)*scale[:, None]).reshape(weight.shape)))
        layers.append({'kind': 'head' if layer is model.head else 'fold' if layer is model.mix else 'layer', 'shape': list(weight.shape), 'stride': list(layer.stride), 'dilation': list(layer.dilation),
                       'groups': layer.groups, 'scale': shortest(scale), 'bias': shortest(layer.bias.detach()), 'bits': bits, 'weights': base64.b64encode(pack_bits(quant, bits)).decode()})
    artifact = {'version': 1, 'kind': 'line-reader', 'architecture': ARCH, 'classes': CLASSES, 'preparation': {'height': HEIGHT, 'maxWidth': MAX_WIDTH, 'method': 'line', 'stride': STRIDE}, 'layers': layers}
    return artifact, model


def export(run, bits=6):
    model = Reader(); model.load_state_dict(torch.load(OUT/run/'best.pt', map_location='cpu')); artifact, _ = artifact_of(model, bits)
    save(OUT/run/'reader.json', artifact); print('Reader', sum(int(np.prod(l['shape'])) for l in artifact['layers']), 'weights,', (OUT/run/'reader.json').stat().st_size, 'bytes', flush=True)
    return artifact


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['train', 'export']); p.add_argument('--run', default='reader')
    p.add_argument('--steps', type=int, default=20000); p.add_argument('--workers', type=int, default=6); p.add_argument('--check', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-3); p.add_argument('--photo', type=float, default=0.5); p.add_argument('--bits', type=int, default=6); a = p.parse_args()
    if a.command == 'train': train(a.run, a.steps, a.workers, a.check, a.lr, a.photo)
    else: export(a.run, a.bits)
