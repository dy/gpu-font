"""Web screens as training views: the crops scripts/web_screens.mjs cuts from the top sites' homepages, each labelled by
the font file Chromium drew it with, prepared as the page prepares a crop. Only the development sites are packed: a site
is held out by the hash of its origin, one in ten (`site_role`, the rule scripts/screens.mjs reads by), so the screens
read stays a read. A crop counts when its font's family is a training family, Google's or an open one, in a script the
teacher knows; the face is the family's face of the drawn weight and posture when it has one.

    node scripts/python.mjs -m train.style_screens          # .data/style/screens-development.json and .u8
"""
import base64
import hashlib
import json
import re
import subprocess
import unicodedata

from PIL import Image

from scripts.corpus import ROOT
from train.encoder_data import save, validate_shard
from train.robustness import read, sha
from train.style_bench import OUT, PINS
from train.style_fonts import google_families, open_families

SCREENS = ROOT/'.data/screens'
HELD_OUT = 10
# What a font file adds to its family's name: a weight, a width, a slant, an optical size; scripts/screens.mjs's STYLE.
STYLE = re.compile(r'(\s+(thin|hairline|extra\s?light|ultra\s?light|light|regular|normal|book|medium|semi\s?bold|demi\s?bold|demi|bold|extra\s?bold|ultra\s?bold|black|heavy|italic|oblique|condensed|narrow|\d+pt|\d{3}))+$', re.I)
SCRIPTS = {'LATIN': 'Latn', 'CYRILLIC': 'Cyrl', 'GREEK': 'Grek', 'ARABIC': 'Arab', 'HEBREW': 'Hebr', 'DEVANAGARI': 'Deva', 'THAI': 'Thai',
           'HANGUL': 'Hang', 'HIRAGANA': 'Hira', 'KATAKANA': 'Kana', 'CJK': 'Hani'}


def normalize(name):
    return ''.join(c for c in name.lower() if c.isalnum())


def site_role(origin):
    return 'held-out' if int(hashlib.sha1(origin.encode()).hexdigest()[:8], 16) % HELD_OUT == 0 else 'development'


def family_of(name, families):
    """The training family a drawn font belongs to: by its own name, else by the name without its style words; None otherwise."""
    if not name or not name.strip() or re.search(r'[\x00-\x1f\x7f]', name): return None
    return families.get(normalize(name)) or families.get(normalize(STYLE.sub('', name)))


def script_of(text):
    """The script most of the text's letters are in, by their Unicode names; None when it is not one the views use."""
    counts = {}
    for c in text:
        if not c.isalpha(): continue
        word = unicodedata.name(c, '').split(' ')[0]
        counts[word] = counts.get(word, 0) + 1
    if not counts: return None
    return SCRIPTS.get(max(counts, key=counts.get))


def training_families():
    """Family id by normalized name, Google's first so a name in both keeps its Google identity."""
    out = {}
    for f in google_families() + open_families(): out.setdefault(normalize(f['family']), f['id'])
    return out


def chosen(records, families):
    """Development-site records whose font names a training family in a known script, with the face they label."""
    out = []
    for r in records:
        if site_role(r['origin']) != 'development': continue
        family = family_of(r['font'].get('family'), families); script = script_of(r['text'])
        if not family or not script: continue
        weight = re.match(r'\d+', str(r['css'].get('fontWeight', '400'))); weight = min(900, max(100, round(int(weight.group())/100)*100)) if weight else 400
        out.append((r, family, script, weight, r['css'].get('fontStyle') in ('italic', 'oblique')))
    return out


def pack(count=None):
    """Pack the first `count` records (every record by default): the snapshot is the file's prefix, named by count and hash."""
    lines = (SCREENS/'records.jsonl').read_text().splitlines(); lines = lines[:count] if count else lines
    records = [json.loads(line) for line in lines if line.strip()]; families = training_families()
    picked = chosen(records, families)
    images = []
    for r, *_ in picked:
        image = Image.open(SCREENS/r['image']).convert('L'); images.append({'width': image.width, 'height': image.height, 'pixels': base64.b64encode(image.tobytes()).decode()})
    samples = []; windows = []; chunks = []; offset = 0
    for start in range(0, len(images), 512):
        out = subprocess.run(['node', str(ROOT/'scripts/corpus-prepare.mjs')], input=json.dumps(images[start:start + 512]), text=True, capture_output=True, check=True)
        for k, items in enumerate(json.loads(out.stdout)):
            if not items: continue
            r, family, script, weight, italic = picked[start + k]; source = len(samples)
            samples.append({'role': 'development', 'family': family, 'font': r['font']['family'], 'weight': weight, 'italic': italic, 'script': script,
                            'case': 'upper' if r['text'].isupper() else 'lower', 'text': r['text'], 'length': len(r['text']), 'size': r['css']['fontSize'], 'condition': 'screen', 'origin': r['origin']})
            for w in items:
                raw = base64.b64decode(w['pixels']); windows.append({'source': source, 'offset': offset, 'width': w['width'], 'height': w['height']}); chunks.append(raw); offset += len(raw)
    pixels = b''.join(chunks); (OUT/'screens-development.u8').write_bytes(pixels)
    snapshot = hashlib.sha256('\n'.join(lines).encode()).hexdigest()
    manifest = {'pins': {p: sha(ROOT/p) for p in PINS}, 'set': {'name': 'Web screens', 'collector': 'scripts/web_screens.mjs', 'records': len(lines), 'recordsSha256': snapshot},
                'sha256': sha(OUT/'screens-development.u8'), 'samples': samples, 'windows': windows,
                'scope': 'Lines from the development sites\' homepages (nine in ten by origin hash), labelled by the font Chromium drew them with, cut by the collector and prepared as the page prepares a crop; the held-out sites stay the read (bench/photos.md).'}
    validate_shard(manifest, len(pixels)); save(OUT/'screens-development.json', manifest)
    print('Screens', len(samples), 'of', len(picked), 'crops prepared from', len(records), 'records,', len(windows), 'windows', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--count', type=int, help='records to take from the head of records.jsonl'); a = p.parse_args()
    pack(a.count)
