"""The DaFont forum's identified requests as training views: the picture someone asked about, the line the forum's answer
names, labelled by the font the thread settled on when it is a training family, Google's or an open one, under a name a
free-font page carries (a commercial page's name may be shared by an open font of another design). The lines are
scripts/forum_boxes.py's (macOS Vision); the line taken is the one sharing most words with the text the answerer
previewed the font with (the link's text=), else the tallest confident horizontal one, cut with a little background
around it and prepared as the page prepares a crop. Threads split by the hash of their id, one in ten held out as the
forum read (`thread_role`); Max Halford's benchmark threads (scripts/dafont.mjs) stay out as well.

    node scripts/python.mjs -m train.style_forum          # .data/style/forum-development.json and .u8
"""
import base64
import hashlib
import json
import re
import subprocess
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from scripts.corpus import ROOT
from train.encoder_data import save, validate_shard
from train.robustness import sha
from train.style_bench import OUT, PINS
from train.style_screens import HELD_OUT, STYLE, family_of, normalize, script_of, training_families

FORUM = ROOT/'.data/forum/dafont'
BENCH = ROOT/'.data/dafont/tasks.json'
# Pages whose font is the file anyone downloads, so a training family of that name is that font.
FREE_HOSTS = {'fonts.google.com', 'www.dafont.com', 'www.fontsquirrel.com', 'www.1001fonts.com', 'www.fontspace.com', 'fontlibrary.org', 'www.abstractfonts.com', 'github.com'}
WEIGHTS = (('extralight', 200), ('ultralight', 200), ('semibold', 600), ('demibold', 600), ('extrabold', 800), ('ultrabold', 800), ('hairline', 100), ('thin', 100), ('light', 300),
           ('medium', 500), ('demi', 600), ('bold', 700), ('black', 900), ('heavy', 900))
CONFIDENT = .5
PAD = .25  # of the line's height, around it


def thread_role(thread):
    return 'held-out' if int(hashlib.sha1(str(thread).encode()).hexdigest()[:8], 16) % HELD_OUT == 0 else 'development'


def words(text):
    return set(re.findall(r'[^\W_]+', text.lower()))


def style_of(name):
    """The weight and posture the answer's style words give ('Bebas Neue Bold', 'Lato Light Italic'); regular upright without."""
    tail = normalize(name)[len(normalize(STYLE.sub('', name))):]
    return next((w for word, w in WEIGHTS if word in tail), 400), 'italic' in tail or 'oblique' in tail


def line_of(lines, hint=''):
    """The boxed line the answer is about: among the confident horizontal ones, the line sharing most words with the text
    the answerer previewed the font with, else the tallest; None when no line qualifies."""
    lines = [l for l in lines if l['confidence'] >= CONFIDENT and l['box'][2] - l['box'][0] >= l['box'][3] - l['box'][1] > 0 and words(l['text'])]
    if not lines: return None
    shared = [(len(words(l['text']) & words(hint)), l) for l in lines]; best = max(shared, key=lambda s: s[0])
    return best[1] if best[0] else max(lines, key=lambda l: l['box'][3] - l['box'][1])


def chosen(requests, boxes, families, bench=frozenset()):
    """Development threads whose answer names a training family on a free-font page, with the line, face and script they label."""
    out = []
    for r in requests:
        if not r.get('identified') or not r.get('files') or r['id'] in bench or thread_role(r['id']) != 'development': continue
        answer = r['identified'][0]; url = urlsplit(answer.get('url') or ''); family = family_of(answer.get('name'), families)
        found = boxes.get(r['files'][0]['file'])
        if url.netloc not in FREE_HOSTS or not family or not found or not found.get('lines'): continue
        line = line_of(found['lines'], parse_qs(url.query).get('text', [''])[0]); script = script_of(line['text']) if line else None
        if not script: continue
        out.append((r, line, family, script, *style_of(answer['name']), url.netloc))
    return out


def crop(image, box):
    x0, y0, x1, y1 = box; pad = round((y1 - y0)*PAD)
    return image.crop((max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pack():
    if not BENCH.exists(): raise SystemExit(f'{BENCH} is missing: without the benchmark threads the forum read would leak into training (scripts/dafont.mjs fetches it)')
    requests = rows(FORUM/'requests.jsonl'); boxes = {b['file']: b for b in rows(FORUM/'boxes.jsonl')}; bench = {t['task_id'] for t in json.loads(BENCH.read_text())}
    picked = chosen(requests, boxes, training_families(), bench)
    samples = []; windows = []; chunks = []; offset = 0
    for start in range(0, len(picked), 512):
        batch = picked[start:start + 512]; images = []
        for r, line, *_ in batch:
            image = crop(Image.open(FORUM/'images'/r['files'][0]['file']).convert('L'), line['box'])
            images.append({'width': image.width, 'height': image.height, 'pixels': base64.b64encode(image.tobytes()).decode()})
        out = subprocess.run(['node', str(ROOT/'scripts/corpus-prepare.mjs')], input=json.dumps(images), text=True, capture_output=True, check=True)
        for k, items in enumerate(json.loads(out.stdout)):
            if not items: continue
            r, line, family, script, weight, italic, host = batch[k]; source = len(samples); text = line['text']
            samples.append({'role': 'development', 'family': family, 'font': r['identified'][0]['name'], 'weight': weight, 'italic': italic, 'script': script, 'case': 'upper' if text.isupper() else 'lower',
                            'text': text, 'length': len(text), 'size': line['box'][3] - line['box'][1], 'condition': 'forum', 'thread': r['id'], 'host': host})
            for w in items:
                raw = base64.b64decode(w['pixels']); windows.append({'source': source, 'offset': offset, 'width': w['width'], 'height': w['height']}); chunks.append(raw); offset += len(raw)
    pixels = b''.join(chunks); (OUT/'forum-development.u8').write_bytes(pixels)
    manifest = {'pins': {p: sha(ROOT/p) for p in PINS}, 'set': {'name': 'DaFont forum requests', 'collector': 'scripts/dafont_forum.py', 'requests': len(requests), 'boxed': len(boxes), 'requestsSha256': sha(FORUM/'requests.jsonl')},
                'sha256': sha(OUT/'forum-development.u8'), 'samples': samples, 'windows': windows,
                'scope': 'The line the forum\'s answer names in each development thread\'s picture (nine in ten by thread hash, the benchmark threads apart), labelled by the answer when it is a training family on a free-font page; the held-out threads stay the forum read (bench/photos.md).'}
    validate_shard(manifest, len(pixels)); save(OUT/'forum-development.json', manifest)
    print('Forum', len(samples), 'of', len(picked), 'lines prepared from', len(requests), 'requests,', len(windows), 'windows', flush=True)


if __name__ == '__main__': pack()
