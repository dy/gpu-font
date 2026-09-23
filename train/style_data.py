"""On-the-fly training views: fresh random text in any face and script, screenshot damage, browser preparation.

Every view is generated from (face, script, seed); nothing is stored, so the text never repeats. Pixels pass
through the browser's own prepareLine (scripts/style-prepare.mjs), identical to inference.
"""
from collections import Counter, OrderedDict
import io
import multiprocessing
import random
import struct
import subprocess
import threading
import queue

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from fontTools.ttLib import TTFont

from scripts.corpus import ROOT, CACHE

SIZES = [14, 16, 18, 20, 24, 28, 32, 40, 48, 56, 64, 72]
SPACED = {'Latn', 'Cyrl', 'Grek', 'Arab', 'Hebr', 'Deva', 'Beng', 'Gujr', 'Guru', 'Knda', 'Mlym', 'Taml', 'Telu', 'Sinh', 'Copt', 'Armn', 'Geor'}
CASED = {'Latn', 'Cyrl', 'Grek'}
SIMPLE = CASED | {'Hani', 'Hira', 'Kana', 'Hang', 'Bopo'}  # no shaping: letter-spacing by per-glyph placement is safe
# English letter frequencies, so frequent shapes dominate as on real pages; uniform draws keep rare letters present.
FREQUENT = 'eeeeeeeeeeeettttttttaaaaaaaooooooiiiiiinnnnnnsssssshhhhhhrrrrrdddlllcccuuummwwffggyyppbbvkjxqz'


class Preparer:
    """The browser's prepareLine in a persistent Node child."""
    def __init__(self):
        self.child = subprocess.Popen(['node', str(ROOT/'scripts/style-prepare.mjs')], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def read(self, n):
        data = self.child.stdout.read(n)
        if len(data) != n: raise RuntimeError('Preparation child failed')
        return data

    def __call__(self, gray):
        h, w = gray.shape
        if not h or not w or h*w > 16_777_216: raise ValueError('Image outside the preparation limits')
        try:
            self.child.stdin.write(struct.pack('<II', w, h) + np.ascontiguousarray(gray, np.uint8).tobytes()); self.child.stdin.flush()
            windows = []
            for _ in range(struct.unpack('<I', self.read(4))[0]):
                ww, hh = struct.unpack('<II', self.read(8)); windows.append(np.frombuffer(self.read(ww*hh), np.uint8).reshape(hh, ww))
            return windows
        except (RuntimeError, BrokenPipeError, OSError):
            # A dead child loses this view only: restart it and let the caller retry another seed.
            self.child.kill(); self.__init__(); raise ValueError('Preparation child restarted')

    def close(self):
        self.child.stdin.close(); self.child.wait()


class Fonts:
    """Bounded cache of sized, instanced FreeType faces. Faces whose hinting program fails are reloaded unhinted."""
    def __init__(self, limit=48): self.cache = OrderedDict(); self.limit = limit; self.order = {}; self.unhinted = {}

    def broken(self, face):
        from train.corpus_data import without_hints
        self.unhinted[face['path']] = without_hints(CACHE/face['path'])
        for key in [k for k in self.cache if k[0] == face['id']]: del self.cache[key]

    def render(self, face, size, draw):
        """draw(font) with the hinted face, or once more unhinted if its hinting program overflows."""
        try: return draw(self(face, size))
        except OSError as error:
            if 'execution context too long' not in str(error) or face['path'] in self.unhinted: raise
            self.broken(face); return draw(self(face, size))

    def __call__(self, face, size):
        key = (face['id'], size)
        if key in self.cache: self.cache.move_to_end(key); return self.cache[key]
        source = io.BytesIO(self.unhinted[face['path']]) if face['path'] in self.unhinted else str(CACHE/face['path'])
        font = ImageFont.truetype(source, size, layout_engine=ImageFont.Layout.RAQM)
        if face['axes']:
            if face['path'] not in self.order:
                with TTFont(CACHE/face['path'], lazy=True) as tt: self.order[face['path']] = [a.axisTag for a in tt['fvar'].axes]
            font.set_variation_by_axes([face['axes'].get(tag, 0) for tag in self.order[face['path']]])
        self.cache[key] = font
        if len(self.cache) > self.limit: self.cache.popitem(last=False)
        return font


def text(rng, script, pool, case=None):
    """Random words from the face's own characters: 1-24 characters, all cases, digits sometimes. `case` (lower, upper
    or title) fixes a cased script's case, so two views of one face can be planned in opposite cases."""
    length = rng.choice([1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 12, 14, 16, 18, 20, 24])
    if script in CASED:
        letters = [c for c in pool if c.isalpha()]; lower = sorted({c.lower() for c in letters if c.lower() in pool and c.upper() in pool}) or letters
        draw = rng.random(); plain = [c for c in lower if c.isascii()] or lower
        source = [c for c in FREQUENT if c in lower] if script == 'Latn' and draw < .6 else plain if draw < .9 else lower
        chars = rng.choices(source, k=length)
        digits = [c for c in pool if c.isdigit()]
        if digits and rng.random() < .1: chars = [rng.choice(digits) if rng.random() < .5 else c for c in chars]
        value = ''.join(chars); mode = rng.random()
        if case: value = value.upper() if case == 'upper' else value[:1].upper() + value[1:] if case == 'title' else value
        elif mode < .2: value = value.upper()
        elif mode < .4: value = value[:1].upper() + value[1:]
        elif mode < .5: value = ''.join(c.upper() if rng.random() < .3 else c for c in value)
        value = ''.join(c if c in pool else c.lower() for c in value)
    else:
        value = ''.join(rng.choices(pool, k=length))
    if script in SPACED and length > 5:
        spaced = list(value); i = rng.randint(2, 8)
        while i < len(spaced) - 1: spaced[i] = ' '; i += rng.randint(3, 9)
        value = ''.join(spaced)
    return value


def render(font, value, rng, script):
    """Dark text on white; occasional letter-spacing by per-glyph placement for scripts without shaping."""
    size = font.size; pad = max(4, size // 3)
    if script in SIMPLE and rng.random() < .15:
        tracking = size * rng.uniform(-.03, .25); x = pad; boxes = []
        for c in value:
            boxes.append((x, c)); x += font.getlength(c) + tracking
        top = min(font.getbbox(c, anchor='ls')[1] for c in value); bottom = max(font.getbbox(c, anchor='ls')[3] for c in value)
        image = Image.new('L', (int(x + pad), int(bottom - top + 2*pad)), 255); draw = ImageDraw.Draw(image)
        for x, c in boxes: draw.text((x, pad - top), c, font=font, fill=0, anchor='ls')
        return image
    left, top, right, bottom = font.getbbox(value)
    if right <= left or bottom <= top: raise ValueError('Empty rendering')
    image = Image.new('L', (right - left + 2*pad, bottom - top + 2*pad), 255)
    ImageDraw.Draw(image).text((pad - left, pad - top), value, font=font, fill=0)
    return image


def skeleton(mask):
    """Zhang-Suen thinning: one-pixel centre lines that keep every stroke and its connections."""
    m = mask.copy()
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(m, 1); n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
            count = sum(x.astype(np.uint8) for x in n); ring = n + [n[0]]
            flips = sum(((~ring[i]) & ring[i + 1]).astype(np.uint8) for i in range(8))
            a, b = (n[0] & n[2] & n[4], n[2] & n[4] & n[6]) if step == 0 else (n[0] & n[2] & n[6], n[0] & n[4] & n[6])
            remove = m & (count >= 2) & (count <= 6) & (flips == 1) & ~a & ~b
            if remove.any(): m &= ~remove; changed = True
        if not changed: return m


def sketch(image, rng):
    """A hand drawing of the face's letterforms: centre lines redrawn with a round pen, a smooth wobble and a slight
    tilt. Stroke contrast and fine serifs fade, as in real drawings."""
    lines = skeleton(np.asarray(image) < 128); size = image.height
    pen = max(3, round(size*rng.uniform(.04, .09)) | 1)  # odd width, proportional to the text size
    drawn = Image.fromarray(np.uint8(lines)*255).filter(ImageFilter.MaxFilter(pen)).filter(ImageFilter.GaussianBlur(.6))
    a = np.asarray(drawn, np.float32); h, w = a.shape; r = np.random.default_rng(rng.getrandbits(32))
    # Smooth displacement: two low-frequency sines per axis.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32); amp = size*rng.uniform(.01, .035)
    dx = sum(amp/k*np.sin(2*np.pi*(ys/h*r.uniform(.5, 1.5) + xs/w*r.uniform(.5, 3)) + r.uniform(0, 6.3)) for k in (1, 2))
    dy = sum(amp/k*np.sin(2*np.pi*(xs/w*r.uniform(.5, 3) + ys/h*r.uniform(.5, 1.5)) + r.uniform(0, 6.3)) for k in (1, 2))
    x = np.clip(np.round(xs + dx), 0, w - 1).astype(int); y = np.clip(np.round(ys + dy), 0, h - 1).astype(int)
    return Image.fromarray(np.uint8(255 - a[y, x])).rotate(rng.uniform(-3, 3), Image.Resampling.BICUBIC, expand=True, fillcolor=255)


def damage(image, rng):
    """Screenshot conditions: colour/contrast/polarity, rotation, rescaling, blur, JPEG, noise, clipped edges."""
    a = np.asarray(image, np.float32)/255
    ink, paper = rng.uniform(0, .45), rng.uniform(.75, 1)
    if rng.random() < .3: ink, paper = paper, ink  # dark mode
    image = Image.fromarray(np.uint8(np.clip(paper + (ink - paper)*(1 - a), 0, 1)*255))
    fill = int(image.getpixel((0, 0)))
    if rng.random() < .15: image = image.rotate(rng.uniform(-4, 4), Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
    if rng.random() < .5:
        s = rng.uniform(.5, 1); w, h = image.size
        image = image.resize((max(1, round(w*s)), max(1, round(h*s))), Image.Resampling.BILINEAR if rng.random() < .5 else Image.Resampling.LANCZOS)
        if rng.random() < .4: image = image.resize((w, h), Image.Resampling.BILINEAR)
    if rng.random() < .2: image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.3, 1)))
    if rng.random() < .1:
        w, h = image.size; image = image.crop((0, 0, max(8, round(w*rng.uniform(.85, .97))), h))
    if rng.random() < .3:
        stream = io.BytesIO(); image.save(stream, 'JPEG', quality=rng.randint(35, 90)); image = Image.open(io.BytesIO(stream.getvalue())).convert('L')
    a = np.asarray(image, np.float32)
    if rng.random() < .1: a = a + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 8), a.shape)
    return np.uint8(np.clip(a, 0, 255))


COMMON = {'Latn':'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789àáâäçèéêëíîïñóôöùúûüßÀÁÂÄÇÈÉÊËÍÓÖÜ',
          'Cyrl':'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789',
          'Grek':'αβγδεζηθικλμνξοπρσςτυφχψωάέήίόύώΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ0123456789'}


def pools(families, sets):
    """Characters a view may use: common ones per script, within each family's coverage. Rare code points
    (fullwidth forms, phonetic extensions) would teach glyphs pages almost never show."""
    counts = {'Hani': Counter(), 'Hang': Counter()}
    for f in families:
        for script, count in counts.items():
            if script in f['alphabets']: count.update(set(f['alphabets'][script]))
    common = dict(COMMON)
    for script, limit in [('Hani', 3000), ('Hang', 2000)]:
        ranked = sorted(((c, n) for c, n in counts.get(script, {}).items() if script != 'Hang' or '가' <= c <= '힣'), key=lambda v: (-v[1], v[0]))
        common[script] = ''.join(c for c, _ in ranked[:limit])
    result = {}
    for f in families:
        own = {}
        for script, alphabet in f['alphabets'].items():
            # Scripts too rare for a shared set (Lao, Myanmar, N'Ko...) use the family's own characters.
            allowed = common.get(script) or sets.get(script) or alphabet[:64]; covered = set(alphabet)
            chars = ''.join(c for c in allowed if c in covered)
            if len(chars) >= 8: own[script] = chars
        result[f['id']] = own
    return result


STATE = {}


def init(faces, pools, drawn=0.0):
    STATE.update(faces=faces, pools=pools, fonts=Fonts(), prepare=Preparer(), drawn=drawn)


def sample(spec):
    """(face index, script, seed, clean, case) -> prepared windows and labels. Unrenderable text retries with new seeds."""
    index, script, seed, clean, case = spec; face = STATE['faces'][index]; pool = STATE['pools'][face['family']][script]
    for attempt in range(6):
        rng = random.Random(seed*7 + attempt)
        try:
            value = text(rng, script, pool, case); size = rng.choice(SIZES)
            image = STATE['fonts'].render(face, size, lambda font: render(font, value, rng, script))
            drawn = not clean and rng.random() < STATE.get('drawn', 0)
            if drawn: image = sketch(image, rng)  # hand-drawn view: style, not identity
            gray = np.asarray(image, np.uint8) if clean else damage(image, rng)
            windows = STATE['prepare'](gray)
            if windows: return {'face':index,'script':script,'text':value,'windows':windows,'drawn':drawn}
        except (OSError, ValueError): continue
    return None


class Stream:
    """Prefetched batches of prepared views from a worker pool."""
    def __init__(self, faces, pools, plan, workers=10, prefetch=6, drawn=0.0):
        self.pool = multiprocessing.get_context('spawn').Pool(workers, initializer=init, initargs=(faces, pools, drawn))
        self.plan = plan; self.queue = queue.Queue(prefetch); self.stop = False; self.error = None
        self.thread = threading.Thread(target=self.fill, daemon=True); self.thread.start()

    def fill(self):
        pending = []
        try:
            while not self.stop:
                pending.append(self.pool.map_async(sample, self.plan(), chunksize=4))
                if len(pending) >= 3:
                    result = pending.pop(0).get()
                    self.queue.put([r for r in result if r is not None])
        except BaseException as error:
            self.error = error; self.queue.put(None)

    def __next__(self):
        views = self.queue.get()
        if views is None: raise RuntimeError('Training view stream failed') from self.error
        return views

    def close(self):
        self.stop = True; self.pool.terminate()


def tensors(views, device, bucket=64):
    """Windows of all views in a fixed 48x128 frame, count rounded up to a bucket: stable shapes avoid Metal kernel
    recompilation. Filler rows repeat real windows (keeping batch-norm statistics representative) with owner -1."""
    import torch
    windows = [(i, w) for i, v in enumerate(views) for w in v['windows']]; total = -(-len(windows)//bucket)*bucket
    sizes = np.ones((total, 2), np.int64); array = np.ones((total, 1, 48, 128), np.float32); owners = np.full(total, -1, np.int64)
    for n in range(total):
        i, w = windows[n % len(windows)]
        array[n, 0, :w.shape[0], :w.shape[1]] = w/np.float32(255); sizes[n] = w.shape
        if n < len(windows): owners[n] = i
    return torch.from_numpy(array).to(device), torch.from_numpy(sizes).to(device), torch.from_numpy(owners).to(device)
