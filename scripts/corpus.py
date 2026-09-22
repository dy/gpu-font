"""Pinned Google Fonts sources and explicit family/face/glyph coverage."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import time
import tempfile
import unicodedata
import urllib.request
from urllib.parse import quote

from fontTools.ttLib import TTFont
from fontTools.unicodedata import script
from fontTools.pens.boundsPen import BoundsPen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'e44c4b011a820c2cbe2fd2cfa8052037d7edb571'
CACHE = ROOT / '.data/google' / COMMIT
MANIFEST = ROOT / 'bench/corpus.json'


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def source_path(path):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or len(p.parts) < 3 or p.parts[0] not in ('ofl', 'apache', 'ufl') or str(p) != path:
        raise ValueError('Invalid source path')
    return CACHE / path


def request(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'gpu-font'}), timeout=120) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 4: raise
            time.sleep(2 ** attempt)


def tree():
    path = ROOT / '.data/google/tree-full.json'
    if path.exists():
        result = json.loads(path.read_text())
    else:
        result = json.loads(request(f'https://api.github.com/repos/google/fonts/git/trees/{COMMIT}?recursive=1'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    if result.get('sha') != COMMIT or result.get('truncated') is not False or not isinstance(result.get('tree'), list):
        raise ValueError('Incomplete or wrong source tree')
    return [f for f in result['tree'] if f['type'] == 'blob' and f['path'].startswith(('ofl/', 'apache/', 'ufl/'))]


def fetch(item):
    path = source_path(item['path'])
    data = path.read_bytes() if path.exists() else request(f'https://raw.githubusercontent.com/google/fonts/{COMMIT}/{quote(item["path"])}')
    if len(data) != item['size'] or blob(data) != item['sha']:
        raise ValueError('Source checksum mismatch: ' + item['path'])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.part', delete=False) as out:
            temporary = Path(out.name)
            out.write(data)
        try: temporary.replace(path)
        finally: temporary.unlink(missing_ok=True)
    return path


def download():
    files = [f for f in tree() if f['path'].endswith(('.ttf', '/METADATA.pb', '/OFL.txt', '/LICENSE.txt', '/LICENCE.txt', '/UFL.txt'))]
    total = sum(f['size'] for f in files)
    print(f'{len(files)} pinned files; {total:,} bytes (cached files verified and reused)', flush=True)
    with ThreadPoolExecutor(max_workers=20) as pool:
        jobs = [pool.submit(fetch, item) for item in files]
        for i, job in enumerate(as_completed(jobs), 1):
            job.result()
            if i % 100 == 0 or i == len(files): print(f'Sources {i}/{len(files)}', flush=True)


def metadata_strings(text, field):
    # Project only top-level protobuf string fields. Nested source/font names are not family names.
    return [json.loads(m) for m in re.findall(r'^' + re.escape(field) + r':\s*("(?:[^"\\]|\\.)*")\s*$', text, re.M)]


def face_info(item):
    # The download command verifies all blobs first. Fail fast on incomplete caches.
    path = source_path(item['path'])
    if not path.exists() or path.stat().st_size != item['size'] or blob(path.read_bytes()) != item['sha']:
        raise ValueError('Run corpus download first; missing or corrupt source: ' + item['path'])
    with TTFont(path, lazy=True, recalcTimestamp=False) as font:
        cmap = font.getBestCmap() or {}
        color = any(tag in font for tag in ('COLR', 'CBDT', 'sbix', 'SVG '))
        glyphs = font['glyf'] if 'glyf' in font else None
        glyph_set = None if glyphs is not None or color else font.getGlyphSet()
        visible = {}
        def ink(name):
            if color: return True
            if name in visible: return visible[name]
            if glyphs is not None:
                glyph = glyphs.glyphs[name]
                # Read the glyph header without expanding every outline in large CJK fonts.
                if hasattr(glyph, 'data'):
                    bounds = struct.unpack('>hhhhh', glyph.data[:10]) if len(glyph.data) >= 10 else (0,0,0,0,0)
                else:
                    bounds = (glyph.numberOfContours, *[getattr(glyph, k, 0) for k in ['xMin','yMin','xMax','yMax']])
                contours, left, top, right, bottom = bounds
                value = contours != 0 and right > left and bottom > top
            else:
                pen = BoundsPen(glyph_set); glyph_set[name].draw(pen)
                value = pen.bounds is not None and pen.bounds[2] > pen.bounds[0] and pen.bounds[3] > pen.bounds[1]
            visible[name] = value
            return value
        letters = sorted(cp for cp, name in cmap.items() if name != '.notdef' and unicodedata.category(chr(cp)).startswith('L') and ink(name))
        groups = {}
        for cp in letters: groups.setdefault(script(chr(cp)), []).append(cp)
        axes = {a.axisTag: {'min': a.minValue, 'default': a.defaultValue, 'max': a.maxValue} for a in font['fvar'].axes} if 'fvar' in font else {}
        return {'path': item['path'], 'blob': item['sha'], 'bytes': item['size'],
                'weight': font['OS/2'].usWeightClass if 'OS/2' in font else 400,
                'italic': bool(font['OS/2'].fsSelection & 1) if 'OS/2' in font else False,
                'axes': axes, 'scripts': {s: len(v) for s, v in sorted(groups.items())},
                'embeddedLicense': font['name'].getDebugName(13) if 'name' in font else None,
                'color': color}, groups


def catalog():
    files = tree(); folders = {}
    for item in files:
        if item['path'].endswith('.ttf'): folders.setdefault('/'.join(PurePosixPath(item['path']).parts[:2]), []).append(item)
    by_path = {item['path']: item for item in files}
    entries = []
    for i, (folder, sources) in enumerate(sorted(folders.items())):
        metadata = by_path.get(folder + '/METADATA.pb')
        if metadata:
            text = fetch(metadata).read_text()
            names = metadata_strings(text, 'name')
            if len(names) != 1: raise ValueError('Missing/duplicate family name: ' + folder)
            name = names[0]; subsets = metadata_strings(text, 'subsets'); license_names = metadata_strings(text, 'license')
        else:
            name = folder.split('/')[-1]; subsets = []; license_names = []
        faces = [face_info(f) for f in sources]
        # The genuine upright face closest to regular; variable fonts use their actual defaults.
        selected, groups = min(faces, key=lambda pair: (pair[0]['italic'], abs(pair[0]['axes'].get('wght', {}).get('default', pair[0]['weight']) - 400), pair[0]['path']))
        reason = 'unregistered' if not metadata else 'color' if selected['color'] else 'no-letter-glyphs' if not groups else None
        # Keep all letter scripts. Training text pools are shared per script, never per-family secret phrases.
        alphabets = {s: ''.join(map(chr, cps)) for s, cps in groups.items() if len(cps) >= 8 and s not in ('Zyyy', 'Zinh')}
        if not reason and not alphabets: reason = 'insufficient-letter-coverage'
        licenses = [f for p, f in by_path.items() if str(PurePosixPath(p).parent) == folder and PurePosixPath(p).name in ('OFL.txt', 'LICENSE.txt', 'LICENCE.txt', 'UFL.txt')]
        entries.append({'id': folder.split('/')[-1], 'family': name, 'folder': folder,
                        'metadataBlob': metadata['sha'] if metadata else None, 'subsets': subsets,
                        'declaredLicense': license_names,
                        'licenses': [{'path': f['path'], 'blob': f['sha']} for f in licenses],
                        'faces': [f for f, _ in faces], 'selected': selected['path'],
                        'alphabets': alphabets, 'excluded': reason})
        if (i + 1) % 100 == 0: print(f'Catalog {i+1}/{len(folders)}', flush=True)
    ids = [e['id'] for e in entries]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate family IDs')
    result = {'version': 1, 'commit': COMMIT, 'source': 'https://github.com/google/fonts',
              'scope': 'All TTF files under the three font license roots at the pinned revision. One default face per eligible family; exclusions explicit.',
              'fontFiles': sum(len(e['faces']) for e in entries), 'sourceBytes': sum(f['size'] for f in files if f['path'].endswith('.ttf')),
              'families': entries}
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(Counter(e['excluded'] or 'eligible' for e in entries), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['download', 'catalog']); args = p.parse_args()
    globals()[args.command]()
