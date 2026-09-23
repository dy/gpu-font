"""Openly licensed font files beyond Google Fonts, inventoried exactly like bench/corpus.json.

Each source is a pinned official archive. Faces are read with the corpus's own
face_info, so scripts, alphabets, exclusions and the normal-face choice mean the
same thing here as for Google Fonts, and the catalogue compiler reads both alike.
Files stay local under .data/fonts-open; bench/open-fonts.json pins the archives.

    python scripts/open_fonts.py            download (verifying pins) and inventory
"""
import hashlib, io, json, re, sys, tarfile, urllib.request, zipfile
from pathlib import Path, PurePosixPath
from fontTools.ttLib import TTFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.corpus as corpus  # face_info, normal_axes, blob

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / '.data/fonts-open'
MANIFEST = ROOT / 'bench/open-fonts.json'

# Official release archives only. `include` picks the font files to inventory.
SOURCES = [
    {'source': 'dejavu', 'licence': 'Bitstream Vera License; DejaVu changes public domain',
     'url': 'https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip', 'include': r'/ttf/[^/]+\.ttf$'},
    {'source': 'liberation', 'licence': 'OFL-1.1',
     'url': 'https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz', 'include': r'\.ttf$'},
    {'source': 'gnu-freefont', 'licence': 'GPL-3.0-or-later with font exception',
     'url': 'https://ftp.gnu.org/gnu/freefont/freefont-otf-20120503.tar.gz', 'include': r'\.otf$'},
    {'source': 'urw-base35', 'licence': 'AGPL-3.0 with font exception',
     'url': 'https://github.com/ArtifexSoftware/urw-base35-fonts/archive/refs/tags/20200910.tar.gz', 'include': r'/fonts/[^/]+\.otf$'},
    {'source': 'latin-modern', 'licence': 'GUST Font License',
     'url': 'https://www.gust.org.pl/projects/e-foundry/latin-modern/download/lm2.004otf.zip', 'include': r'\.otf$'},
    {'source': 'latin-modern', 'licence': 'GUST Font License',
     'url': 'https://www.gust.org.pl/projects/e-foundry/lm-math/download/latinmodern-math-1959.zip', 'include': r'\.otf$'},
]
# Microsoft's "Core fonts for the web" (1996-2002). The EULA inside each installer
# allows unlimited installation and use, and non-profit redistribution of true
# copies, and forbids alteration. Only the original installers are used.
COREFONTS = ['andale32', 'arial32', 'arialb32', 'comic32', 'courie32', 'georgi32', 'impact32', 'times32', 'trebuc32', 'verdan32', 'webdin32']
SOURCES += [{'source': 'ms-corefonts', 'licence': 'Microsoft Core Fonts for the Web EULA (use unlimited; no alteration; non-profit redistribution of true copies only)',
             'url': f'https://downloads.sourceforge.net/corefonts/{name}.exe', 'include': r'\.ttf$', 'format': 'cab'} for name in COREFONTS]
# Fontshare publishes its whole catalogue through a public API. Its font files carry
# "false" as their family name, so names come from the catalogue instead.
FONTSHARE_API = 'https://api.fontshare.com/v2/fonts?limit=200'
FONTSHARE_LICENCES = {'itf_ffl': 'ITF Free Font License (free for personal and commercial use; no redistribution or modification)', 'sil_ofl': 'OFL-1.1'}
LICENCE_NAMES = re.compile(r'(^|/)(LICEN[CS]E|COPYING|OFL|GUST-FONT-LICENSE|README)[^/]*$', re.I)


def slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')


def fetch(url, pinned):
    """Download once; a pinned archive must match its recorded hash byte for byte."""
    cache = STORE / '.archives' / hashlib.sha1(url.encode()).hexdigest()
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'gpu-font open-font inventory'})
        cache.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    data = cache.read_bytes(); digest = hashlib.sha256(data).hexdigest()
    if pinned and pinned != digest:
        raise ValueError(f'Archive changed since it was pinned: {url}')
    return data, digest


def members(data, url, format=None):
    """(path, bytes) for every file in a zip, tar or cabinet archive."""
    if format == 'cab':  # self-extracting cabinet installers; libarchive's bsdtar reads them
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'installer.exe'; archive.write_bytes(data)
            subprocess.run(['bsdtar', '-xf', str(archive), '-C', folder], check=True, capture_output=True)
            for path in sorted(Path(folder).iterdir()):
                if path.is_file() and path.name != 'installer.exe': yield path.name, path.read_bytes()
        return
    if url.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if not info.is_dir(): yield info.filename, archive.read(info)
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for info in archive.getmembers():
                if info.isfile(): yield info.name, archive.extractfile(info).read()


def symbol_encoded(path):
    """Symbol fonts (Dingbats, Symbol) put pictographs or Greek on the Latin code points.

    Read from glyph names: a text face names the glyph for "A" A (or uni0041, A.ss01);
    a symbol font names it a10 or Alpha. Unnamed glyphs (glyph123) leave it undetermined.
    """
    with TTFont(path, lazy=True) as font:
        cmap = font.getBestCmap() or {}
        names = [(letter, cmap.get(ord(letter))) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz']
        names = [(letter, name) for letter, name in names if name]
        if not names or all(re.fullmatch(r'(glyph|gid|g)\d+', name) for _, name in names): return False
        own = sum(name.split('.')[0] in (letter, f'uni{ord(letter):04X}') for letter, name in names)
        return own < len(names) / 2


def fontshare_groups(google_names):
    """Static styles of every Fontshare family that Google Fonts does not already carry."""
    import time
    request = urllib.request.Request(FONTSHARE_API, headers={'User-Agent': 'gpu-font open-font inventory'})
    fonts = json.loads(urllib.request.urlopen(request, timeout=60).read())['fonts']
    key = lambda value: re.sub(r'[^a-z0-9]', '', value.lower())
    for font in sorted(fonts, key=lambda f: f['slug']):
        if key(font['name']) in google_names: continue
        items = []
        for style in font['styles']:
            if style['is_variable']: continue
            relative = f"fontshare/{font['slug']}/{slug(style['weight']['name'])}.ttf"
            target = STORE / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                download = urllib.request.Request('https:' + style['file'] + '.ttf', headers={'User-Agent': 'gpu-font open-font inventory'})
                target.write_bytes(urllib.request.urlopen(download, timeout=60).read())
                time.sleep(0.3)
            content = target.read_bytes()
            items.append({'path': relative, 'size': len(content), 'sha': corpus.blob(content)})
        if items:
            yield font['name'], {'spec': {'licence': FONTSHARE_LICENCES.get(font['license_type'], font['license_type'])}, 'items': items, 'licences': [],
                                 'extra': {'sourceUrl': f"https://www.fontshare.com/fonts/{font['slug']}", 'sourceId': font['id']}}


def family_name(path):
    with TTFont(path, lazy=True) as font:
        name = font['name']
        return (name.getDebugName(16) or name.getDebugName(1) or path.stem).strip()


def main():
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'archives': []}
    pins = {entry['url']: entry['sha256'] for entry in previous.get('archives', []) if 'sha256' in entry}
    archives, groups = [], {}
    original_source_path = corpus.source_path
    corpus.source_path = lambda path: STORE / path  # face_info reads our store, not the Google cache
    try:
        for spec in SOURCES:
            data, digest = fetch(spec['url'], pins.get(spec['url']))
            archives.append({'source': spec['source'], 'url': spec['url'], 'sha256': digest, 'bytes': len(data), 'licence': spec['licence']})
            root = PurePosixPath(spec['source']) / Path(spec['url']).name.split('?')[0]
            licences = []
            for name, content in members(data, spec['url'], spec.get('format')):
                keep_font = re.search(spec['include'], name, re.I)
                keep_licence = LICENCE_NAMES.search(name)
                if not (keep_font or keep_licence): continue
                relative = str(root / PurePosixPath(name))
                target = STORE / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.read_bytes() != content: target.write_bytes(content)
                if keep_licence and not keep_font:
                    licences.append({'path': relative, 'blob': corpus.blob(content)}); continue
                item = {'path': relative, 'size': len(content), 'sha': corpus.blob(content)}
                groups.setdefault((spec['source'], family_name(target)), {'spec': spec, 'items': [], 'licences': licences})['items'].append(item)
        google = json.loads((ROOT / 'bench/corpus.json').read_text())['families']
        google_names = {re.sub(r'[^a-z0-9]', '', f['family'].lower()) for f in google}
        for name, group in fontshare_groups(google_names): groups[('fontshare', name)] = group
        archives.append({'source': 'fontshare', 'url': FONTSHARE_API, 'licence': 'per family (ITF Free Font License or OFL-1.1)'})
        families = []
        for (source, name), group in sorted(groups.items()):
            faces = [corpus.face_info(item) for item in group['items']]
            selected, letters = min(faces, key=lambda pair: (pair[0]['italic'], abs(corpus.normal_axes(pair[0]).get('wght', pair[0]['weight']) - 400), pair[0]['path']))
            reason = 'color' if selected['color'] else 'no-letter-glyphs' if not letters else None
            alphabets = {s: ''.join(map(chr, cps)) for s, cps in letters.items() if len(cps) >= 8 and s not in ('Zyyy', 'Zinh')}
            if not reason and not alphabets: reason = 'insufficient-letter-coverage'
            symbol = symbol_encoded(STORE / selected['path'])
            families.append({'id': f'{source}-{slug(name)}', 'family': name, 'source': source, 'folder': str(PurePosixPath(selected['path']).parent),
                             **({'symbolEncoded': True} if symbol else {}), **group.get('extra', {}),
                             'declaredLicense': [group['spec']['licence']], 'licenses': group['licences'],
                             'faces': [face for face, _ in faces], 'selected': selected['path'],
                             'trainingAxes': corpus.normal_axes(selected), 'alphabets': alphabets, 'excluded': reason})
    finally:
        corpus.source_path = original_source_path
    ids = [f['id'] for f in families]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate family ids')
    result = {'version': 1, 'store': '.data/fonts-open',
              'scope': 'Openly licensed families outside Google Fonts, from pinned official archives. Same family/face/alphabet semantics as bench/corpus.json; one normal face selected per family; exclusions explicit.',
              'archives': archives, 'fontFiles': sum(len(f['faces']) for f in families), 'families': families}
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    counts = {}
    for f in families: counts.setdefault(f['source'], [0, 0])[0 if not f['excluded'] else 1] += 1
    for source, (ok, excluded) in counts.items(): print(f'{source}: {ok} families{f", {excluded} excluded" if excluded else ""}')


if __name__ == '__main__':
    main()
