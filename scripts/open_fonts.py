"""Openly licensed font files beyond Google Fonts, inventoried exactly like bench/corpus.json.

Each source is a pinned release: the rights holder's own where it still publishes
one, otherwise a faithful copy named at the source. Faces are read with the corpus's own
face_info, so scripts, alphabets, exclusions and the normal-face choice mean the
same thing here as for Google Fonts, and the catalogue compiler reads both alike.
Files stay local under .data/fonts-open; bench/open-fonts.json pins the archives.

    python scripts/open_fonts.py            download (verifying pins) and inventory
"""
import hashlib, io, json, re, sys, tarfile, zipfile
from pathlib import Path, PurePosixPath
from fontTools.agl import toUnicode
from fontTools.ttLib import TTFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.corpus as corpus  # face_info, normal_axes, blob

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / '.data/fonts-open'
MANIFEST = ROOT / 'bench/open-fonts.json'

# Pinned releases, from the rights holder where possible. `include` picks the font files to inventory.
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
    {'source': 'tex-gyre', 'licence': 'GUST Font License',
     'url': 'https://www.gust.org.pl/projects/e-foundry/tex-gyre/whole/tex_gyre-otf-2_609-31_03_2026.zip', 'include': r'\.otf$'},
    *[{'source': 'tex-gyre', 'licence': 'GUST Font License', 'include': r'\.otf$',
       'url': f'https://www.gust.org.pl/projects/e-foundry/tg-math/download/texgyre{name}.zip'}
      for name in ['bonum-math-1005', 'pagella-math-1632', 'schola-math-1533', 'termes-math-1543']],
]
# Microsoft's "Core fonts for the web" (1996-2002). The EULA inside each installer
# allows unlimited installation and use, and non-profit redistribution of true
# copies, and forbids alteration. Only the original installers are used.
COREFONTS = ['andale32', 'arial32', 'arialb32', 'comic32', 'courie32', 'georgi32', 'impact32', 'times32', 'trebuc32', 'verdan32', 'webdin32']
SOURCES += [{'source': 'ms-corefonts', 'licence': 'Microsoft Core Fonts for the Web EULA (use unlimited; no alteration; non-profit redistribution of true copies only)',
             'url': f'https://downloads.sourceforge.net/corefonts/{name}.exe', 'include': r'\.ttf$', 'format': 'cab'} for name in COREFONTS]
GITHUB = 'https://github.com/{}/archive/{}.zip'  # a commit archive: the repository exactly as pinned
CTAN = 'https://mirrors.ctan.org/fonts/{}.zip'   # CTAN keeps one live version; the pin catches any change
# The last Google Fonts revision that still carried Droid, before its removal in 4930caf (2017).
DROID = 'https://raw.githubusercontent.com/google/fonts/5fb32282c5969930c4268483ffa5664680bf73c8/apache/'
SOURCES += [
    {'source': 'bitstream-vera', 'licence': 'Bitstream Vera License',
     'url': 'https://download.gnome.org/sources/ttf-bitstream-vera/1.10/ttf-bitstream-vera-1.10.tar.bz2', 'include': r'\.ttf$'},
    {'source': 'droid', 'licence': 'Apache-2.0', 'format': 'file', 'include': r'\.ttf$',
     'files': [DROID + path for path in ['droidsans/DroidSans-Regular.ttf', 'droidsans/DroidSans-Bold.ttf', 'droidsans/LICENSE.txt',
                                         'droidserif/DroidSerif-Regular.ttf', 'droidserif/DroidSerif-Bold.ttf', 'droidserif/DroidSerif-Italic.ttf',
                                         'droidserif/DroidSerif-BoldItalic.ttf', 'droidsansmono/DroidSansMono-Regular.ttf']]},
    {'source': 'linux-libertine', 'licence': 'OFL-1.1 or GPL-2.0-or-later with font exception',
     'url': 'https://downloads.sourceforge.net/project/linuxlibertine/linuxlibertine/5.3.0/LinLibertineOTF_5.3.0_2012_07_02.tgz', 'include': r'\.otf$'},
    {'source': 'junicode', 'licence': 'OFL-1.1',
     'url': 'https://github.com/psb1558/Junicode-font/releases/download/v2.226/Junicode_2.226.zip', 'include': r'/VAR/[^/]+\.ttf$'},
    {'source': 'sil', 'licence': 'OFL-1.1',
     'url': 'https://github.com/silnrsi/font-doulos/releases/download/v7.000/DoulosSIL-7.000.zip', 'include': r'\.ttf$'},
    {'source': 'gnu-unifont', 'licence': 'OFL-1.1 or GPL-2.0-or-later with font exception', 'format': 'file', 'include': r'\.otf$',
     'files': ['https://ftp.gnu.org/gnu/unifont/unifont-18.0.01/unifont-18.0.01.otf']},
    {'source': 'hershey', 'licence': 'WTFPL-2.0 (AVHershey conversion of the public-domain Hershey vector fonts)',
     'url': GITHUB.format('scruss/AVHershey-OTF', '4344992ea54a2082a613f4ba05451991da84f46f'), 'include': r'/otf/[^/]+\.otf$'},
    {'source': 'opendyslexic', 'licence': 'OFL-1.1',
     'url': 'https://github.com/antijingoist/opendyslexic/releases/download/v0.91.12/opendyslexic-0.910.12-rc2-2019.10.17.zip', 'include': r'\.otf$'},
    {'source': 'ctan', 'licence': 'OFL-1.1', 'url': CTAN.format('euler-math'), 'include': r'\.otf$'},
    {'source': 'ctan', 'licence': 'OFL-1.1 or LPPL-1.3', 'url': CTAN.format('erewhon'), 'include': r'/opentype/[^/]+\.otf$'},
    {'source': 'ctan', 'licence': 'OFL-1.1', 'url': CTAN.format('Asana-Math'), 'include': r'\.otf$'},
    {'source': 'ctan', 'licence': 'OFL-1.1 or LPPL', 'url': CTAN.format('theanodidot'), 'include': r'\.ttf$'},
    {'source': 'ocr-skala', 'licence': 'Free use including commercial; the converter\'s own work public domain, contributors\' claims licensed (tsukurimashou.org/ocr.php.en)',
     'url': 'https://tsukurimashou.org/files/ocr-0.3.1.zip', 'include': r'\.otf$'},
    {'source': 'd-din', 'licence': 'OFL-1.1', 'url': 'https://fontlibrary.org/assets/downloads/d-din/39fc8c3de156292478f87c2ff0b96df2/d-din.zip', 'include': r'\.otf$'},
    {'source': 'railway-sans', 'licence': 'OFL-1.1', 'url': GITHUB.format('googlefonts/railway-sans', 'f57159bb2ec23e96830ef1b0c81ad112f6cf3efd'), 'include': r'/fonts/OTF/[^/]+\.otf$'},
    {'source': 'copperplate-cc', 'licence': 'OFL-1.1', 'url': 'https://github.com/CowboyCollective/CopperplateCC/releases/download/1.0/CopperplateCC.zip', 'include': r'(^|/)otf/[^/]+\.otf$'},
    {'source': 'cc-accidenz-commons', 'licence': 'CC-BY-SA-4.0', 'url': GITHUB.format('creativecommons/cc-accidenz-commons', 'bd2f0e4ca7751907e0c420a933b31dd72476dc54'), 'include': r'\.otf$'},
    {'source': 'chicago-kare', 'licence': 'MIT (repository LICENSE; the font\'s name table says OFL)', 'url': GITHUB.format('KingDuane/Chicago-Kare', 'dca7a9e4f2b971e39cf3a5e8a3f9309d4293d56d'), 'include': r'\.otf$'},
    {'source': 'roadgeek', 'licence': 'MIT', 'url': 'https://github.com/sammdot/roadgeek-fonts/releases/download/3.1/RG2014-3.10.zip', 'include': r'\.ttf$'},
    {'source': 'fixedsys-excelsior', 'licence': 'CC0-1.0', 'format': 'file', 'include': r'\.ttf$',
     'files': ['https://github.com/kika/fixedsys/releases/download/v3.09.10/FSEX302.ttf']},
    {'source': 'ubuntu-titling', 'licence': 'OFL-1.1', 'url': 'https://deb.debian.org/debian/pool/main/f/fonts-ubuntu-title/fonts-ubuntu-title_0.3.orig.tar.gz', 'include': r'\.ttf$'},
]
# Families from open catalogues (Velvetyne, Collletttivo, Uncut): pinned to their designers'
# repositories by scripts/open_releases.py, or a release file on the designer's own site.
SOURCES += [{'source': row['source'], 'licence': row['licence'], 'url': row['url'], 'include': row['include'],
             **({'format': row['format']} if row.get('format') else {}), **({'commit': row['commit']} if row.get('format') == 'git' else {})}
            for row in json.loads((ROOT / 'bench/open-releases.json').read_text())['releases'] if row.get('include')]
# The League of Moveable Type: all 18 families are OFL-1.1. Eleven reach us through Google
# Fonts; these are the other seven, as the GitHub organisation holds them.
LEAGUE_REPOS = {'junction': 'fb73260e86dd301b383cf6cc9ca8e726ef806535', 'chunk': '12a243f3fb7c7a68844901023f7d95d6eaf14104',
                'blackout': '4864cfc1749590e9f78549c6e57116fe98480c0f', 'fanwood': 'cbaaed9704e7d37d3dcdbdf0b472e9efd0e39432',
                'ostrich-sans': 'a949d40d0576d12ba26e2a45e19c91fd0228c964'}
SOURCES += [{'source': 'league-of-moveable-type', 'licence': 'OFL-1.1', 'url': GITHUB.format(f'theleagueof/{repo}', sha),
             # Blackout's "Two AM.ttf" is version 2.002 of "2 AM.ttf" (2.003): the stale copy is skipped.
             'include': r'^(?!.*/webfonts/)(?!.*Two AM).*\.ttf$' if repo == 'blackout' else r'^(?!.*/webfonts/).*\.otf$'} for repo, sha in LEAGUE_REPOS.items()]
SOURCES += [{'source': 'league-of-moveable-type', 'licence': 'OFL-1.1', 'include': r'/static/OTF/[^/]+\.otf$',
             'url': f'https://github.com/theleagueof/{repo}/releases/download/{version}/{name}-{version}.zip'}
            for repo, name, version in [('league-mono', 'LeagueMono', '2.300'), ('the-neue-black', 'TheNeueBlack', '1.007')]]
# Fontshare publishes its whole catalogue through a public API. Its font files carry
# "false" as their family name, so names come from the catalogue instead.
FONTSHARE_API = 'https://api.fontshare.com/v2/fonts?limit=200'
FONTSHARE_LICENCES = {'itf_ffl': 'ITF Free Font License (free for personal and commercial use; no redistribution or modification)', 'sil_ofl': 'OFL-1.1'}
# Fontsource's own fonts (type "other"), each pinned to its npm release. Its robots.txt
# signals ai-train=yes. Blackout Two AM is the League's stale copy of Blackout 2AM, held above.
FONTSOURCE_API = 'https://api.fontsource.org/v1/fonts'
FONTSOURCE_SKIP = {'blackout-two-am'}
# A font is known by its signature, not its name: repositories also commit macOS "._" forks,
# Git LFS pointers and HTML error pages under font extensions.
SFNT = (b'OTTO', b'\x00\x01\x00\x00', b'true', b'ttcf')
is_font = lambda content: content[:4] in SFNT
LICENCE_NAMES = re.compile(r'(^|/)(LICEN[CS]E|COPYING|COPYRIGHT|OFL|GUST-FONT-LICENSE|README)[^/]*$', re.I)


def slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')


def name_key(value):
    """Family names compared across sources: case, spaces and punctuation ignored."""
    return re.sub(r'[^a-z0-9]', '', value.lower())


def fetch(url, pinned, commit=None):
    """Download once; a pinned archive must match its recorded hash byte for byte.
    With a commit, `url` is a git repository and the archive is that commit, made by git."""
    cache = STORE / '.archives' / hashlib.sha1(f'{url}#{commit or ""}'.encode()).hexdigest()
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        if commit:
            import subprocess, tempfile
            with tempfile.TemporaryDirectory() as folder:
                subprocess.run(['git', 'clone', '--quiet', '--filter=blob:none', '--no-checkout', url, folder], check=True, capture_output=True)
                prefix = f"{PurePosixPath(url).stem}-{commit}/"
                cache.write_bytes(subprocess.run(['git', '-C', folder, 'archive', '--format=zip', f'--prefix={prefix}', commit], check=True, capture_output=True).stdout)
        else:
            cache.write_bytes(corpus.request(url))
    data = cache.read_bytes(); digest = hashlib.sha256(data).hexdigest()
    if pinned and pinned != digest:
        raise ValueError(f'Archive changed since it was pinned: {url}')
    return data, digest


def members(data, url, format=None):
    """(path, bytes) for every file in a zip, tar or cabinet archive, or the file itself."""
    if format == 'file':
        yield PurePosixPath(url.split('?')[0]).name, data
        return
    if format == 'cab':  # self-extracting cabinet installers; libarchive's bsdtar reads them
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'installer.exe'; archive.write_bytes(data)
            subprocess.run(['bsdtar', '-xf', str(archive), '-C', folder], check=True, capture_output=True)
            for path in sorted(Path(folder).iterdir()):
                if path.is_file() and path.name != 'installer.exe': yield path.name, path.read_bytes()
        return
    if url.endswith('.zip') or format == 'git':
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if not info.is_dir(): yield info.filename, archive.read(info)
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for info in archive.getmembers():
                if info.isfile(): yield info.name, archive.extractfile(info).read()


def symbol_encoded(path):
    """Symbol fonts (Dingbats, Symbol) put pictographs or Greek on the Latin code points.

    Read from glyph names: a text face names the glyph for "A" A or uni0041; a symbol
    font names it after another character (Alpha) or a dingbat (a10). Names that
    denote nothing standard (mupA in a maths font, cid00066) decide nothing.
    """
    with TTFont(path, lazy=True) as font:
        cmap = font.getBestCmap() or {}
        own = foreign = 0
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
            if not (name := cmap.get(ord(letter))): continue
            base = name.split('.')[0]
            meaning = '' if re.fullmatch(r'a\d+', base) else toUnicode(base)
            if meaning == letter: own += 1
            elif meaning or re.fullmatch(r'a\d+', base): foreign += 1
        return foreign > own


def colour_letters(path):
    """Whether colour data draws the letters themselves. OpenDyslexic colours only a few
    stylistic alternates; its text renders in plain outlines."""
    with TTFont(path, lazy=True) as font:
        if 'CBDT' in font or 'sbix' in font: return True
        letters = {name for cp, name in (font.getBestCmap() or {}).items() if chr(cp).isalpha()}
        coloured = set()
        if 'COLR' in font:
            colr = font['COLR']
            if colr.version == 0: coloured |= set(colr.ColorLayers)
            else:
                if colr.table.BaseGlyphRecordArray: coloured |= {r.BaseGlyph for r in colr.table.BaseGlyphRecordArray.BaseGlyphRecord}
                if colr.table.BaseGlyphList: coloured |= {r.BaseGlyph for r in colr.table.BaseGlyphList.BaseGlyphPaintRecord}
        if 'SVG ' in font:
            order = font.getGlyphOrder()
            for doc in font['SVG '].docList:
                start, end = (doc.startGlyphID, doc.endGlyphID) if hasattr(doc, 'startGlyphID') else doc[1:3]
                coloured |= set(order[start:end + 1])
        return bool(letters & coloured)


def stored_font(relative, url):
    """One font file from an API, downloaded once into the store. None when the server's file is not
    a font: Fontsource serves some Open Sauce Sans styles as Type 1 under a .ttf name."""
    import time
    target = STORE / relative
    if target.exists(): return target.read_bytes()
    if not is_font(content := corpus.request(url)): print(f'{relative}: not an OpenType or TrueType font; style skipped'); return None
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content); time.sleep(0.2)
    return content


def fontshare_groups(google_names):
    """Static styles of every Fontshare family that Google Fonts does not already carry."""
    fonts = get_json(FONTSHARE_API)['fonts']
    for font in sorted(fonts, key=lambda f: f['slug']):
        if name_key(font['name']) in google_names: continue
        items = []
        for style in font['styles']:
            if style['is_variable']: continue
            relative = f"fontshare/{font['slug']}/{slug(style['weight']['name'])}.ttf"
            if not (content := stored_font(relative, 'https:' + style['file'] + '.ttf')): continue
            items.append({'path': relative, 'size': len(content), 'sha': corpus.blob(content)})
        if items:
            yield font['name'], {'spec': {'licence': FONTSHARE_LICENCES.get(font['license_type'], font['license_type'])}, 'items': items, 'licences': [],
                                 'extra': {'sourceUrl': f"https://www.fontshare.com/fonts/{font['slug']}", 'sourceId': font['id']}}


def get_json(url):
    return json.loads(corpus.request(url))  # retried with backoff


def fontsource_groups(held):
    """Fontsource's own fonts that no source above already carries, under either the catalogue's
    name or the one inside the file (its "DejaVu Mono" is DejaVu Sans Mono). Fontsource serves
    these in one subset, usually Latin; `subset` records which."""
    for entry in sorted(get_json(FONTSOURCE_API), key=lambda f: f['id']):
        if entry['type'] == 'google' or entry['id'] in FONTSOURCE_SKIP or name_key(entry['family']) in held: continue
        try: font = get_json(f"{FONTSOURCE_API}/{entry['id']}"); items = fontsource_files(font)
        except OSError as error: print(f"fontsource: {entry['family']} skipped: {error}"); continue
        if not items or name_key(family_name(STORE / items[0]['path'])) in held: continue
        held.add(name_key(font['family']))
        yield font['family'], {'spec': {'licence': font['license']}, 'items': items, 'licences': [],
                               'extra': {'sourceUrl': f"https://fontsource.org/fonts/{font['id']}", 'sourceId': font['id'],
                                         'upstream': font.get('source'), 'version': font['npmVersion'], 'subset': font['defSubset']}}


def fontsource_files(font):
    """Every style of one Fontsource font in its default subset, pinned to its npm release."""
    items = []
    for weight, styles in sorted(font['variants'].items()):
        for style, subsets in sorted(styles.items()):
            if not (files := subsets.get(font['defSubset'])): continue
            url = files['url']['ttf'].replace('@latest/', f"@{font['npmVersion']}/")
            relative = f"fontsource/{font['id']}/{PurePosixPath(url).name}"
            if not (content := stored_font(relative, url)): continue
            items.append({'path': relative, 'size': len(content), 'sha': corpus.blob(content)})
    return items


def family_name(path):
    with TTFont(path, lazy=True) as font:
        name = font['name']
        return (name.getDebugName(16) or name.getDebugName(1) or path.stem).strip()


def main():
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'archives': []}
    pins = {entry['url']: entry['sha256'] for entry in previous.get('archives', []) if 'sha256' in entry}
    archives, groups = [], {}
    owners, elsewhere = {}, {}  # a family belongs to the first source that lists it: foundries before catalogues
    original_source_path = corpus.source_path
    corpus.source_path = lambda path: STORE / path  # face_info reads our store, not the Google cache
    try:
        for spec in SOURCES:
            licences = []
            for url in spec.get('files') or [spec['url']]:  # an archive, or loose files pinned one by one
                key = f"{url}#{spec['commit']}" if 'commit' in spec else url
                data, digest = fetch(url, pins.get(key), spec.get('commit'))
                archives.append({'source': spec['source'], 'url': key, 'sha256': digest, 'bytes': len(data), 'licence': spec['licence']})
                root = PurePosixPath(spec['source']) / ('' if 'files' in spec else Path(url).name.split('?')[0])
                for name, content in members(data, url, spec.get('format')):
                    keep_font = re.search(spec['include'], name, re.I) and is_font(content)
                    keep_licence = LICENCE_NAMES.search(name)
                    if not (keep_font or keep_licence): continue
                    relative = str(root / PurePosixPath(name))
                    target = STORE / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists() or target.read_bytes() != content: target.write_bytes(content)
                    if keep_licence and not keep_font:
                        licences.append({'path': relative, 'blob': corpus.blob(content)}); continue
                    item = {'path': relative, 'size': len(content), 'sha': corpus.blob(content)}
                    name = family_name(target)
                    if (owner := owners.setdefault(name_key(name), spec['source'])) != spec['source']:
                        elsewhere.setdefault(spec['source'], set()).add(f'{name} ({owner})'); continue
                    groups.setdefault((spec['source'], name), {'spec': spec, 'items': [], 'licences': licences})['items'].append(item)
        google = json.loads((ROOT / 'bench/corpus.json').read_text())['families']
        google_names = {name_key(f['family']) for f in google}
        for name, group in fontshare_groups(google_names): groups[('fontshare', name)] = group
        archives.append({'source': 'fontshare', 'url': FONTSHARE_API, 'licence': 'per family (ITF Free Font License or OFL-1.1)'})
        # Aggregators last: a family the rights holder's own release or Google Fonts carries is taken from there.
        held = google_names | {name_key(name) for _, name in groups}
        for name, group in fontsource_groups(held): groups[('fontsource', name)] = group
        archives.append({'source': 'fontsource', 'url': FONTSOURCE_API, 'licence': 'per family (OFL-1.1, Apache-2.0, MIT, CC0-1.0, Unlicense)'})
        families, in_google = [], {}
        for (source, name), group in sorted(groups.items()):
            if name_key(name) in google_names:
                in_google.setdefault(source, []).append(name); continue
            faces = []
            for item in group['items']:
                # Third-party files are sometimes malformed; fontTools raises all kinds on them.
                try: faces.append(corpus.face_info(item))
                except Exception as error: print(f"{item['path']}: unreadable, left out ({type(error).__name__}: {error})")
            if not faces: continue
            selected, letters = min(faces, key=lambda pair: (pair[0]['italic'], abs(corpus.normal_axes(pair[0]).get('wght', pair[0]['weight']) - 400), pair[0]['path']))
            reason = 'color' if selected['color'] and colour_letters(STORE / selected['path']) else 'no-letter-glyphs' if not letters else None
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
    for source, names in in_google.items(): print(f'{source}: left to Google Fonts: {", ".join(names)}')
    for source, names in elsewhere.items(): print(f'{source}: left to an earlier source: {", ".join(sorted(names))}')


if __name__ == '__main__':
    main()
