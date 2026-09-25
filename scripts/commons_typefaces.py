"""Photos of typefaces in use from Wikimedia Commons, every one under a free licence.

Commons sorts files into a category per typeface (Category:Typefaces by name: Helvetica, Futura,
DIN 1451…), and those hold photos of signs, packaging and print beside specimens and single-glyph
renders. Each file takes the label of the nearest typeface category above it: a typeface category
inside another (Arial inside Helvetica) labels its own files, and comparisons ("Helvetica vs.
Bodoni") label none. The walk goes one level of subcategories down ("DIN 1451 in Greece"): deeper
ones drift into broad categories (every street sign of a town) and into files made with the font
(Noto charts, rendered formulas). A photo is a JPEG with a camera in its metadata. The API is read
serially, one request a second, as Wikimedia's API etiquette asks.

    python scripts/commons_typefaces.py list      every file and its typeface -> files.jsonl
    python scripts/commons_typefaces.py info      licence, size and camera of each -> info.jsonl
    python scripts/commons_typefaces.py images    the photos, at most 2560 px wide -> images/
"""
import hashlib, json, re, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.data/photos/commons'
API = 'https://commons.wikimedia.org/w/api.php'
AGENT = 'gpu-font font-recognition research (https://github.com/dy/gpu-font)'
WIDTH = 2560  # wide enough for any line of text in a photo; larger originals come as Commons' own thumbnail
SKIP = re.compile(r' vs\.? ', re.I)  # comparisons show two typefaces side by side
JPEG = re.compile(r'\.jpe?g$', re.I)  # a camera photo; specimens and renders are SVG, PNG or PDF


def get(url, data=None):
    time.sleep(1)
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers={'User-Agent': AGENT}), timeout=120) as response: return response.read()


def api(**query):
    """Every page of a query's results, following `continue`."""
    extra = {}
    while True:
        # POST: fifty long file titles overflow a URL.
        data = json.loads(get(API, urllib.parse.urlencode({**query, **extra, 'format': 'json', 'formatversion': 2, 'maxlag': 5}).encode()))
        if 'error' in data: raise RuntimeError(data['error'])
        yield data['query']
        if 'continue' not in data: return
        extra = data['continue']


def members(category, kind):
    return [m['title'] for page in api(action='query', list='categorymembers', cmtitle=category, cmtype=kind, cmlimit=500) for m in page['categorymembers']]


def typeface(category):
    """Category:Futura (typeface) -> Futura."""
    return re.sub(r'\s*\((typeface|font|typefaces|fonts)\)$', '', category.removeprefix('Category:'))


def listing(depth=1):
    OUT.mkdir(parents=True, exist_ok=True)
    typefaces = set(members('Category:Typefaces by name', 'subcat'))
    seen, rows = set(), []
    def walk(category, label, path, level):
        # Keyed by label too: a category shared by two typefaces lists its files under both, and they are dropped below.
        if (category, label) in seen or SKIP.search(category): return
        seen.add((category, label))
        rows.extend({'title': title, 'typeface': typeface(label), 'category': label, 'path': path} for title in members(category, 'file') if JPEG.search(title))
        if level < depth:
            for sub in members(category, 'subcat'):
                if sub not in typefaces: walk(sub, label, path + [sub], level + 1)  # a typeface inside is walked under its own name
    for category in sorted(typefaces):
        walk(category, category, [], 0)
        print(f'{typeface(category)}: {len(rows)} files so far', flush=True)
    labels = {}
    for row in rows: labels.setdefault(row['title'], []).append(row)
    # A file filed under two typefaces cannot say which of them it shows.
    kept = [found[0] for found in labels.values() if len({row['typeface'] for row in found}) == 1]
    with (OUT / 'files.jsonl').open('w') as out:
        for row in kept: out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{len(kept)} files, {len(labels) - len(kept)} under several typefaces left out', flush=True)


def photo(info):
    meta = {item['name']: item['value'] for item in info.get('commonmetadata', [])}
    return info.get('mime') == 'image/jpeg' and bool(meta.get('Make') or meta.get('Model'))


def details():
    rows = [json.loads(line) for line in (OUT / 'files.jsonl').open()]
    done = {json.loads(line)['title'] for line in (OUT / 'info.jsonl').open()} if (OUT / 'info.jsonl').exists() else set()
    todo = [row for row in rows if row['title'] not in done]
    with (OUT / 'info.jsonl').open('a') as out:
        for start in range(0, len(todo), 50):
            batch = {row['title']: row for row in todo[start:start + 50]}
            for page in api(action='query', prop='imageinfo', titles='|'.join(batch), iiprop='url|size|mime|sha1|extmetadata|commonmetadata',
                            iiurlwidth=WIDTH, iiextmetadatafilter='LicenseShortName|Artist|Credit|UsageTerms'):
                for file in page.get('pages', []):
                    if file['title'] not in batch or not file.get('imageinfo'): continue
                    info = file['imageinfo'][0]; meta = info.get('extmetadata', {})
                    field = lambda key: re.sub(r'<[^>]+>', '', meta.get(key, {}).get('value', '')).strip() or None
                    out.write(json.dumps({**batch[file['title']], 'kind': 'photo' if photo(info) else 'other', 'mime': info.get('mime'),
                                          'width': info.get('width'), 'height': info.get('height'), 'sha1': info.get('sha1'),
                                          'url': info.get('thumburl') or info.get('url'), 'page': info.get('descriptionurl'),
                                          'licence': field('LicenseShortName'), 'artist': field('Artist'), 'credit': field('Credit')}, ensure_ascii=False) + '\n')
            print(f'{min(start + 50, len(todo))}/{len(todo)} files read', flush=True)


def images():
    folder = OUT / 'images'; folder.mkdir(parents=True, exist_ok=True)
    photos = [row for row in map(json.loads, (OUT / 'info.jsonl').open()) if row['kind'] == 'photo']
    silent = 0  # downloads in a row that got no answer
    for index, row in enumerate(photos):
        target = folder / (hashlib.sha1(row['title'].encode()).hexdigest()[:16] + '.jpg')
        if target.exists(): continue
        try: data = get(row['url'])
        except urllib.error.HTTPError as error: print(f"{row['title']}: {error.code}, skipped", flush=True); continue
        except (urllib.error.URLError, OSError) as error:  # a timeout or a dropped connection: a rerun fetches it
            silent += 1
            if silent == 5: raise SystemExit(f'5 downloads in a row got no answer, the last {row["url"]}: {error}')
            print(f"{row['title']}: {error}, skipped", flush=True); continue
        silent = 0
        if data[:3] != b'\xff\xd8\xff': print(f"{row['title']}: not a JPEG, skipped", flush=True); continue
        target.write_bytes(data)
        with (OUT / 'images.jsonl').open('a') as out: out.write(json.dumps({'title': row['title'], 'file': target.name, 'sha256': hashlib.sha256(data).hexdigest()}, ensure_ascii=False) + '\n')
        if index % 100 == 0: print(f'{index}/{len(photos)} photos', flush=True)


if __name__ == '__main__':
    {'list': listing, 'info': details, 'images': images}[sys.argv[1]]()
