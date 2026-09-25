"""Type specimen books from the Internet Archive, for a benchmark of historical print.

A founder's specimen book prints each typeface under its own name ("18 Point Lining Caslon
Oldstyle No. 540") and then sets lines in it, so every specimen is a real printed line with its
label beside it, scanned from paper. Books from before 1929 are public domain in the US; a book
is taken only when the Archive also marks it out of copyright. The OCR's word boxes (hOCR) come
with the pages, so captions can later be found and the lines below them boxed. One request a
second; archive.org's robots.txt closes only /control/ and /report/.

    python scripts/specimen_books.py list              the specimen books -> books.jsonl
    python scripts/specimen_books.py pages [id ...]    page images and word boxes (default: the founders' books below)
"""
import json, re, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.data/specimens'
AGENT = 'gpu-font font-recognition research (https://github.com/dy/gpu-font)'
WIDTH = 1600  # the Archive's page images at this width: 18 point type stands about 40 px high
QUERY = ('mediatype:texts AND year:[1450 TO 1928] AND (subject:("type specimens" OR "printing specimens" OR "type and type-founding specimens" '
         'OR "printing -- specimens" OR "type and type-founding -- specimens") OR title:("specimen of printing types" OR "specimen book" '
         'OR "specimens of type" OR "type specimen" OR "specimen of type" OR "book of type" OR "specimen of modern printing types"))')
# A subject search also finds books printed as specimens of fine printing (The Faerie Queene); a specimen book says so in its title.
SPECIMEN = re.compile(r'specim|type|typograph|caract[eè]r|schrift|letter|fonderie|foundr', re.I)
# The founders' own books: large, captioned, and spanning the American and French trade.
FOUNDERS = ['americanspecimen00amerrich', 'specimenbookcata00amer', 'specimensofprint00geor', 'bookoftypespecim00barnrich',
            'monotypespecimen00lansrich', 'abridgedspecimen00keysrich', 'lelivrettypograp00fond', 'cu31924032183760']


def get(url):
    time.sleep(1)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': AGENT}), timeout=180) as response: return response.read()


def specimen_book(doc):
    """A type specimen book from before 1929, judged by its search record."""
    year = int(str(doc.get('year') or '0')[:4] or 0)
    return 0 < year <= 1928 and bool(SPECIMEN.search(str(doc.get('title') or '')))


def listing():
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ('identifier', 'title', 'year', 'imagecount', 'publisher', 'creator')
    url = 'https://archive.org/advancedsearch.php?' + urllib.parse.urlencode([('q', QUERY), ('rows', 2000), ('output', 'json')] + [('fl[]', f) for f in fields])
    docs = [doc for doc in json.loads(get(url))['response']['docs'] if specimen_book(doc)]
    with (OUT / 'books.jsonl').open('w') as out:
        for doc in sorted(docs, key=lambda doc: str(doc.get('year'))): out.write(json.dumps({field: doc.get(field) for field in fields}, ensure_ascii=False) + '\n')
    print(f'{len(docs)} specimen books, {sum(int(doc.get("imagecount") or 0) for doc in docs)} pages', flush=True)


def pages(ids):
    for identifier in ids or FOUNDERS:
        meta = json.loads(get(f'https://archive.org/metadata/{identifier}'))
        status = meta.get('metadata', {}).get('possible-copyright-status')
        if status != 'NOT_IN_COPYRIGHT': print(f'{identifier}: the Archive marks it {status!r}, skipped', flush=True); continue
        folder = OUT / identifier / 'pages'; folder.mkdir(parents=True, exist_ok=True)
        (OUT / identifier / 'metadata.json').write_text(json.dumps(meta['metadata'], ensure_ascii=False, indent=1))
        hocr = OUT / identifier / 'hocr.html'
        if not hocr.exists() and any(f['name'] == f'{identifier}_hocr.html' for f in meta['files']):
            try: hocr.write_bytes(get(f'https://archive.org/download/{identifier}/{identifier}_hocr.html'))
            except (urllib.error.URLError, OSError) as error: print(f'{identifier}: word boxes not served ({error}); a rerun tries again', flush=True)
        count, silent = int(meta['metadata'].get('imagecount') or 0), 0
        for index in range(count):
            target = folder / f'n{index:04}.jpg'
            if target.exists(): continue
            try: data = get(f'https://archive.org/download/{identifier}/page/n{index}_w{WIDTH}.jpg')
            except urllib.error.HTTPError as error: print(f'{identifier} n{index}: {error.code}, skipped', flush=True); continue
            except (urllib.error.URLError, OSError) as error:  # a timeout or a dropped connection: a rerun fetches it
                silent += 1
                if silent == 5: raise SystemExit(f'5 pages in a row got no answer, the last {identifier} n{index}: {error}')
                continue
            silent = 0
            if data[:3] == b'\xff\xd8\xff': target.write_bytes(data)
        print(f'{identifier}: {len(list(folder.iterdir()))} of {count} pages', flush=True)


if __name__ == '__main__':
    {'list': listing, 'pages': lambda: pages(sys.argv[2:])}[sys.argv[1]]()
