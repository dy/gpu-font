"""The previews stored with the site: a 64-pixel black-and-white picture of every face of a shipped catalog but Google
Fonts' (whose faces the page sets in their own font) and DaFont's (whose own stored previews the page loads from its site),
filed under the SHA-1 of the face id as previews.mjs finds it, so no catalog carries an address. Each shows its family's
name in the face, so no two rows repeat one phrase: Adobe Fonts' is the tester capture of it (a word line where the face
lacks the name's letters or has not been captured yet); every other face is its pinned font file setting it. Pictures of
faces no longer shipped are removed; faces with nothing to draw are listed in bench/previews.json.

    python scripts/previews.py
"""
import hashlib, io, json, os
from multiprocessing import Pool
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT, REPORT = ROOT / 'previews', ROOT / 'bench/previews.json'
HEIGHT, WIDTH, MARGIN = 64, 1280, .12  # a row's line at 2x; wider than any row is never seen; white above and below the ink
# Adobe Fonts' captures: the snapshot its catalog was compiled from, and the live archive, which holds the name lines.
ADOBE = [ROOT / '.data/previews/catalog-v1/adobe-fonts-snapshot', ROOT / '.data/previews/catalog-v1/adobe-fonts']
RECIPES = ['name-v1', 'words-a-v1', 'words-b-v1', 'latin-lower-v1', 'latin-upper-v1', 'provided-v1', 'digits-v1']  # the line shown, first found


def path(face_id):
    digest = hashlib.sha1(face_id.encode()).hexdigest()
    return OUT / digest[:2] / f'{digest[2:16]}.webp'


def sources():
    """Each face id's picture: ('image', file, crop box or None) for a captured one, ('font', file, weight, italic, family name)
    for a file."""
    found = captured(ADOBE)
    inventory = json.loads((ROOT / 'bench/open-fonts.json').read_text())
    for family in inventory['families']:
        for face in family['faces']:  # the id scripts/catalog-files.mjs gives it
            found[f"{family['id']}/{face['path'].split('/')[-1].rsplit('.', 1)[0]}"] = ('font', str(ROOT / inventory['store'] / face['path']), face['weight'], face['italic'], family['family'])
    return found


def captured(archives):
    """Each captured face's picture from tester archives: the first line RECIPES names that it has, from whichever archive
    holds it (the name line is only in the live archive)."""
    lines = {}
    for archive in archives:
        for record in map(json.loads, (archive / 'manifest.jsonl').open()):
            lines.setdefault(record['faceId'], {})[record['recipeId']] = (archive, record)
    found = {}
    for face_id, held in lines.items():
        recipe = next((r for r in RECIPES if r in held), None)
        if not recipe: continue
        archive, record = held[recipe]; g = record['region']
        found[face_id] = ('image', str(archive / record['image']['path']), (g['x'], g['y'], g['x'] + g['width'], g['y'] + g['height']))
    return found


def cut(image, faint):
    """The level below which a pixel is ink: the middle, 128, for black ink. A faint face, one that reaches no pixel darker
    than that (a colour font captured in greys, strokes finer than a pixel), keeps every mark clearly darker than white:
    anything in the darker five sixths from its darkest pixel to white."""
    darkest = image.getextrema()[0]
    return darkest + (255 - darkest) * 5 / 6 if faint else 128


def unhinted(font_file):
    """The font without its TrueType hinting programs, which FreeType fails to run in a few files (Ume P Gothic S4, XW Zar)
    and which change nothing at this size."""
    font = TTFont(font_file)
    for tag in ('fpgm', 'prep', 'cvt ', 'hdmx', 'VDMX', 'LTSH'):
        if tag in font: del font[tag]
    if 'glyf' in font:
        for name in font.getGlyphOrder(): font['glyf'][name].removeHinting()
    data = io.BytesIO(); font.save(data); data.seek(0)
    return data


def setting(source, text, weight, italic, size):
    font = ImageFont.truetype(source, size)
    try:  # a variable font at the face's weight, and italic where it has the axis
        axes = font.get_variation_axes()
        font.set_variation_by_axes([min(max(weight if a['name'] in (b'Weight', 'Weight') else 1 if italic and a['name'] in (b'Italic', 'Italic') else a['default'], a['minimum']), a['maximum']) for a in axes])
    except OSError: pass
    # Room by the size, not the face's line box, which some faces leave empty (IM Fell Flowers) and swashes overrun.
    canvas = Image.new('L', (int(font.getlength(text)) + 4 * size, 5 * size), 255)
    ImageDraw.Draw(canvas).text((2 * size, 3 * size), text, font=font, fill=0, anchor='ls')
    return canvas


def wording(cmap, name):
    """What a face sets: its family's name, in capitals when it draws no lowercase, else the letters it has (a face whose
    name is in another script, or holds a sign it lacks)."""
    has = lambda text: all(ord(c) in cmap for c in text if c != ' ')
    return next((t for t in (name, name.upper()) if t.strip() and has(t)), None) or ''.join(c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' if ord(c) in cmap)[:16]


def line(font_file, weight, italic, name, size=96):
    """The face setting its wording, and whether it is faint. A face whose strokes are finer than a pixel at this size
    (Kohinoor Zerone One) is set four times larger; one still faint there has outlines that enclose next to no area, lines to
    be stroked (FifteenTwenty UltraLight), which no browser draws either."""
    with TTFont(font_file, lazy=True, fontNumber=0) as font: text = wording(font.getBestCmap() or {}, name)
    if not text: raise ValueError('no Latin letters')
    try: canvas = setting(font_file, text, weight, italic, size)
    except OSError: canvas = setting(unhinted(font_file), text, weight, italic, size)  # hinting FreeType cannot run
    if canvas.getextrema()[0] < 128: return canvas, size > 96
    if size < 384: return line(font_file, weight, italic, name, 4 * size)[0], True
    raise ValueError('its outlines enclose next to no area: lines to be stroked, which no browser draws')


def picture(item):
    face_id, source = item
    try:
        if source[0] == 'image':
            image = Image.open(source[1]).convert('L')
            if source[2]: image = image.crop(source[2])
            faint = image.getextrema()[0] >= 128
        else: image, faint = line(*source[1:])
        level = cut(image, faint); ink = image.point(lambda v: 255 if v < level else 0).getbbox()
        if not ink: raise ValueError('draws nothing')
        # The ink with a margin: letters fill about four fifths of the height, whatever the face's line box (Zapfino's is vast).
        margin = max(2, round((ink[3] - ink[1]) * MARGIN))
        image = ImageOps.expand(image, margin, fill=255).crop((ink[0], ink[1], ink[2] + 2 * margin, ink[3] + 2 * margin))
        if faint:  # its strokes thickened by the reduction to come, so each keeps at least a pixel
            image = image.filter(ImageFilter.MinFilter(2 * round(image.height / HEIGHT / 2) + 1))
        image = image.resize((max(1, round(image.width * HEIGHT / image.height)), HEIGHT), Image.LANCZOS)
        image = image.crop((0, 0, min(image.width, WIDTH), HEIGHT)); level = cut(image, faint)
        image = image.point(lambda v: 255 if v >= level else 0)
        data = io.BytesIO(); image.save(data, 'WEBP', lossless=True, quality=100, method=6)
        target = path(face_id); target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != data.getvalue(): target.write_bytes(data.getvalue())
        return face_id, len(data.getvalue()), None
    except Exception as error: return face_id, 0, f'{type(error).__name__}: {error}'


def main():
    site = json.loads((ROOT / 'site.json').read_text())
    shipped = {option['id']: [face['id'] for face in json.loads((ROOT / option['file']).read_text())['faces']] for option in site['catalogs'] if option['id'] not in ('google-fonts', 'dafont')}
    found = sources()
    missing = [{'id': face_id, 'reason': 'no source picture or file'} for ids in shipped.values() for face_id in ids if face_id not in found]
    todo = [(face_id, found[face_id]) for ids in shipped.values() for face_id in ids if face_id in found]
    with Pool(os.cpu_count()) as pool: done = pool.map(picture, todo, chunksize=64)
    missing += [{'id': face_id, 'reason': reason} for face_id, _, reason in done if reason]
    kept = {path(face_id) for face_id, _, reason in done if not reason}
    stale = [file for file in OUT.glob('*/*.webp') if file not in kept]
    for file in stale: file.unlink()
    for folder in OUT.glob('*'):
        if folder.is_dir() and not any(folder.iterdir()): folder.rmdir()
    sizes = {face_id: size for face_id, size, reason in done if not reason}
    report = {'height': HEIGHT, 'text': 'the family name', 'catalogs': {catalog: {'faces': len(ids), 'pictures': sum(i in sizes for i in ids), 'bytes': sum(sizes.get(i, 0) for i in ids)} for catalog, ids in shipped.items()},
              'missing': sorted(missing, key=lambda m: m['id'])}
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'missing'}, indent=1), f'\n{len(missing)} faces without a picture; {len(stale)} stale pictures removed')


if __name__ == '__main__':
    main()
