"""Local pilot fixtures. Font binaries and generated crops stay in .data/."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data'


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config):
    fonts = config['fonts']
    ids = [f['id'] for f in fonts]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Expected unique, nonempty family IDs')
    if any(f['split'] not in ('train', 'unseen-dev') for f in fonts):
        raise ValueError('Unknown family split')
    train = set(config['training']['texts'])
    evaluation = config['evaluation']['texts']
    if not train or len(evaluation) < 2 or len(evaluation) != len(set(evaluation)) or train.intersection(evaluation):
        raise ValueError('Training, prototype, and query texts must be disjoint')
    if any(not t.strip() for t in [*train, *evaluation]):
        raise ValueError('Blank text')
    return config


def check_coverage(font, texts):
    cmap = font.getBestCmap() or {}
    missing = sorted({c for t in texts for c in t if ord(c) not in cmap})
    if missing:
        raise ValueError(f'Missing glyphs: {missing}')


def import_fonts(config, source):
    manifest_path = ROOT / 'bench/fonts.json'
    old = {f['id']: f for f in read_json(manifest_path)['fonts']} if manifest_path.exists() else {}
    entries = []
    for entry in config['fonts']:
        src = source / entry['file']
        digest = sha(src)
        if entry['id'] in old and old[entry['id']]['sourceSha256'] != digest:
            raise ValueError(f"Source changed for {entry['id']}; review bench/fonts.json before replacing the pin")
        with TTFont(src, recalcTimestamp=False) as font:
            check_coverage(font, config['training']['texts'] + config['evaluation']['texts'])
            names = font['name']
            family = names.getDebugName(16) or names.getDebugName(1)
            license_text = names.getDebugName(13) or ''
            if 'Open Font License' not in license_text:
                raise ValueError(f'Expected embedded OFL notice in {src.name}')
            axes = {a.axisTag: a.defaultValue for a in font['fvar'].axes} if 'fvar' in font else {}
            if 'wght' in axes:
                axes['wght'] = 400
            if 'wdth' in axes:
                axes['wdth'] = 100
            metadata = dict(entry, family=family, sourceSha256=digest,
                            sourceVersion=names.getDebugName(5), sourceRevision=None,
                            license=license_text, licenseUrl=names.getDebugName(14),
                            copyright=names.getDebugName(0), axes=axes)
            if axes:
                instantiateVariableFont(font, axes, inplace=True)
            if font['OS/2'].usWeightClass != 400 or font['post'].italicAngle != 0:
                raise ValueError(f'Expected upright regular face: {src.name}')
            dest = DATA / 'fonts' / f"{entry['id']}.ttf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            font.save(dest)
            metadata['instanceSha256'] = sha(dest)
            entries.append(metadata)
    write_json(manifest_path, {'version': 1, 'source': 'fontr/data/fonts_collected/google (local collection; upstream revisions unknown)', 'fonts': entries})
    print(f'Imported {len(entries)} verified regular faces')


def render_sample(font_path, text, size, dpr, polarity, blur=0, quality=None):
    font = ImageFont.truetype(str(font_path), size=round(size * dpr))
    bbox = font.getbbox(text)
    pad = 6 * dpr
    image = Image.new('RGB', (bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2), 0)
    bg, fg = (255, 0) if polarity == 'dark' else (0, 255)
    image.paste((bg, bg, bg), (0, 0, *image.size))
    ImageDraw.Draw(image).text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(fg, fg, fg))
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if quality:
        buffer = io.BytesIO()
        image.save(buffer, 'JPEG', quality=quality)
        image = Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
    return image.convert('RGBA')


def render(config):
    rng = random.Random(config['seed'])
    samples = []
    folder = DATA / 'pillow'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read_json(ROOT / 'bench/fonts.json')
    for face in manifest['fonts']:
        font_path = DATA / 'fonts' / f"{face['id']}.ttf"
        if sha(font_path) != face['instanceSha256']:
            raise ValueError(f'Instance checksum failed: {face["id"]}')
        plans = []
        for i, text in enumerate(config['evaluation']['texts']):
            for size in config['evaluation']['sizes']:
                for dpr in config['evaluation']['dprs']:
                    for polarity in config['evaluation']['polarities']:
                        plans.append(dict(role='prototype' if i == 0 else 'query', text=text, size=size, dpr=dpr, polarity=polarity, blur=0, quality=None))
        if face['split'] == 'train':
            for _ in range(config['training']['samplesPerFamily']):
                plans.append(dict(role='train', text=rng.choice(config['training']['texts']),
                                  size=rng.randint(12, 48), dpr=rng.choice([1, 2]), polarity=rng.choice(['dark', 'light']),
                                  blur=rng.choice([0, 0, .3, .6]), quality=rng.choice([None, None, 70, 90])))
        for i, plan in enumerate(plans):
            image = render_sample(font_path, **{k: v for k, v in plan.items() if k != 'role'})
            sample_id = f"{face['id']}-{i:04d}"
            raw = folder / f'{sample_id}.rgba'
            raw.write_bytes(image.tobytes())
            if plan['role'] != 'train':
                image.save(folder / f'{sample_id}.png')
            samples.append(dict(id=sample_id, family=face['id'], split=face['split'], renderer='pillow',
                                width=image.width, height=image.height, raw=str(raw.relative_to(ROOT)), **plan))
    write_json(folder / 'samples.json', {'renderer': f'Pillow {PIL.__version__}', 'configSha256': sha(ROOT / 'bench/pilot.json'), 'fontsSha256': sha(ROOT / 'bench/fonts.json'), 'samples': samples})
    print(f'Rendered {len(samples)} crops ({sum(s["role"] == "train" for s in samples)} training)')


def preview(config):
    manifest = read_json(DATA / 'prepared.json')
    for key, path in [('configSha256', ROOT / 'bench/pilot.json'), ('fontsSha256', ROOT / 'bench/fonts.json'), ('preparationSha256', ROOT / 'src/prepare.mjs'), ('tensorSha256', DATA / 'prepared.f32')]:
        if sha(path) != manifest[key]:
            raise ValueError(f'Stale or corrupt preparation: {path}')
    width, height = manifest['width'], manifest['height']
    if (DATA / 'prepared.f32').stat().st_size != len(manifest['samples']) * height * width * 4:
        raise ValueError('Prepared tensor length mismatch')
    pixels = np.fromfile(DATA / 'prepared.f32', dtype='<f4').reshape(len(manifest['samples']), height, width)
    sheet = Image.new('RGB', (1010, len(config['fonts']) * 86 + 42), '#ededed')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), 'Family / Pillow source', fill='black')
    draw.text((430, 10), 'Pillow -> shared JS (128 x 32)', fill='black')
    draw.text((720, 10), 'Chromium -> shared JS (128 x 32)', fill='black')
    for row, font in enumerate(config['fonts']):
        index, sample = next((i, s) for i, s in enumerate(manifest['samples']) if s['family'] == font['id'] and s['role'] == 'query' and s['renderer'] == 'pillow' and s['size'] == 28 and s['dpr'] == 1 and s['polarity'] == 'dark')
        y = row * 86 + 38
        draw.text((12, y), f"{font['id']} ({font['split']})", fill='black')
        source = Image.open(ROOT / sample['raw'].replace('.rgba', '.png')).convert('RGB')
        source.thumbnail((400, 50))
        sheet.paste(source, (12, y + 18))
        normalized = Image.fromarray(np.uint8(np.clip(pixels[index], 0, 1) * 255)).resize((256, 64))
        sheet.paste(normalized, (430, y))
        browser = next((i for i, s in enumerate(manifest['samples']) if s['family'] == font['id'] and s['renderer'] == 'chromium' and s['text'] == sample['text'] and s['size'] == 28 and s['dpr'] == 1 and s['polarity'] == 'dark'), None)
        if browser is not None:
            normalized = Image.fromarray(np.uint8(np.clip(pixels[browser], 0, 1) * 255)).resize((256, 64))
            sheet.paste(normalized, (720, y))
    dest = ROOT / 'bench/contact-sheet.png'
    sheet.save(dest)
    print(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['import', 'render', 'preview'])
    parser.add_argument('--source', type=Path, help='Directory of collected Google Font files (import only)')
    args = parser.parse_args()
    config = validate_config(read_json(ROOT / 'bench/pilot.json'))
    if args.command == 'import':
        if not args.source:
            parser.error('import requires --source')
        import_fonts(config, args.source.expanduser())
    elif args.command == 'render':
        render(config)
    else:
        preview(config)
