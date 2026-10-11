"""Static instances of the inventory's variable fonts: one TTF per named instance (fvar), so a file-built catalog indexes,
and the previews draw, every style a variable family offers rather than the file's default alone (PP Kyoto's default is
Thin; PP Monument's Bold had no row). Files go to .data/fonts-open/instances/<family id>/<file stem>@<style slug>.ttf,
pinned by git blob as the inventory's own files are; the manifest .data/fonts-open/instances.json lists each variable
face's instances and each family's default instance, the one of its selected file nearest Regular upright, which carries
the family's twins in the catalog (scripts/catalog-faces.mjs, scripts/previews.py).

    node scripts/python.mjs scripts/instances.py [--workers 6]
"""
import argparse
import hashlib
import io
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[1]
OPEN = ROOT/'bench/open-fonts.json'
FOLDER = 'instances'
SLOPE = re.compile(r'\b(italic|oblique|slanted)\b', re.I)
WEIGHTS = {100: 'Thin', 200: 'ExtraLight', 300: 'Light', 400: 'Regular', 500: 'Medium', 600: 'SemiBold', 700: 'Bold', 800: 'ExtraBold', 900: 'Black'}


def blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-') or 'instance'


def named_instances(font):
    """The fvar named instances as (style name, coordinates over every axis), a second instance at the same coordinates
    dropped, then the nine CSS weights the weight axis spans that no instance names (PP Telegraf names three of its 24
    styles, PP Pangram Sans Rounded none), at the other axes' defaults, slanted when the file is."""
    names = font['name']; seen = set(); found = []
    def add(name, coordinates):
        key = tuple(sorted(coordinates.items()))
        if key in seen: return
        seen.add(key); found.append((name, coordinates))
    for instance in font['fvar'].instances:
        add(names.getDebugName(instance.subfamilyNameID) or f'Instance {len(found) + 1}', {tag: float(value) for tag, value in instance.coordinates.items()})
    axes = {a.axisTag: a for a in font['fvar'].axes}
    if 'wght' in axes:
        slanted = font['post'].italicAngle != 0
        for weight, name in WEIGHTS.items():
            if axes['wght'].minValue <= weight <= axes['wght'].maxValue:
                add(name + (' Italic' if slanted else ''), {tag: float(weight if tag == 'wght' else a.defaultValue) for tag, a in axes.items()})
    return found


def style_of(name, coordinates, axes, weight):
    """Weight, slope and width of an instance from its coordinates: wght, slnt and ital (italic past the axis's lower end,
    PP Monument's running 100 to 900), the style name where the font has no axis for the slope; the file's weight when
    it has no weight axis."""
    low, _, high = axes.get('ital', (0, 0, 0))
    italic = bool(SLOPE.search(name)) or coordinates.get('slnt', 0) != 0 or (high > low and coordinates.get('ital', low) > low)
    return {'weight': round(coordinates['wght']) if 'wght' in coordinates else weight, 'italic': italic, 'width': round(coordinates['wdth']) if 'wdth' in coordinates else 100}


def plan(inventory):
    """Every variable face's instances, by face path, and each family's default instance: records without blobs yet."""
    store = ROOT/inventory['store']; faces = {}; families = {}
    for family in inventory['families']:
        if family.get('excluded'): continue
        for face in family['faces']:
            if not face.get('axes'): continue
            with TTFont(store/face['path'], lazy=True) as font:
                axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in font['fvar'].axes}; found = named_instances(font)
            if not found: continue
            stem = Path(face['path']).stem; used = set(); records = []
            for name, coordinates in found:
                key = slug(name); n = 2
                while key in used: key = f'{slug(name)}-{n}'; n += 1
                used.add(key)
                records.append({'id': f"{family['id']}/{stem}@{key}", 'path': f"{FOLDER}/{family['id']}/{stem}@{key}.ttf", 'styleName': name,
                                'axes': coordinates, **style_of(name, coordinates, axes, face['weight'])})
            faces[face['path']] = records
            if face['path'] == family['selected']:
                families[family['id']] = {'default': min(records, key=lambda r: (r['italic'], abs(r['weight'] - 400), abs(r['width'] - 100)))['id']}
    return faces, families


def instantiate(source, coordinates):
    """The font at the coordinates. A positioning variation store the instancer cannot read (PP Palma's, PP Valve's) is
    dropped and tried again: the letters are right and the kerning stays at the default's."""
    with TTFont(source, recalcTimestamp=False) as font:
        try: instantiateVariableFont(font, coordinates, inplace=True)
        except Exception:
            if 'GDEF' not in font or getattr(font['GDEF'].table, 'VarStore', None) is None: raise
            font.close(); font = TTFont(source, recalcTimestamp=False); font['GDEF'].table.VarStore = None
            instantiateVariableFont(font, coordinates, inplace=True)
        stream = io.BytesIO(); font.save(stream)
    return stream.getvalue()


def instance(job):
    """One static instance, written unless already there: (source file, coordinates, target file) → the target's git blob,
    or the error that stopped it."""
    source, coordinates, target = job; target = Path(target)
    try:
        if not target.exists():
            data = instantiate(source, coordinates)
            target.parent.mkdir(parents=True, exist_ok=True); part = target.with_suffix('.part'); part.write_bytes(data); part.replace(target)
        return blob(target.read_bytes()), None
    except Exception as error: return None, f'{type(error).__name__}: {error}'


def build(inventory, workers, run=None):
    """The manifest, every instance file written: faces instancing fails on leave the manifest, so their file row stays."""
    store = ROOT/inventory['store']; faces, families = plan(inventory)
    jobs = [(str(store/path), r['axes'], str(store/r['path'])) for path, records in faces.items() for r in records]
    # Workers restart after a few fonts: the largest variable files take a gigabyte each to instance.
    def pooled(jobs):
        with Pool(workers, maxtasksperchild=4) as pool: return pool.map(instance, jobs, chunksize=1)
    results = list((run or pooled)(jobs))
    blobs = dict(zip((job[2] for job in jobs), results)); failed = []
    for path in list(faces):
        for r in faces[path]:
            r['blob'], error = blobs[str(store/r['path'])]
            if error: failed.append((r['id'], error))
        if any(r['blob'] is None for r in faces[path]):
            del faces[path]; families = {f: v for f, v in families.items() if not v['default'].startswith(path.split('/')[0] + '-') or any(v['default'] == r['id'] for rs in faces.values() for r in rs)}
    kept = {store/r['path'] for records in faces.values() for r in records}
    for stale in (store/FOLDER).glob('*/*.ttf'):
        if stale not in kept: stale.unlink()
    manifest = {'version': 1, 'store': inventory['store'], 'inventorySha256': hashlib.sha256(OPEN.read_bytes()).hexdigest(), 'faces': faces, 'families': families}
    return manifest, failed


def main():
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=6); a = p.parse_args()
    inventory = json.loads(OPEN.read_text()); manifest, failed = build(inventory, a.workers)
    target = ROOT/inventory['store']/'instances.json'; target.write_text(json.dumps(manifest, indent=1) + '\n')
    for face_id, error in failed: print(f'{face_id}: {error}', file=sys.stderr)
    print(f"{sum(len(r) for r in manifest['faces'].values())} instances of {len(manifest['faces'])} variable files, {len(manifest['families'])} families with an instanced default; {len(failed)} failed")


if __name__ == '__main__': main()
