"""Compare source crops with the actual tensors used by both training runs."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/robustness'
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run():
    manifest = json.loads((DATA / 'samples.json').read_text())
    config = json.loads((ROOT / 'bench/robustness.json').read_text())
    samples = manifest['samples']
    if manifest['configSha256'] != sha(ROOT / 'bench/robustness.json'):
        raise ValueError('Stale preview configuration')
    offset = 0
    for sample in samples:
        if sample['offset'] != offset or any(type(sample[k]) is not int or sample[k] <= 0 for k in ['width', 'height']):
            raise ValueError('Invalid preview source geometry')
        offset += sample['width'] * sample['height'] * 4
    if (DATA / 'source.rgba').stat().st_size != offset:
        raise ValueError('Preview source length mismatch')
    if manifest['rawSha256'] != sha(DATA / 'source.rgba'):
        raise ValueError('Preview source checksum mismatch')
    arrays = {}
    for name, settings in config['inputs'].items():
        prepared_path = DATA / name / 'prepared.json'
        prepared = json.loads(prepared_path.read_text())
        report = json.loads((ROOT / f'bench/robustness-{name}.json').read_text())
        if any(prepared.get(key) != value for key, value in manifest.items()) or any(prepared.get(key) != value for key, value in settings.items()):
            raise ValueError(f'Stale preview inputs: {name}')
        if report['manifestSha256'] != sha(prepared_path):
            raise ValueError(f'Preview inputs differ from training report: {name}')
        tensor = DATA / name / 'prepared.f32'
        if tensor.stat().st_size != len(samples) * settings['width'] * settings['height'] * 4:
            raise ValueError(f'Preview tensor length mismatch: {name}')
        if prepared['tensorSha256'] != sha(tensor):
            raise ValueError(f'Preview tensor checksum mismatch: {name}')
        array = np.memmap(tensor, dtype='<f4', mode='r', shape=(len(samples), settings['height'], settings['width']))
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(f'Invalid preview pixels: {name}')
        arrays[name] = array
    selected = [(i, s) for i, s in enumerate(samples) if s['role'] == 'query' and s['renderer'] == 'chromium'
                and s['family'] in ['inter', 'roboto'] and s['condition'] in ['tight', 'short', 'rotate-right']
                and s['text'] == 'Quiet rivers flow' and s['size'] == 28 and s['dpr'] == 1 and s['polarity'] == 'dark']
    if not selected:
        raise ValueError('No matching preview samples')
    sheet = Image.new('RGB', (1060, 56 + len(selected) * 132), '#ededed')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(ROOT / '.data/fonts/inter.ttf'), 15)
    for x, label in [(16, 'Source crop'), (408, '128 × 32 input (shown at 2×)'), (748, '256 × 64 input (shown at 1×)')]:
        draw.text((x, 15), label, font=font, fill='black')
    with (DATA / 'source.rgba').open('rb') as raw:
        for row, (index, sample) in enumerate(selected):
            y = 56 + row * 132
            draw.text((16, y), f"{sample['family']} · {sample['condition']}", font=font, fill='black')
            raw.seek(sample['offset'])
            image = Image.frombytes('RGBA', (sample['width'], sample['height']), raw.read(sample['width'] * sample['height'] * 4)).convert('RGB')
            image.thumbnail((360, 92))
            sheet.paste(image, (16, y + 28))
            for x, name in [(408, 'pilot'), (748, 'detail')]:
                pixels = np.uint8(np.rint(arrays[name][index] * 255))
                image = Image.fromarray(pixels).resize((256, 64), Image.Resampling.NEAREST)
                sheet.paste(image, (x, y + 28))
    path = ROOT / 'bench/robustness-inputs.png'
    sheet.save(path)
    return path


if __name__ == '__main__':
    print(run())
