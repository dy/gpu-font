"""Plot actual prepared tensors, paired with their checksum-verified source crop."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train.focus import ROOT, DATA, load_data
from train.robustness import sha, read


def run():
    config, manifest, pixels = load_data()
    for policy in config['policies']:
        if read(ROOT / f'bench/focus-{policy}.json')['manifestSha256'] != sha(DATA / 'prepared.json'):
            raise ValueError('Preview and report refer to different inputs')
    if sha(DATA / 'source.rgba') != manifest['rawSha256']:
        raise ValueError('Source checksum mismatch')
    candidates = [(i, s) for i, s in enumerate(manifest['samples']) if s['role'] == 'query'
                  and s['renderer'] == 'chromium' and s['family'] == config['target']
                  and s['condition'] == 'clean' and s['policy'] == 'line']
    long = max(candidates, key=lambda item: len(item[1]['text']))
    short = min(candidates, key=lambda item: len(item[1]['text']))
    rows = [long, short]
    font = ImageFont.load_default(size=18)
    image = Image.new('RGB', (1390, 100 + len(rows) * 220), '#eeeeee')
    draw = ImageDraw.Draw(image)
    for x, label in [(20, 'Source crop'), (340, 'Whole line, 128 x 32'), (870, 'Local window, 128 x 32')]:
        draw.text((x, 20), label, font=font, fill='#111111')
    with (DATA / 'source.rgba').open('rb') as raw:
        for row, (index, sample) in enumerate(rows):
            y = 80 + row * 220
            raw.seek(sample['offset'])
            source = Image.frombytes('RGBA', (sample['width'], sample['height']), raw.read(sample['width'] * sample['height'] * 4)).convert('RGB')
            source.thumbnail((300, 128), Image.Resampling.LANCZOS)
            image.paste(source, (20, y))
            for x, policy in [(340, 'line'), (870, 'patch')]:
                match = next(i for i, s in enumerate(manifest['samples']) if s['family'] == sample['family']
                             and s['role'] == sample['role'] and s['renderer'] == sample['renderer']
                             and s['plan'] == sample['plan'] and s['condition'] == sample['condition'] and s['policy'] == policy)
                tile = Image.fromarray(np.round(pixels[match, 0] * 255).astype(np.uint8))
                image.paste(tile.resize((512, 128), Image.Resampling.NEAREST), (x, y))
            draw.text((20, y + 145), f'Inter · {sample["text"]}', font=font, fill='#111111')
    path = ROOT / 'bench/focus-inputs.png'
    image.save(path)
    return path


if __name__ == '__main__':
    print(run())
