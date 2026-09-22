"""A bounded real-face pilot: 10 families × two weights × normal/italic."""
import argparse
import random
from pathlib import Path
from PIL import Image
from scripts.fixtures import render_sample
from train.robustness import read, write, sha
from train.hundred_data import band, transform
from train.preparation import TEXTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/faces'
TEST = ['G', 'q', 'zr', 'hx', 'sprigs', 'dovetail', 'Woven reeds shimmer', 'Amber lanterns glow']


def plans():
    texts = {'train': read(ROOT/'bench/hundred-texts.json')['train'], 'validation': TEXTS, 'test': TEST}
    pools = [{t.lower() for t in values if len(t)>1} for values in texts.values()]
    if any(a & b for i,a in enumerate(pools) for b in pools[i+1:]): raise ValueError('Face text leakage')
    historical = {t.lower() for values in read(ROOT/'bench/hundred-texts.json').values() for t in values if len(t)>1}
    if pools[2] & historical: raise ValueError('Face final text repeats historical corpus')
    rng = random.Random(20260926)
    rows = []
    for role, values in texts.items():
        for index,text in enumerate(values):
            rows.append({'role':role, 'index':len(rows), 'text':text, 'length':band(text),
                         'size':rng.choice([18,24,32,44,60]) if role=='train' else [18,32,56][index%3],
                         'dpr':1, 'light':False})
    write(DATA/'plans.json', rows)
    write(ROOT/'bench/face-texts.json', texts)


def render():
    plans = read(DATA/'plans.json'); manifest = read(ROOT/'bench/font-faces.json')
    browser = read(DATA/'browser.json'); rows = []
    if browser['rawSha256'] != sha(DATA/'browser.gray'): raise ValueError('Changed browser pixels')
    if browser['facesSha256'] != sha(ROOT/'bench/font-faces.json'): raise ValueError('Changed face manifest')
    with (DATA/'source.gray').open('wb') as out:
        def append(base, plan, face, renderer):
            for condition in ['clean','augmented'] if plan['role']=='train' else ['clean','small','rotate','shadow']:
                image = transform(base,plan,condition)
                rows.append({**plan, 'family':face['id'], 'fontFamily':face['family'], 'weight':face['weight'], 'style':face['style'],
                             'condition':condition, 'renderer':renderer, 'offset':out.tell(), 'width':image.width, 'height':image.height})
                out.write(image.tobytes())
        for face in manifest['faces']:
            path = DATA/f'{face["id"]}.ttf'
            if sha(path) != face['instanceSha256']: raise ValueError('Changed face')
            for plan in plans:
                append(render_sample(path,plan['text'],plan['size'],plan['dpr'],'dark'),plan,face,'pillow')
        faces = {f['id']:f for f in manifest['faces']}
        with (DATA/'browser.gray').open('rb') as raw:
            for row in browser['samples']:
                if row['offset'] != raw.tell(): raise ValueError('Invalid browser offset')
                image = Image.frombytes('L',(row['width'],row['height']),raw.read(row['width']*row['height']))
                append(image,row,faces[row['family']],'chromium')
            if raw.read(1): raise ValueError('Trailing browser pixels')
    write(DATA/'source.json', {'samples':rows,'rawSha256':sha(DATA/'source.gray'),'facesSha256':sha(ROOT/'bench/font-faces.json'),
                              'textsSha256':sha(ROOT/'bench/face-texts.json'),'generatorSha256':sha(Path(__file__))})
    print(f'Faces: {len(rows)} source images',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plans','render']);args=p.parse_args();globals()[args.command]()
