"""Fresh development data and frozen-weight comparison; no final-test selection."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from train.robustness import read, write, sha
from train.ten import batch
from train.ten_model import load_export
from train.hundred import probabilities

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.data/preparation'
TEXTS = ['R', 'm', 'vy', 'kt', 'cedar', 'glyphs', 'Silver marks drift', 'Nimble owls turn']


def plans():
    DATA.mkdir(parents=True, exist_ok=True)
    used = {t.lower() for pool in read(ROOT / 'bench/hundred-texts.json').values() for t in pool}
    if any(t.lower() in used for t in TEXTS if len(t)>1): raise ValueError('Development text repeats historical corpus')
    write(DATA / 'plans.json', {'texts':TEXTS,'seed':20260924,'scope':'Development only; alphabet shared, multi-character strings disjoint from historical corpus.'})


def render():
    fonts = [f for f in read(ROOT/'bench/fonts-100.json')['fonts'] if f['split']=='train']
    browser = read(DATA/'browser.json'); raw = (DATA/'browser.gray').read_bytes()
    rows=[]
    with (DATA/'source.gray').open('wb') as out:
        def append(image, row):
            original=image
            for condition in ['clean','small','rotate','neighbor']:
                image=original.copy(); regions=np.array(row['regions'],dtype=float); angle=0
                if condition=='small':
                    image=image.resize((max(1,round(image.width*.45)),max(1,round(image.height*.45))),Image.Resampling.LANCZOS)
                    regions[:,:,0]*=image.width/original.width;regions[:,:,1]*=image.height/original.height
                elif condition=='rotate':
                    angle=6 if len(row['text'])%2 else -6
                    image=image.rotate(-angle,Image.Resampling.BICUBIC,expand=True,fillcolor=255)
                    theta=math.radians(angle);matrix=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
                    regions=(regions-[original.width/2,original.height/2])@matrix.T+[image.width/2,image.height/2]
                elif condition=='neighbor':
                    # A visible sliver from a real v in the same font, adjacent to the query.
                    f=ImageFont.truetype(str(ROOT/f'.data/fonts/{row["family"]}.ttf'),row['size'])
                    v=Image.new('L',(row['size']*2,row['size']*2),255);ImageDraw.Draw(v).text((0,0),'v',font=f,fill=0)
                    box=ImageOps.invert(v).getbbox(); ink=ImageOps.invert(original).getbbox()
                    if box and ink:image.paste(v.crop((box[0],box[1],box[0]+2,box[3])),(min(image.width-2,ink[2]+1),max(0,ink[3]-(box[3]-box[1]))))
                rows.append({**row,'condition':condition,'angle':angle,'regions':regions.tolist(),'width':image.width,'height':image.height,'offset':out.tell()})
                out.write(image.tobytes())
        for idx,font in enumerate(fonts):
            path=ROOT/f'.data/fonts/{font["id"]}.ttf'
            if sha(path)!=font['instanceSha256']:raise ValueError('Font changed')
            for i,text in enumerate(TEXTS):
                size=[18,32,56][(idx+i)%3];f=ImageFont.truetype(str(path),size)
                box=f.getbbox(text,anchor='ls');x,y=8-box[0],8-box[1]
                image=Image.new('L',(box[2]-box[0]+16,box[3]-box[1]+16),255)
                ImageDraw.Draw(image).text((x,y),text,font=f,fill=0,anchor='ls')
                regions=[];prefix=''
                for word in text.split(' '):
                    a,b,c,d=f.getbbox(word,anchor='ls');left=x+f.getlength(prefix)
                    regions.append([[left+a,y+b],[left+c,y+b],[left+c,y+d],[left+a,y+d]]);prefix+=word+' '
                append(image,{'family':font['id'],'text':text,'size':size,'renderer':'pillow','regions':regions})
        offset=0
        for row in browser['samples']:
            n=row['width']*row['height']
            if row['offset']!=offset:raise ValueError('Invalid browser offset')
            image=Image.frombytes('L',(row['width'],row['height']),raw[offset:offset+n]);offset+=n
            append(image,{**row,'renderer':'chromium'})
        if offset!=len(raw):raise ValueError('Trailing browser pixels')
    write(DATA/'source.json',{'samples':rows,'sha256':sha(DATA/'source.gray'),'fontsSha256':sha(ROOT/'bench/fonts-100.json'),'generatorSha256':sha(Path(__file__))})
    print(f'{len(rows)} fresh development images',flush=True)


def evaluate():
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    manifest=read(DATA/'prepared.json'); artifact=read(ROOT/'models/hundred/model.json')
    for key,path in [('inputSha256','src/input.mjs'),('lineSha256','src/line.mjs'),('normalizerSha256','src/prepare.mjs')]:
        if manifest[key]!=sha(ROOT/path):raise ValueError('Changed preparation source: '+path)
    model=load_export(artifact).to('mps');families=artifact['fonts']; report={}
    for method,entry in manifest['methods'].items():
        path=DATA/f'{method}.u8'
        if sha(path)!=entry['sha256']:raise ValueError('Changed tensors')
        pixels=np.memmap(path,dtype=np.uint8,mode='r');windows=entry['windows'];logits=[];owners=[]
        with torch.no_grad():
            for start in range(0,len(windows),128):
                chosen=list(range(start,min(start+128,len(windows))))
                logits.append(model(*(t.to('mps') for t in batch(pixels,windows,chosen))).cpu().numpy());owners.extend(windows[i]['source'] for i in chosen)
        if len(set(owners))!=len(manifest['samples']):raise ValueError('Dropped development samples')
        probs=probabilities(np.concatenate(logits),np.array(owners),len(manifest['samples']))
        groups={};predictions=[]
        for sample,p in zip(manifest['samples'],probs):
            label=families.index(sample['family']);n=len(sample['text'].replace(' ',''));length='1' if n==1 else '2-3' if n<=3 else '4-7' if n<=7 else '8+'
            hit=int(p.argmax()==label);top=int(label in p.argsort()[-5:])
            for key in ['all',sample['condition'],'length/'+length,sample['renderer']]:
                g=groups.setdefault(key,{'count':0,'correct':0,'top5':0});g['count']+=1;g['correct']+=hit;g['top5']+=top
            predictions.append({'family':sample['family'],'predicted':families[int(p.argmax())],'correct':bool(hit)})
        for g in groups.values():g['accuracy']=g['correct']/g['count'];g['top5Accuracy']=g['top5']/g['count']
        report[method]={'groups':groups,'preparationMs':entry['timing'],'windows':len(windows)}
        write(DATA/f'{method}-predictions.json',predictions)
        print(method,{k:round(v['accuracy'],4) for k,v in groups.items()},flush=True)
    write(ROOT/'bench/preparation.json',{'modelSha256':sha(ROOT/'models/hundred/model.json'),'manifestSha256':sha(DATA/'prepared.json'),'lineSha256':sha(ROOT/'src/line.mjs'),'scope':'Frozen weights; new synthetic development bank, no final-test claims. Oracle uses renderer word boxes and the known rotation.','methods':report})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plans','render','evaluate']);args=p.parse_args()
    globals()[args.command]()
