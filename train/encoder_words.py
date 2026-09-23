"""Case-diverse Latin training views; held-out families and query text stay excluded."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import random

from train.encoder_data import ROOT, render_pillow, save, SPLIT, validate_shard
from train.robustness import read, sha

OUT=ROOT/'.data/encoder/case-training'


def plan():
    inventory=read(ROOT/'bench/corpus.json');split=read(SPLIT)
    development=read(ROOT/'.data/encoder/development.json')
    reserved={s['text'].casefold() for s in development['samples']}
    # Every previously opened query bank stays excluded from new optimization.
    reserved.update(s['text'].casefold() for s in read(ROOT/'.data/encoder/final.json')['samples'])
    reserved.update(s['text'].casefold() for s in read(ROOT/'.data/detection-quality/phrases.json')['samples'])
    reserved.update(['quiet rivers flow','hamburgefontsiv','packmyboxwithfivedozenliquorjugs'])
    rng=random.Random(20260923);texts=[]
    while len(texts)<32:
        i=len(texts);n=[2,4,8,12][i%4]
        text=''.join(rng.choices('aaaaaaaabcddddeeeeeeeeeeeeffgghhhhhhiiiiiiijkllllmmnnnnnnnooooooooppqrrrrrrsssssstttttttttuuvwxyyz',k=n))
        if text.casefold() in reserved:continue
        reserved.add(text.casefold())
        texts.append(text if i<16 else text.capitalize() if i<24 else text.upper())
    jobs=[]
    for f in inventory['families']:
        if f['excluded'] or split['families'][f['id']]!='train':continue
        coverage=set(f['alphabets'].get('Latn',''))
        if any(not set(t)<=coverage for t in texts):continue
        plans=[{'text':text,'size':[24,32,44,56][i%4],'index':20260923+i,'light':bool(i%2),'script':'Latn','length':'2-3' if len(text)<4 else '4-7' if len(text)<8 else '8+',
                'condition':'clean' if i%2 else 'augmented','role':'train'} for i,text in enumerate(texts)]
        jobs.append({'family':f,'plans':plans})
    pins={p:sha(ROOT/p) for p in ['train/encoder_words.py','scripts/encoder-words-browser.mjs','train/encoder_data.py','train/hundred_data.py','bench/encoder-split.json','bench/corpus.json','src/line.mjs','src/input.mjs','src/prepare.mjs']}
    target=OUT/'plan.json';data={'pins':pins,'texts':texts,'jobs':jobs}
    if target.exists() and read(target)!=data:raise ValueError('Changed case-training plan')
    save(target,data);print('Case training plan',len(jobs),'families',len(texts),'strings',flush=True)


def render():
    data=read(OUT/'plan.json')
    jobs=[(j['family'],j['plans'],OUT/'pillow'/j['family']['id'],data['pins']) for j in data['jobs']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(render_pillow,jobs),1):
            if i%250==0:print('Case training Pillow',i,len(jobs),flush=True)


def pack():
    plan=read(OUT/'plan.json');samples=[];windows=[]
    for file,expected in plan['pins'].items():
        if sha(ROOT/file)!=expected:raise ValueError('Changed case-training dependency')
    with (OUT/'pixels.u8').open('wb') as output:
        for job in plan['jobs']:
            for renderer in ['pillow','chromium']:
                prefix=OUT/renderer/job['family']['id'];meta=read(prefix.with_suffix('.json'));raw=prefix.with_suffix('.u8').read_bytes()
                if meta['pins']!=plan['pins'] or meta['samples']!=job['plans'] or sha(prefix.with_suffix('.u8'))!=meta['sha256']:raise ValueError('Changed case-training shard')
                validate_shard(meta,len(raw));offset=output.tell();owner=len(samples)
                windows.extend({**w,'source':w['source']+owner,'offset':w['offset']+offset} for w in meta['windows'])
                samples.extend({**s,'family':job['family']['id'],'split':'train','renderer':renderer} for s in meta['samples']);output.write(raw)
    save(OUT/'manifest.json',{'pins':plan['pins'],'planSha256':sha(OUT/'plan.json'),'sha256':sha(OUT/'pixels.u8'),'samples':samples,'windows':windows})
    print('Packed case training',len(samples),'sources',len(windows),'windows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['plan','render','pack']);args=parser.parse_args();globals()[args.command]()
