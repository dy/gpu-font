"""Additional reference specimens with separate lower/upper-case Latin text."""
import argparse
from concurrent.futures import ProcessPoolExecutor

from train.encoder_data import ROOT, render_pillow, save, SPLIT
from train.robustness import read, sha

OUT=ROOT/'.data/detection-quality/word-references'
TEXTS=['hamburgefontsiv','packmyboxwithfivedozenliquorjugs']


def build(phase='development'):
    inventory=read(ROOT/'bench/corpus.json');split=read(SPLIT);target=OUT/phase
    pins={p:sha(ROOT/p) for p in ['train/encoder_word_refs.py','train/encoder_data.py','bench/corpus.json','bench/encoder-split.json','src/line.mjs','src/input.mjs','src/prepare.mjs']}
    jobs=[];families=[];plans=[]
    for family in inventory['families']:
        if family['excluded'] or (phase=='development' and split['families'][family['id']]=='test'):continue
        alphabet=set(family['alphabets'].get('Latn',''));own=[]
        for case in ['lower','upper']:
            words=[text if case=='lower' else text.upper() for text in TEXTS]
            if any(not set(text)<=alphabet for text in words):continue
            for text in words:
                for size in [32,56]:
                    own.append({'text':text,'size':size,'script':'Latn','referenceGroup':'Latn-words-'+case,'condition':'clean','light':False,'index':len(own),'role':'reference'})
        if not own:continue
        prefix=target/family['id'];jobs.append((family,own,prefix,pins));families.append(family['id']);plans.append(own)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(render_pillow,jobs),1):
            if i%250==0:print('Word references',phase,i,len(jobs),flush=True)
    samples=[];windows=[];target.mkdir(parents=True,exist_ok=True)
    with (target/'pixels.u8').open('wb') as output:
        for family,own in zip(families,plans):
            meta=read(target/f'{family}.json');path=target/f'{family}.u8'
            if meta['pins']!=pins or sha(path)!=meta['sha256'] or meta['samples']!=own:raise ValueError('Changed word reference shard')
            offset=output.tell();source=len(samples)
            windows.extend({**w,'source':w['source']+source,'offset':w['offset']+offset} for w in meta['windows'])
            samples.extend({**s,'family':family,'split':split['families'][family],'renderer':'pillow'} for s in own);output.write(path.read_bytes())
    save(target/'manifest.json',{'pins':pins,'sha256':sha(target/'pixels.u8'),'samples':samples,'windows':windows})
    print('Word references complete',phase,len(families),len(samples),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['development','final'],default='development');args=parser.parse_args();build(args.phase)
