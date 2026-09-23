"""Confirm case-diverse training after checkpoint selection; never optimize queries."""
import argparse
import numpy as np
import torch

from train.encoder import ROOT, load_encoder, embed, rank, metrics
from train.encoder_data import save, validate_shard
from train.encoder_quality import OUT, BASE, vectors, reference_vectors, catalog, summary
from train.robustness import read, sha

def confirm(large=False):
    name='large' if large else 'case';candidate=ROOT/f'.data/encoder/{name}-refine/encoder.json'
    selected=read(candidate.parent/'selection.json')
    if selected['encoderSha256']!=sha(candidate):raise ValueError('Changed selected encoder')
    rs,rv,qs,qv=vectors(candidate,'development');families=sorted({s['family'] for s in rs})
    refs,owners,_,_=reference_vectors(candidate,'development',rs,rv,families,'script-words')
    development=metrics(rank(qv,refs,owners,families),qs,families)
    save(OUT/f'{name}-development.json',development)
    print(name,'full development',development['groups']['split/development'],flush=True)
    bank='large' if large else 'next';phrase_path=OUT/f'phrases-{bank}.json';phrase=read(phrase_path);pixel_path=OUT/f'phrases-{bank}.u8'
    if sha(pixel_path)!=phrase['sha256']:raise ValueError('Changed confirmation pixels')
    for file,expected in phrase['pins'].items():
        if sha(ROOT/file)!=expected:raise ValueError('Changed confirmation dependency')
    validate_shard(phrase,pixel_path.stat().st_size)
    pixels=np.memmap(pixel_path,dtype=np.uint8,mode='r');cases=read(OUT/'demo-before.json');reports={}
    for stage,path,method in [('before',BASE,'mean'),('after',candidate,'script-words')]:
        rs,rv,qs,qv=vectors(path,'final');data,refs,owners,families=catalog(path,rs,rv,method)
        historical=metrics(rank(qv,refs,owners,families),qs,families)
        model=load_encoder(path).to('mps');pv=embed(model,phrase,pixels,list(range(len(phrase['samples']))))
        fresh=metrics(rank(pv,refs,owners,families),phrase['samples'],families);regressions=[]
        for case in cases:
            chunks=[];windows=[];offset=0
            for window in case['inputs']:
                raw=np.round(np.array(window['pixels'])*255).astype(np.uint8)
                windows.append({'source':0,'offset':offset,'width':window['width'],'height':window['height']});chunks.append(raw);offset+=len(raw)
            query=embed(model,{'windows':windows},np.concatenate(chunks),[0])
            scores=rank(query,refs,owners,families)[0];order=np.argsort(-scores,kind='stable');family=case['source']['known'].replace('-','')
            regressions.append({'family':family,'rank':int(np.flatnonzero(np.array(families)[order]==family)[0]+1),'matches':[{'family':families[i],'score':float(scores[i])} for i in order[:5]]})
        reports[stage]={'encoderSha256':sha(path),'referenceMethod':method,'historical':summary(historical),'phrases':summary(fresh),'regressions':regressions}
        save(OUT/f'{name}-{stage}-phrases.json',fresh)
        print(stage,'new phrases',fresh['groups']['all'],'regressions',[(c['family'],c['rank']) for c in regressions],flush=True)
        if stage=='after':
            save(OUT/f'{name}-google-fonts.json',data)
            reports[stage].update(catalogSha256=sha(OUT/f'{name}-google-fonts.json'),catalogBytes=(OUT/f'{name}-google-fonts.json').stat().st_size,encoderBytes=path.stat().st_size)
    save(ROOT/f'bench/encoder-{name}-quality.json',{'checkpointSelection':selected,'development':summary(development),'phraseManifestSha256':sha(phrase_path),'reports':reports,
         'scope':('Third phrase bank frozen before capacity experiment results. ' if large else 'Second phrase bank frozen before case-training results. ')+'First phrase bank restricted to train/development families is now checkpoint-selection data. No final-family pixels enter optimization or selection. Historical final results and reported demo cases are regressions; synthetic evaluation is not real-image certainty.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--large',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    confirm(args.large)
