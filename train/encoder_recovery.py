"""Choose a recovery from opened development data; reserve fresh phrases for confirmation."""
import numpy as np
import torch

from train.encoder import ROOT, embed, load_encoder, rank, metrics
from train.encoder_quality import OUT, BASE, CANDIDATE, vectors, reference_vectors, summary
from train.encoder_data import save
from train.robustness import read, sha


def choose(candidates,baseline):
    # Every reported sample must improve or tie; the two reported failures must
    # improve strictly. Prefer the smaller artifact within one percentage point.
    eligible=[c for c in candidates if c['selection']>baseline['selection'] and
              set(c['ranks'])==set(baseline['ranks']) and
              all(c['ranks'][f]<=r for f,r in baseline['ranks'].items()) and
              all(c['ranks'][f]<baseline['ranks'][f] for f in ['lora','montserrat'])]
    if not eligible:raise ValueError('No candidate improves the reported failures without another regression')
    best=max(c['selection'] for c in eligible)
    return min((c for c in eligible if c['selection']>=best-.01),key=lambda c:(c['bytes'],-c['selection'],c['name']))


def select():
    target=ROOT/'bench/encoder-recovery-selection.json'
    if target.exists():raise ValueError('Recovery selection already frozen')
    cases=read(OUT/'demo-before.json');split=read(ROOT/'bench/encoder-split.json')['families']
    samples=[];windows=[];arrays=[];offset=0
    for i,c in enumerate(cases):
        family=c['source']['known'].replace('-','')
        if split.get(family) not in ['train','development']:raise ValueError('Final or unknown family in recovery selection')
        samples.append(family)
        for w in c['inputs']:
            raw=np.round(np.array(w['pixels'])*255).astype(np.uint8)
            arrays.append(raw);windows.append({'source':i,'offset':offset,'width':w['width'],'height':w['height']});offset+=len(raw)
    pixels=np.concatenate(arrays);candidates=[];baseline=None
    paths=[('original',BASE),('refined',CANDIDATE),('case',ROOT/'.data/encoder/case-refine/encoder.json'),('large',ROOT/'.data/encoder/large-refine/encoder.json')]
    for name,path in paths:
        dev=vectors(path,'development');rs,rv,qs,qv=dev;families=sorted({s['family'] for s in rs})
        frs,frv,_,_=vectors(path,'final');all_families=sorted({s['family'] for s in frs})
        query=embed(load_encoder(path).to('mps'),{'windows':windows},pixels,list(range(len(cases))))
        for method in ['mean'] if name=='original' else ['scripts','script-words']:
            refs,owners,_,_=reference_vectors(path,'development',rs,rv,families,method)
            report=metrics(rank(qv,refs,owners,families),qs,families);a=report['groups']['split/train']['macroTop5'];b=report['groups']['split/development']['macroTop5']
            refs,owners,_,_=reference_vectors(path,'final',frs,frv,all_families,method)
            order=np.argsort(-rank(query,refs,owners,all_families),axis=1,kind='stable')
            ranks={f:int(np.flatnonzero(np.array(all_families)[row]==f)[0])+1 for f,row in zip(samples,order)}
            item={'name':name+'-'+method,'encoder':str(path.relative_to(ROOT)),'encoderSha256':sha(path),'bytes':path.stat().st_size,
                  'referenceMethod':method,'selection':2*a*b/(a+b),'ranks':ranks,'development':summary(report)}
            if name=='original':baseline=item
            else:candidates.append(item)
            print(item['name'],item['selection'],ranks,flush=True)
    winner=choose(candidates,baseline)
    save(target,{'winner':winner,'baseline':baseline,'candidates':candidates,'regressionsSha256':sha(OUT/'demo-before.json'),
        'criterion':'Improve full-development harmonic known/unseen macro top-5; no worse rank on any of eight reported train/development-family cases, strictly better Lora/Montserrat ranks. Smallest encoder within one absolute percentage point of best eligible development score.',
        'scope':'Reported examples are now explicitly development-selection data. No final-family query or third phrase result enters selection. Final catalog reference images supply distractor candidates. Selection does not establish top-one correctness on every reported example.'})
    print('Selected recovery',winner['name'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    select()
