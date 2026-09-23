"""Compare averaging Latin specimens with taking their maximum similarity."""
from collections import defaultdict
import numpy as np

from train.encoder import unit, rank, metrics
from train.encoder_quality import ROOT, OUT, BASE, CANDIDATE, vectors
from train.encoder_data import save
from train.robustness import read, sha


def means(rs,rv,ws,wv,families,method):
    if method not in ['mixed','words','balanced']:raise ValueError('Unknown reference average')
    groups=defaultdict(list);words=defaultdict(list)
    for i,s in enumerate(rs):groups[s['family'],s['script']].append(i)
    for i,s in enumerate(ws):words[s['family']].append(i)
    rows=[];owners=[]
    for owner,f in enumerate(families):
        scripts=sorted(s for family,s in groups if family==f)
        if not scripts:raise ValueError('Missing family references')
        for script in scripts:
            r=rv[groups[f,script]]
            if script=='Latn' and f in words:
                w=wv[words[f]]
                r=w if method=='words' else np.concatenate([r,w]) if method=='mixed' else np.array([r.mean(0),w.mean(0)])
            rows.append(r.mean(0));owners.append(owner)
    rows=unit(np.array(rows));scales=np.maximum(np.abs(rows).max(1)/127,1e-12);packed=np.round(rows/scales[:,None]).clip(-127,127).astype(np.int8)
    return unit(packed.astype(np.float32)*scales[:,None]),np.array(owners),packed,scales


if __name__=='__main__':
    reports=[];words=read(OUT/'word-references/development/manifest.json')['samples']
    for name,path in [('original',BASE),('refined',CANDIDATE)]:
        rs,rv,qs,qv=vectors(path,'development');wv=np.load(OUT/sha(path)[:16]/'development/words.npy');families=sorted({s['family'] for s in rs})
        for method in ['mixed','words','balanced']:
            refs,owners,_,_=means(rs,rv,words,wv,families,method)
            result=metrics(rank(qv,refs,owners,families),qs,families)
            a=result['groups']['split/train']['macroTop5'];b=result['groups']['split/development']['macroTop5']
            row={'encoder':name,'method':method,'selection':2*a*b/(a+b),'results':result};reports.append(row)
            print(name,method,row['selection'],result['groups']['split/development'],flush=True)
            save(OUT/'reference-means-development.json',reports)
