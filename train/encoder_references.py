"""Compare reference aggregation with frozen weights and development queries only."""
import argparse
import base64
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from train.encoder import load_data, load_encoder, embed, selection_sources, unit, rank, metrics
from train.encoder_data import ROOT, save
from train.robustness import read, sha

OUT=ROOT/'.data/detection-quality'
ENCODER=ROOT/'models/encoder/encoder.json'


def cluster(vectors,count):
    vectors=unit(vectors);count=min(count,len(vectors))
    center=unit(vectors.mean(0,keepdims=True))[0]
    chosen=[int(np.argmax(vectors@center))]
    for _ in range(1,count):
        distance=(vectors@vectors[chosen].T).max(1);distance[chosen]=np.inf
        chosen.append(int(np.argmin(distance)))
    centers=vectors[chosen].copy()
    for _ in range(20):
        labels=(vectors@centers.T).argmax(1)
        updated=np.array([unit(vectors[labels==i].mean(0,keepdims=True))[0] if np.any(labels==i) else centers[i] for i in range(count)])
        if np.allclose(updated,centers,atol=1e-6):break
        centers=updated
    return centers


def prototypes(samples,vectors,families,method):
    pools=defaultdict(list)
    for i,s in enumerate(samples):pools[s['family']].append(i)
    refs=[];owners=[]
    for owner,f in enumerate(families):
        ids=pools[f]
        if not ids:raise ValueError('Missing references: '+f)
        if method=='mean':rows=unit(vectors[ids].mean(0,keepdims=True))
        elif method=='scripts':
            groups=defaultdict(list)
            for i in ids:groups[samples[i]['script']].append(i)
            rows=np.concatenate([unit(vectors[group].mean(0,keepdims=True)) for _,group in sorted(groups.items())])
        elif method in ['cluster4','cluster8']:rows=cluster(vectors[ids],int(method[-1]))
        else:raise ValueError('Unknown reference method')
        refs.extend(rows);owners.extend([owner]*len(rows))
    refs=unit(np.array(refs));scales=np.maximum(np.abs(refs).max(1,keepdims=True)/127,1e-12)
    packed=np.round(refs/scales).clip(-127,127).astype(np.int8)
    return unit(packed.astype(np.float32)*scales),np.array(owners),packed,scales[:,0]


def cache(phase):
    target=OUT/phase;target.mkdir(parents=True,exist_ok=True)
    manifest,pixels,_=load_data(phase);binding={'encoderSha256':sha(ENCODER),'dataSha256':sha(ROOT/f'.data/encoder/{phase}.json')}
    if (target/'cache.json').exists():
        if read(target/'cache.json')!=binding:raise ValueError('Stale embedding cache')
        return target
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    torch.set_num_threads(4);model=load_encoder(ENCODER).to('mps')
    for role,ids in zip(['reference','query'],selection_sources(manifest)):
        vectors=[];start=time.perf_counter()
        for offset in range(0,len(ids),20000):
            vectors.append(embed(model,manifest,pixels,ids[offset:offset+20000]))
            print(phase,role,min(offset+20000,len(ids)),len(ids),round(time.perf_counter()-start),'s',flush=True)
        np.save(target/f'{role}.npy',np.concatenate(vectors),allow_pickle=False)
        save(target/f'{role}.json',[manifest['samples'][i] for i in ids])
    save(target/'cache.json',binding)
    return target


def compare():
    target=cache('development');rs=read(target/'reference.json');qs=read(target/'query.json')
    rv=np.load(target/'reference.npy');qv=np.load(target/'query.npy');families=sorted({s['family'] for s in rs});reports={}
    for method in ['mean','scripts','cluster4','cluster8']:
        refs,owners,packed,scales=prototypes(rs,rv,families,method)
        result=metrics(rank(qv,refs,owners,families),qs,families);reports[method]={'references':len(refs),'int8Bytes':packed.nbytes+scales.nbytes,'results':result}
        print(method,{k:result['groups'][k] for k in ['all','split/train','split/development']},flush=True)
        save(OUT/'reference-comparison.json',{'binding':read(target/'cache.json'),'methods':reports,'scope':'Frozen weights, development queries only. No final query selection.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['compare','cache']);p.add_argument('--phase',choices=['development','final'],default='development');a=p.parse_args()
    if a.command=='compare':compare()
    else:cache(a.phase)
