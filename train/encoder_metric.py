"""Fit an affine retrieval head from training-only within-family variation."""
import numpy as np
import torch

from train.encoder import ROOT, load_data, load_encoder, embed, selection_sources, rank, metrics
from train.encoder_data import save, validate_shard
from train.encoder_references import prototypes
from train.encoder_quality import BASE
from train.ten import batch
from train.ten_model import export
from train.robustness import read, sha

OUT=ROOT/'.data/encoder/metric-head'


def fit():
    OUT.mkdir(parents=True,exist_ok=True)
    data=ROOT/'.data/encoder/case-training';manifest=read(data/'manifest.json');pixels=np.memmap(data/'pixels.u8',dtype=np.uint8,mode='r')
    if sha(data/'pixels.u8')!=manifest['sha256']:raise ValueError('Changed metric training pixels')
    validate_shard(manifest,len(pixels));split=read(ROOT/'bench/encoder-split.json')
    if any(s['role']!='train' or split['families'][s['family']]!='train' for s in manifest['samples']):raise ValueError('Held-out data in metric fit')
    families=sorted({s['family'] for s in manifest['samples']});positions={f:i for i,f in enumerate(families)}
    model=load_encoder(BASE).to('mps');model.eval();sums=np.zeros((len(families),128));counts=np.zeros(len(families));cross=np.zeros((128,128));total=0
    with torch.no_grad():
        for start in range(0,len(manifest['windows']),128):
            ids=list(range(start,min(start+128,len(manifest['windows']))))
            values=model(*(t.to('mps') for t in batch(pixels,manifest['windows'],ids))).cpu().numpy().astype(np.float64)
            labels=[positions[manifest['samples'][manifest['windows'][i]['source']]['family']] for i in ids]
            np.add.at(sums,labels,values);np.add.at(counts,labels,1);cross+=values.T@values;total+=len(values)
    mean=sums.sum(0)/total;within=(cross-(sums/counts[:,None]).T@sums)/total
    eigen,basis=np.linalg.eigh((within+within.T)*.5);floor=max(eigen.mean()*.1,1e-8)
    save(OUT/'fit.json',{'baseEncoderSha256':sha(BASE),'trainingManifestSha256':sha(data/'manifest.json'),'families':len(families),'windows':total,'eigenFloor':float(floor),'scope':'Affine head fitted to training families only. No query vectors used.'})
    dev,dp,_=load_data();rids,qids=selection_sources(dev,quick=True);rs=[dev['samples'][i] for i in rids];qs=[dev['samples'][i] for i in qids];labels=sorted({s['family'] for s in rs});results=[]
    for power in [0,.25,.5]:
        model=load_encoder(BASE);transform=(basis*np.maximum(eigen,floor)**(-power))@basis.T
        with torch.no_grad():
            weight=transform@model.head.weight.numpy();bias=transform@(model.head.bias.numpy()-mean)
            model.head.weight.copy_(torch.from_numpy(weight).float());model.head.bias.copy_(torch.from_numpy(bias).float())
        artifact,restored=export(model,[str(i) for i in range(128)],read(BASE)['preparation'])
        del artifact['fonts'];artifact.update(kind='font-encoder',dimensions=128,normalization='l2')
        path=OUT/f'power-{power}.json';save(path,artifact)
        rv=embed(restored.to('mps'),dev,dp,rids);qv=embed(restored,dev,dp,qids);refs,owners,_,_=prototypes(rs,rv,labels,'scripts')
        result=metrics(rank(qv,refs,owners,labels),qs,labels)
        a=result['groups']['split/train']['macroTop5'];b=result['groups']['split/development']['macroTop5']
        results.append({'power':power,'encoderSha256':sha(path),'selection':2*a*b/(a+b),'results':result})
        print('Affine head',power,'known',a,'unseen',b,flush=True)
        save(OUT/'development.json',results)


if __name__=='__main__':
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    fit()
