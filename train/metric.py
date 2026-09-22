"""Equal-budget same-letter negatives; classification versus added style contrast."""
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from train.robustness import write, sha
from train.ten_model import Classifier, export
from train.ten import batch
from train.model import contrastive_loss
from train.retrieval import ROOT, load_data, score


def paired_pools(manifest,fonts):
    pools={};texts=set()
    for i,w in enumerate(manifest['windows']):
        s=manifest['samples'][w['source']]
        if s['role']!='train':continue
        if s['family'] not in fonts:raise ValueError('Unseen family in training')
        key=(s['family'],s['text'],s['renderer']);pools.setdefault(key,[]).append(i);texts.add(s['text'])
    if len(texts)<2 or any((f,t,r) not in pools for f in fonts for t in texts for r in ['pillow','chromium']):raise ValueError('Incomplete paired pool')
    return pools,sorted(texts)


def paired_batch(pools,texts,fonts,rng,families=32):
    labels=rng.choice(len(fonts),families,replace=False);a,b=rng.choice(texts,2,replace=False)
    ids=[]
    for label in labels:
        for text in [a,b]:ids.append(int(rng.choice(pools[fonts[label],text,'pillow' if rng.integers(2) else 'chromium'])))
    return ids,np.repeat(labels,2)


def train(method,steps=12000,seed=20260928):
    if method not in ['control','contrastive'] or type(steps) is not int or steps<1:raise ValueError('Invalid metric settings')
    torch.set_num_threads(4);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    data=load_data();artifact=data['artifact'];fonts=artifact['fonts'];windows=data['manifest']['windows']
    pools,texts=paired_pools(data['manifest'],fonts);rng=np.random.default_rng(seed)
    model=Classifier(len(fonts),context=True,dilations=artifact['dilations'])
    model.load_state_dict(torch.load(ROOT/'models/hundred/best.pt',weights_only=True,map_location='cpu')['state'])
    if export(model,fonts,artifact['preparation'])[0]!=artifact:raise ValueError('Changed initial encoder')
    model.to('mps');out=ROOT/f'.data/retrieval/{method}';out.mkdir(parents=True,exist_ok=True)
    captured=[]
    hook=model.head.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.000025)
    history=[];best=-1;start=time.perf_counter()
    def retain(step):
        nonlocal best
        result=score(model,data)['1'];known=result['known']['all']['top5Accuracy'];added=result['added']['all']['top5Accuracy']
        value=2*known*added/max(known+added,1e-12)
        history.append({'step':step,'known':result['known']['all'],'added':result['added']['all'],'selection':value})
        print(f'{method} {step}: top5 known {known:.2%}, added {added:.2%}',flush=True)
        if value>best:
            best=value;torch.save({'state':model.state_dict(),'step':step},out/'best.pt')
    retain(0)
    try:
        for step in range(1,steps+1):
            model.train();chosen,labels=paired_batch(pools,texts,fonts,rng);target=torch.tensor(labels,device='mps')
            captured.clear();optimizer.zero_grad(set_to_none=True)
            logits=model(*(t.to('mps') for t in batch(data['pixels'],windows,chosen)))
            loss=F.cross_entropy(logits,target,label_smoothing=.03)
            if method=='contrastive':loss=loss+.1*contrastive_loss(F.normalize(captured[0],dim=1),target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite metric loss')
            loss.backward();optimizer.step();scheduler.step();captured.clear()
            if step%1000==0:print(f'{method} {step}/{steps}: {time.perf_counter()-start:.0f}s',flush=True)
            if step%3000==0 or step==steps:retain(step)
    finally:hook.remove()
    ckpt=torch.load(out/'best.pt',weights_only=True,map_location='cpu');model.cpu().load_state_dict(ckpt['state'])
    packed,quantized=export(model,fonts,artifact['preparation']);result=score(quantized.to('mps'),data)
    write(out/'model.json',packed)
    write(ROOT/f'bench/metric-{method}.json',{'method':method,'steps':steps,'seed':seed,'selectedStep':ckpt['step'],'seconds':time.perf_counter()-start,
        'initialModelSha256':sha(ROOT/'models/hundred/model.json'),'modelSha256':sha(out/'model.json'),'trainingTensorSha256':data['manifest']['tensorSha256'],
        'developmentManifestSha256':sha(ROOT/'.data/preparation/prepared.json'),'trainerSha256':sha(ROOT/'train/metric.py'),
        'scope':'Synthetic development selection. Same-text negatives and different-text same-family positives in both arms. Harmonic mean of known/added top-5 selects one-prototype checkpoints. Added families are never optimized.',
        'results':result,'history':history})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=['control','contrastive']);p.add_argument('--steps',type=int,default=12000);args=p.parse_args();train(args.method,args.steps)
