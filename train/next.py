"""Matched continuation: same initial checkpoint, budget and seed; fresh development queries."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train.robustness import read, write, sha
from train.ten import batch
from train.ten_model import Classifier, export
from train.hundred import calibrate, metrics, probabilities
from train.hundred_data import band

ROOT=Path(__file__).resolve().parents[1]

def predict(model,pixels,windows,samples,role=None):
    indexes=[i for i,w in enumerate(windows) if role is None or samples[w['source']]['role']==role]
    if not indexes:raise ValueError('No prediction windows')
    sources=sorted({windows[i]['source'] for i in indexes});mapping={s:i for i,s in enumerate(sources)}
    outputs=[];owners=[];device=next(model.parameters()).device;model.eval()
    with torch.no_grad():
        for start in range(0,len(indexes),128):
            chosen=indexes[start:start+128]
            outputs.append(model(*(t.to(device) for t in batch(pixels,windows,chosen))).cpu().numpy());owners.extend(mapping[windows[i]['source']] for i in chosen)
    return [samples[s] for s in sources],np.concatenate(outputs),np.array(owners)


def train(method,steps=20000,seed=20260925):
    if method not in ['current','deskew'] or type(steps) is not int or steps<1:raise ValueError('Invalid comparison settings')
    torch.set_num_threads(4);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    root=ROOT/'.data/next';out=root/f'{method}-{seed}';out.mkdir(parents=True,exist_ok=True)
    artifact=read(ROOT/'models/hundred/model.json');fonts=artifact['fonts']
    if method=='current':
        manifest=read(ROOT/'.data/hundred/prepared.json');path=ROOT/'.data/hundred/prepared.u8'
    else:
        manifest=read(root/'deskew.json');path=root/'deskew.u8'
        if manifest['lineSha256']!=sha(ROOT/'src/line.mjs'):raise ValueError('Changed line preparation')
    if sha(path)!=manifest['tensorSha256']:raise ValueError('Changed training pixels')
    pixels=np.memmap(path,dtype=np.uint8,mode='r');samples=manifest['samples'];windows=manifest['windows']
    dev=read(ROOT/'.data/preparation/prepared.json');name='current' if method=='current' else 'deskew-windows'
    for key,p in [('inputSha256','src/input.mjs'),('lineSha256','src/line.mjs'),('normalizerSha256','src/prepare.mjs')]:
        if dev[key]!=sha(ROOT/p):raise ValueError('Changed development preparation')
    dev_path=ROOT/f'.data/preparation/{name}.u8'
    if sha(dev_path)!=dev['methods'][name]['sha256']:raise ValueError('Changed development pixels')
    dev_pixels=np.memmap(dev_path,dtype=np.uint8,mode='r');dev_windows=dev['methods'][name]['windows']
    dev_samples=[{**s,'length':band(s['text']),'role':'validation'} for s in dev['samples']]
    used={s['text'].lower() for s in samples if s['role']=='train' and len(s['text'])>1}
    if any(s['text'].lower() in used for s in dev_samples if len(s['text'])>1):raise ValueError('Development leakage')
    model=Classifier(len(fonts),dilations=artifact['dilations'],context=True)
    model.load_state_dict(torch.load(ROOT/'models/hundred/best.pt',weights_only=True,map_location='cpu')['state'])
    if export(model,fonts,artifact['preparation'])[0]!=artifact:raise ValueError('Initialization mismatch')
    model.to('mps');rng=np.random.default_rng(seed)
    pools={(family,renderer):[] for family in fonts for renderer in ['pillow','chromium']}
    for i,w in enumerate(windows):
        s=samples[w['source']]
        if s['role']=='train':pools[s['family'],s['renderer']].append(i)
    if any(not p for p in pools.values()):raise ValueError('Empty training pool')
    val_manifest={'dataConfig':{'fonts':fonts},'samples':dev_samples}
    def evaluate(net):
        _,logits,owners=predict(net,dev_pixels,dev_windows,dev_samples)
        return metrics(val_manifest,list(range(len(dev_samples))),probabilities(logits,owners,len(dev_samples)))
    history=[];best=-1;start=time.perf_counter()
    def retain(step):
        nonlocal best
        report=evaluate(model);accuracy=report['groups']['all']['accuracy'];history.append({'step':step,'groups':report['groups']})
        print(f'{method} {step}: dev {accuracy:.2%}; rotate {report["groups"]["condition/rotate"]["accuracy"]:.2%}',flush=True)
        if accuracy>best:
            best=accuracy;torch.save({'state':model.state_dict(),'step':step},out/'best.pt')
    retain(0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.000025)
    for step in range(1,steps+1):
        model.train();labels=rng.integers(len(fonts),size=64)
        chosen=[int(rng.choice(pools[fonts[label],'pillow' if rng.integers(2) else 'chromium'])) for label in labels]
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(*(t.to('mps') for t in batch(pixels,windows,chosen))),torch.tensor(labels,device='mps'),label_smoothing=.03)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();optimizer.step();scheduler.step()
        if step%1000==0:print(f'{method} {step}/{steps}: {time.perf_counter()-start:.0f}s',flush=True)
        if step%5000==0 or step==steps:retain(step)
    ckpt=torch.load(out/'best.pt',weights_only=True,map_location='cpu');model.cpu().load_state_dict(ckpt['state'])
    preparation={**artifact['preparation'],**({'method':'deskew-windows','lineSha256':sha(ROOT/'src/line.mjs')} if method=='deskew' else {})}
    packed,quantized=export(model,fonts,preparation);quantized.to('mps')
    known,logits,owners=predict(quantized,dev_pixels,dev_windows,dev_samples)
    unknown,other,other_owners=predict(quantized,pixels,windows,samples,'unknown-validation')
    calibration_manifest={'dataConfig':{'fonts':fonts},'samples':known+unknown}
    combined=np.concatenate([logits,other]);all_owners=np.concatenate([owners,other_owners+len(known)])
    calibration=calibrate(calibration_manifest,list(range(len(known)+len(unknown))),combined,all_owners)
    result=evaluate(quantized)
    if result['groups']['all']['accuracy']<best-.01:raise ValueError('Int8 loss exceeds one point')
    write(out/'model.json',packed);write(out/'calibration.json',calibration)
    write(ROOT/f'bench/next-{method}-{seed}.json',{'method':method,'seed':seed,'steps':steps,'selectedStep':ckpt['step'],'seconds':time.perf_counter()-start,'initialModelSha256':sha(ROOT/'models/hundred/model.json'),'modelSha256':sha(out/'model.json'),'trainingTensorSha256':manifest['tensorSha256'],'developmentManifestSha256':sha(ROOT/'.data/preparation/prepared.json'),'trainerSha256':sha(Path(__file__)),'scope':'Development selection only. No final-test or independent screenshot claims.','validationInt8':result,'calibration':calibration,'history':history})
    print(f'Completed {method}: {result["groups"]["all"]["accuracy"]:.2%}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=['current','deskew']);p.add_argument('--steps',type=int,default=20000);p.add_argument('--seed',type=int,default=20260925);args=p.parse_args();train(args.method,args.steps,args.seed)
