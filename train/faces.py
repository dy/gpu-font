"""Joint family/weight/style pilot; synthetic validation selects, test stays separate."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import parametrize
from train.robustness import read, write, sha
from train.ten import batch
from train.ten_model import Classifier, export, load_export
from train.hundred import probabilities
from train.next import predict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'.data/faces'


def load_data():
    manifest = read(DATA/'prepared.json'); faces = read(ROOT/'bench/font-faces.json')['faces']
    for key,path in [('tensorSha256',DATA/'prepared.u8'),('facesSha256',ROOT/'bench/font-faces.json'),('textsSha256',ROOT/'bench/face-texts.json'),
                     ('inputSha256',ROOT/'src/input.mjs'),('normalizerSha256',ROOT/'src/prepare.mjs'),
                     ('generatorSha256',ROOT/'train/faces_data.py'),('runnerSha256',ROOT/'scripts/faces.mjs')]:
        if manifest[key] != sha(path): raise ValueError(f'Changed face data: {path}')
    face_map = {f['id']:f for f in faces}; samples = manifest['samples']; pools = {r:set() for r in ['train','validation','test']}
    for sample in samples:
        face = face_map.get(sample['family'])
        if not face or sample['role'] not in pools or any(sample[k] != face[k] for k in ['weight','style']) or sample['fontFamily'] != face['family']: raise ValueError('Invalid face label')
        if len(sample['text'])>1: pools[sample['role']].add(sample['text'].lower())
    if any(not pool for pool in pools.values()): raise ValueError('Missing face text split')
    if any(a & b for i,a in enumerate(pools.values()) for b in list(pools.values())[i+1:]): raise ValueError('Face text leakage')
    offset = 0; coverage = set()
    for w in manifest['windows']:
        if not isinstance(w,dict) or any(type(w.get(k)) is not int for k in ['source','offset','width','height']) or w['offset'] != offset or not 0 <= w['source'] < len(samples) or not 1 <= w['width'] <= 128 or not 1 <= w['height'] <= 48: raise ValueError('Invalid face window')
        offset += w['width']*w['height']; coverage.add(w['source'])
    if len(coverage) != len(samples) or (DATA/'prepared.u8').stat().st_size != offset: raise ValueError('Incomplete face data')
    return faces,manifest,np.memmap(DATA/'prepared.u8',mode='r',dtype=np.uint8)


def metrics(faces,samples,probs):
    if not faces or not samples or probs.shape != (len(samples),len(faces)) or not np.isfinite(probs).all() or (probs<0).any() or not np.allclose(probs.sum(1),1): raise ValueError('Invalid face probabilities')
    ids=[f['id'] for f in faces]; families=list(dict.fromkeys(f['family'] for f in faces))
    attributes={k:list(dict.fromkeys(f[k] for f in faces)) for k in ['family','weight','style']}
    scores={k:np.stack([probs[:,[f[k]==value for f in faces]].sum(1) for value in values],axis=1) for k,values in attributes.items()}
    groups={}; confusion=np.zeros((len(families),len(families)),dtype=int)
    for i,sample in enumerate(samples):
        label=ids.index(sample['family']); face=faces[label]
        hits={'face':int(probs[i].argmax()==label)}
        for k,values in attributes.items(): hits[k]=int(values[int(scores[k][i].argmax())]==face[k])
        hits['familyTop3']=int(attributes['family'].index(face['family']) in scores['family'][i].argsort()[-3:])
        # Diagnostic oracle: how well are attributes identified when family is supplied?
        same=[j for j,f in enumerate(faces) if f['family']==face['family']]
        for k in ['weight','style']:
            values=attributes[k]; sums=[sum(probs[i,j] for j in same if faces[j][k]==v) for v in values]
            hits['knownFamily'+k.title()]=int(values[int(np.argmax(sums))]==face[k])
        confusion[families.index(face['family']),scores['family'][i].argmax()]+=1
        for key in ['all','length/'+sample['length'],'condition/'+sample['condition'],'font/'+face['family'],'renderer/'+sample['renderer']]:
            g=groups.setdefault(key,{'count':0,**{k:0 for k in hits}});g['count']+=1
            for k,v in hits.items():g[k]+=v
    for g in groups.values():
        for k in list(g):
            if k!='count':g[k+'Accuracy']=g[k]/g['count']
    return {'groups':groups,'familyOrder':families,'familyConfusion':confusion.tolist()}


def evaluate(model,faces,manifest,pixels,role):
    samples,logits,owners=predict(model,pixels,manifest['windows'],manifest['samples'],role)
    return metrics(faces,samples,probabilities(logits,owners,len(samples)))


def train(steps=15000,seed=20260926):
    if type(steps) is not int or steps<1:raise ValueError('Training steps must be positive')
    torch.set_num_threads(4);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    faces,manifest,pixels=load_data(); baseline=read(ROOT/'models/hundred/model.json')
    original=Classifier(100,context=True,dilations=baseline['dilations'])
    original.load_state_dict(torch.load(ROOT/'models/hundred/best.pt',weights_only=True,map_location='cpu')['state'])
    if export(original,baseline['fonts'],baseline['preparation'])[0]!=baseline:raise ValueError('Changed initial model')
    model=Classifier(len(faces),context=True,dilations=baseline['dilations'])
    model.load_state_dict({k:v for k,v in original.state_dict().items() if not k.startswith('head.')},strict=False)
    with torch.no_grad():
        for i,face in enumerate(faces):
            j=baseline['fonts'].index(face['family']);model.head.weight[i].copy_(original.head.weight[j]);model.head.bias[i].copy_(original.head.bias[j]-math.log(4))
    model.to('mps');rng=np.random.default_rng(seed);windows=manifest['windows'];samples=manifest['samples']
    pools={(f['id'],r):[] for f in faces for r in ['pillow','chromium']}
    for i,w in enumerate(windows):
        s=samples[w['source']]
        if s['role']=='train':pools[s['family'],s['renderer']].append(i)
    if any(not p for p in pools.values()):raise ValueError('Missing face training pool')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.000025)
    history=[];best=-1;start=time.perf_counter()
    def retain(step):
        nonlocal best
        report=evaluate(model,faces,manifest,pixels,'validation');g=report['groups']['all'];history.append({'step':step,'groups':report['groups']})
        print(f'Faces {step}: family {g["familyAccuracy"]:.2%}, weight {g["weightAccuracy"]:.2%}, italic {g["styleAccuracy"]:.2%}, joint {g["faceAccuracy"]:.2%}',flush=True)
        if g['faceAccuracy']>best:
            best=g['faceAccuracy'];torch.save({'state':model.state_dict(),'step':step},DATA/'best.pt')
    retain(0)
    for step in range(1,steps+1):
        model.train();labels=rng.integers(len(faces),size=64)
        chosen=[int(rng.choice(pools[faces[label]['id'],'pillow' if rng.integers(2) else 'chromium'])) for label in labels]
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(*(t.to('mps') for t in batch(pixels,windows,chosen))),torch.tensor(labels,device='mps'),label_smoothing=.03)
        if not torch.isfinite(loss):raise ValueError('Nonfinite face loss')
        loss.backward();optimizer.step();scheduler.step()
        if step%1000==0:print(f'Faces {step}/{steps}: {time.perf_counter()-start:.0f}s',flush=True)
        if step%3000==0 or step==steps:retain(step)
    finish(steps,seed,time.perf_counter()-start,history)


class Int8Weights(torch.nn.Module):
    def forward(self,weight):
        row=weight.flatten(1);scale=(row.detach().abs().amax(1)/127).clamp(min=1e-12)
        scale=scale.reshape((-1,)+ (1,)*(weight.ndim-1))
        quantized=(weight/scale).round().clamp(-127,127)*scale
        return weight+(quantized-weight).detach()


def qat_snapshot(model):
    # Removing a parametrization from a deepcopy can mutate the live dynamic class.
    plain=Classifier(model.head.out_features,training=False,context=model.context,dilations=model.dilations)
    state={k.replace('.parametrizations.weight.original','.weight'):v.detach().cpu().clone() for k,v in model.state_dict().items()}
    plain.load_state_dict(state)
    return plain.eval()


def finish(steps=15000,seed=20260926,seconds=None,history=None):
    """Resume export from the selected float checkpoint, without repeating training."""
    torch.set_num_threads(4);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    faces,manifest,pixels=load_data();baseline=read(ROOT/'models/hundred/model.json');ids=[f['id'] for f in faces]
    ckpt=torch.load(DATA/'best.pt',weights_only=True,map_location='cpu')
    model=Classifier(len(faces),context=True,dilations=baseline['dilations']);model.load_state_dict(ckpt['state'])
    float_report=evaluate(model.to('mps'),faces,manifest,pixels,'validation');model.cpu()
    artifact,quantized=export(model,ids,baseline['preparation'])
    report=evaluate(quantized.to('mps'),faces,manifest,pixels,'validation');before=report;qat_history=[];qat_seconds=0
    print(f'Float joint {float_report["groups"]["all"]["faceAccuracy"]:.2%}; int8 joint {report["groups"]["all"]["faceAccuracy"]:.2%}',flush=True)
    if report['groups']['all']['faceAccuracy']<float_report['groups']['all']['faceAccuracy']-.01:
        qat=model.folded().to('mps')
        for layer in [*qat.convs,qat.head]:parametrize.register_parametrization(layer,'weight',Int8Weights())
        pools={f['id']:[] for f in faces};windows=manifest['windows'];samples=manifest['samples'];rng=np.random.default_rng(seed+1)
        for i,w in enumerate(windows):
            if samples[w['source']]['role']=='train':pools[samples[w['source']]['family']].append(i)
        if any(not pool for pool in pools.values()):raise ValueError('Missing QAT training pool')
        optimizer=torch.optim.AdamW(qat.parameters(),lr=.00002,weight_decay=1e-4);start=time.perf_counter()
        for step in range(1,2001):
            labels=rng.integers(len(faces),size=64);chosen=[int(rng.choice(pools[ids[label]])) for label in labels]
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(qat(*(t.to('mps') for t in batch(pixels,windows,chosen))),torch.tensor(labels,device='mps'),label_smoothing=.03)
            if not torch.isfinite(loss):raise ValueError('Nonfinite QAT loss')
            loss.backward();optimizer.step()
            if step%500==0:
                plain=qat_snapshot(qat)
                candidate,net=export(plain,ids,baseline['preparation']);result=evaluate(net.to('mps'),faces,manifest,pixels,'validation')
                qat_history.append({'step':step,'groups':result['groups']});accuracy=result['groups']['all']['faceAccuracy']
                print(f'QAT {step}: joint {accuracy:.2%}',flush=True)
                if accuracy>report['groups']['all']['faceAccuracy']:
                    artifact,quantized,report=candidate,net,result
                    torch.save({'state':plain.state_dict(),'step':step,'folded':True},DATA/'qat.pt')
        qat_seconds=time.perf_counter()-start
    write(DATA/'model.json',artifact)
    write(ROOT/'bench/faces-training.json',{'steps':steps,'seed':seed,'selectedStep':ckpt['step'],'seconds':seconds,'qatSeconds':qat_seconds,
        'parameters':sum(p.numel() for p in quantized.parameters()),'modelBytes':(DATA/'model.json').stat().st_size,'modelSha256':sha(DATA/'model.json'),
        'initialModelSha256':sha(ROOT/'models/hundred/model.json'),'dataSha256':sha(DATA/'prepared.json'),'trainerSha256':sha(Path(__file__)),
        'scope':'Ten known families × genuine 400/700 × normal/italic. Closed-set synthetic pilot, no unknown-family rejection or continuous-weight accuracy claim.',
        'validationFloat':float_report,'validationBeforeQAT':before,'validationInt8':report,'history':history,'qatHistory':qat_history,
        'quantizationGatePassed':report['groups']['all']['faceAccuracy']>=float_report['groups']['all']['faceAccuracy']-.01})
    if report['groups']['all']['faceAccuracy']<float_report['groups']['all']['faceAccuracy']-.01:raise ValueError('Face quantization loss still exceeds one point')


def final_evaluation():
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    faces,manifest,pixels=load_data();training=read(ROOT/'bench/faces-training.json')
    if training['modelSha256']!=sha(DATA/'model.json') or training['dataSha256']!=sha(DATA/'prepared.json'):raise ValueError('Changed selected face model/data')
    report=evaluate(load_export(read(DATA/'model.json')).to('mps'),faces,manifest,pixels,'test')
    write(ROOT/'bench/faces-test.json',{'modelSha256':training['modelSha256'],'dataSha256':training['dataSha256'],'scope':'Held-out synthetic texts/renderings. Both renderers also occur in training; no independent screenshot claim.','test':report})
    print(report['groups']['all'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','finish','evaluate']);p.add_argument('--steps',type=int,default=15000);args=p.parse_args()
    if args.command=='train':train(args.steps)
    elif args.command=='finish':finish(args.steps)
    else:final_evaluation()
