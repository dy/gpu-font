"""Frozen learned features: add seven development families without retraining."""
import time
from pathlib import Path
import numpy as np
import torch
from train.robustness import read, write, sha
from train.ten import batch
from train.ten_model import load_export
from train.hundred_data import band

ROOT=Path(__file__).resolve().parents[1]


def unit(values):
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Invalid feature matrix')
    return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)


def head_projection(weight):
    weight=np.asarray(weight,dtype=np.float32)
    if weight.ndim!=2 or not weight.size or not np.isfinite(weight).all():raise ValueError('Invalid learned head')
    _,singular,right=np.linalg.svd(weight,full_matrices=False)
    return right.T*singular


def features(model,pixels,windows,sources,projection=None):
    if not sources or len(set(sources))!=len(sources):raise ValueError('Expected unique nonempty sources')
    mapping={s:i for i,s in enumerate(sources)};chosen=[i for i,w in enumerate(windows) if w['source'] in mapping]
    sums=np.zeros((len(sources),64 if projection is None else projection.shape[1]),dtype=np.float32);counts=np.zeros(len(sources),dtype=int)
    device=next(model.parameters()).device;model.eval()
    with torch.no_grad():
        for start in range(0,len(chosen),128):
            ids=chosen[start:start+128];values=model(*(t.to(device) for t in batch(pixels,windows,ids))).cpu().numpy()
            values=unit(values if projection is None else values@projection)
            owners=[mapping[windows[i]['source']] for i in ids];np.add.at(sums,owners,values);np.add.at(counts,owners,1)
    if (counts==0).any():raise ValueError('Missing source features')
    return unit(sums/counts[:,None])


def prototypes(vectors,samples,families,by_length=False):
    rows=[];owners=[]
    for family in families:
        for length in ['1','2-3','4-7','8+'] if by_length else [None]:
            indexes=[i for i,s in enumerate(samples) if s['family']==family and (length is None or s['length']==length)]
            if not indexes:raise ValueError('Missing prototype pool')
            rows.append(unit(vectors[indexes].mean(0,keepdims=True))[0]);owners.append(families.index(family))
    rows=np.array(rows);scale=np.maximum(np.abs(rows).max(1,keepdims=True)/127,1e-12)
    quantized=np.round(rows/scale).clip(-127,127).astype(np.int8)
    return unit(quantized*scale),np.array(owners),quantized.nbytes+scale.nbytes


def metrics(vectors,samples,families,refs,owners):
    scores=np.full((len(vectors),len(families)),-np.inf)
    similarities=vectors@refs.T
    for i,owner in enumerate(owners):scores[:,owner]=np.maximum(scores[:,owner],similarities[:,i])
    ranking=np.argsort(-scores,axis=1,kind='stable');groups={}
    for row,sample in zip(ranking,samples):
        label=families.index(sample['family']);hit=int(row[0]==label);top5=int(label in row[:5])
        for key in ['all','length/'+sample['length'],'condition/'+sample['condition'],'font/'+sample['family']]:
            g=groups.setdefault(key,{'count':0,'correct':0,'top5':0});g['count']+=1;g['correct']+=hit;g['top5']+=top5
    for g in groups.values():g['accuracy']=g['correct']/g['count'];g['top5Accuracy']=g['top5']/g['count']
    return groups


def load_data():
    artifact=read(ROOT/'models/hundred/model.json')
    manifest=read(ROOT/'.data/hundred/prepared.json');tensor=ROOT/'.data/hundred/prepared.u8'
    if manifest['tensorSha256']!=sha(tensor):raise ValueError('Changed prototype pixels')
    for key,name in [('inputSha256','src/input.mjs'),('preparationSha256','src/prepare.mjs')]:
        if manifest[key]!=sha(ROOT/name):raise ValueError('Changed prototype preparation')
    samples=manifest['samples'];windows=manifest['windows'];pixels=np.memmap(tensor,mode='r',dtype=np.uint8)
    families=artifact['fonts']+sorted({s['family'] for s in samples if s['role']=='unknown-validation'})
    pids=[i for i,s in enumerate(samples) if s['role'] in ['validation','unknown-validation'] and s['condition']=='clean' and s['index']%2==0]
    uids=[i for i,s in enumerate(samples) if s['role']=='unknown-validation' and s['index']%2==1]
    prototype_samples=[samples[i] for i in pids];unknown=[samples[i] for i in uids]
    if {s['text'].lower() for s in prototype_samples}&{s['text'].lower() for s in unknown}:raise ValueError('Prototype/query leakage')
    dev=read(ROOT/'.data/preparation/prepared.json');dt=ROOT/'.data/preparation/current.u8'
    if dev['methods']['current']['sha256']!=sha(dt) or dev['inputSha256']!=sha(ROOT/'src/input.mjs') or dev['normalizerSha256']!=sha(ROOT/'src/prepare.mjs'):raise ValueError('Changed development data')
    known=[{**s,'length':band(s['text'])} for s in dev['samples']]
    if {s['text'].lower() for s in prototype_samples if len(s['text'])>1}&{s['text'].lower() for s in known if len(s['text'])>1}:raise ValueError('Known prototype/query leakage')
    return {'artifact':artifact,'manifest':manifest,'pixels':pixels,'families':families,'prototypeIds':pids,'prototypeSamples':prototype_samples,
            'unknownIds':uids,'unknown':unknown,'known':known,'knownPixels':np.memmap(dt,mode='r',dtype=np.uint8),'knownWindows':dev['methods']['current']['windows']}


def score(model,data,project=False):
    projection=head_projection(model.head.weight.detach().cpu().numpy()) if project else None
    head=model.head;model.head=torch.nn.Identity()
    try:
        pv=features(model,data['pixels'],data['manifest']['windows'],data['prototypeIds'],projection)
        uv=features(model,data['pixels'],data['manifest']['windows'],data['unknownIds'],projection)
        kv=features(model,data['knownPixels'],data['knownWindows'],list(range(len(data['known']))),projection)
    finally:model.head=head
    families=data['families']
    results={}
    for grouped in [False,True]:
        refs,owners,size=prototypes(pv,data['prototypeSamples'],families,grouped)
        results[str(4 if grouped else 1)]={'catalogTensorBytes':size,'known':metrics(kv,data['known'],families,refs,owners),'added':metrics(uv,data['unknown'],families,refs,owners)}
    return results


def evaluate():
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    start=time.perf_counter();path=ROOT/'models/hundred/model.json';model_hash=sha(path);data=load_data()
    model=load_export(data['artifact']).to('mps');results=score(model,data);projected=score(model,data,project=True);families=data['families']
    for count,result in results.items():print(f'{count} per family: known {result["known"]["all"]}, added {result["added"]["all"]}',flush=True)
    for count,result in projected.items():print(f'Learned head, {count} per family: known {result["known"]["all"]}, added {result["added"]["all"]}',flush=True)
    if sha(path)!=model_hash:raise ValueError('Encoder changed during catalog addition')
    write(ROOT/'bench/retrieval.json',{'modelSha256':model_hash,'catalog':families,'encoderTrainedFamilies':100,'addedDevelopmentFamilies':len(families)-100,
        'dimensions':64,'quantization':'int8 prototypes with float32 scales, cosine after decode','scope':'Frozen classifier features, synthetic development only. Added families were used for historical rejection validation but never encoder optimization. Six unknown-test families remain untouched. Known/added query text distributions differ; report separately.',
        'prototypeText':list(dict.fromkeys(s['text'] for s in data['prototypeSamples'])),'prototypeManifestSha256':sha(ROOT/'.data/hundred/prepared.json'),
        'developmentManifestSha256':sha(ROOT/'.data/preparation/prepared.json'),'seconds':time.perf_counter()-start,'results':results,'classifierMetric':projected,
        'projection':'SVD of the already-trained head; preserves bias-free logit dot products in at most 64 dimensions. This is learned neural geometry, no handcrafted pixel descriptor.'})


if __name__=='__main__':evaluate()
