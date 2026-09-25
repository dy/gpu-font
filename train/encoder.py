"""Equal-budget fresh encoders: episodic catalog selection versus classification."""
import argparse
from collections import Counter, defaultdict
import hashlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train.encoder_data import DATA, ROOT, SPLIT, SEED, save, validate_shard
from train.robustness import read, sha
from train.ten import batch
from train.ten_model import Classifier, export, load_export

DIMENSIONS = 128


def unit(vectors):
    vectors = np.asarray(vectors,dtype=np.float32)
    if vectors.ndim!=2 or not vectors.size or not np.isfinite(vectors).all(): raise ValueError('Invalid embedding matrix')
    norm=np.linalg.norm(vectors,axis=1,keepdims=True)
    if np.any(norm<=1e-12): raise ValueError('Zero embedding')
    return vectors/norm


def load_data(phase='development'):
    manifest=read(DATA/f'{phase}.json');path=DATA/f'{phase}.u8';split=read(SPLIT)
    plan_path=DATA/f'{phase}-plan.json';plan=read(plan_path)
    if manifest.get('planSha256')!=sha(plan_path) or manifest.get('pins')!=plan.get('pins'):raise ValueError('Changed frozen rendering plan')
    for file,expected in manifest['pins'].items():
        if sha(ROOT/file)!=expected:raise ValueError('Changed data dependency: '+file)
    if manifest['sha256']!=sha(path):raise ValueError('Changed training pixels')
    validate_shard(manifest,path.stat().st_size)
    texts=defaultdict(lambda:defaultdict(set));coverage=Counter()
    for s in manifest['samples']:
        if s.get('family') not in split['families'] or s.get('split')!=split['families'][s['family']] or s.get('renderer') not in ['pillow','chromium']:raise ValueError('Invalid sample identity')
        allowed=['reference','test'] if phase=='final' else ['reference','validation']+(['train'] if s['split']=='train' else [])
        if s.get('role') not in allowed or (phase=='development' and s['split']=='test'):raise ValueError('Held-out family/role leakage')
        if not isinstance(s.get('text'),str) or not s['text']:raise ValueError('Empty specimen text')
        texts[s['family']][s['role']].add(s['text'].casefold());coverage[s['family'],s['role'],s['renderer']]+=1
    for family,pools in texts.items():
        values=list(pools.values())
        if any(a&b for i,a in enumerate(values) for b in values[i+1:]):raise ValueError('Reference/query/training text leakage')
        roles=['reference','test'] if phase=='final' else ['reference','validation']+(['train'] if split['families'][family]=='train' else [])
        if any(not coverage[family,r,renderer] for r in roles for renderer in ['pillow','chromium']):raise ValueError('Missing family/renderer coverage')
    expected={f for f,r in split['families'].items() if phase=='final' or r!='test'}
    if set(texts)!=expected:raise ValueError('Missing experiment families')
    return manifest,np.memmap(path,dtype=np.uint8,mode='r'),split


def training_pools(manifest,split):
    pools=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    for i,w in enumerate(manifest['windows']):
        s=manifest['samples'][w['source']]
        if s['role']!='train':continue
        if split['families'].get(s['family'])!='train':raise ValueError('Held-out family in training pool')
        pools[s['family']][s['script']][s['text']][s['renderer']].append(i)
    expected={f for f,r in split['families'].items() if r=='train'}
    if set(pools)!=expected:raise ValueError('Missing training family')
    for scripts in pools.values():
        for texts in scripts.values():
            if len(texts)<2 or any(set(renderers)!={'pillow','chromium'} for renderers in texts.values()):raise ValueError('Missing paired training views')
    return pools


def episode(pools,rng,neighbors=None,count=48):
    families=sorted(pools)
    if len(families)<2 or count<2:raise ValueError('Episode requires distinct families')
    anchor=str(rng.choice(families));script=str(rng.choice(sorted(pools[anchor])))
    shared=rng.choice(sorted(pools[anchor][script]),2,replace=False).tolist()
    available=[f for f in families if script in pools[f] and all(t in pools[f][script] for t in shared)]
    chosen=[anchor]
    if neighbors:
        hard=[f for f in neighbors.get(anchor,[]) if f!=anchor and f in available]
        if hard:chosen.extend(rng.choice(hard,min(8,len(hard),count-1),replace=False).tolist())
    same=[f for f in available if f not in chosen]
    if same:chosen.extend(rng.choice(same,min(count-len(chosen),len(same)),replace=False).tolist())
    rest=[f for f in families if f not in chosen]
    if len(chosen)<min(count,len(families)):chosen.extend(rng.choice(rest,min(count,len(families))-len(chosen),replace=False).tolist())
    rng.shuffle(chosen)
    indexes=[];labels=[];scripts=[]
    for family in chosen:
        own=script if script in pools[family] else str(rng.choice(sorted(pools[family])))
        pair=shared if own==script and all(t in pools[family][own] for t in shared) else rng.choice(sorted(pools[family][own]),2,replace=False)
        first=int(rng.integers(2))
        for j,text in enumerate(pair):
            renderer=['pillow','chromium'][(first+j)%2]
            indexes.append(int(rng.choice(pools[family][own][text][renderer])));labels.append(family);scripts.append(own)
    return indexes,labels,scripts


def alias_targets(families,scripts,candidates,aliases):
    positions={f:i for i,f in enumerate(candidates)}
    if len(positions)!=len(candidates):raise ValueError('Duplicate candidates')
    target=np.zeros((len(families),len(candidates)),dtype=np.float32)
    for i,(family,script) in enumerate(zip(families,scripts)):
        equivalent=aliases.get((family,script),[family]);ids=[positions[f] for f in equivalent if f in positions]
        if not ids:raise ValueError('Missing positive candidate')
        target[i,ids]=1/len(ids)
    return torch.from_numpy(target)


def episodic_loss(vectors,target):
    if vectors.ndim!=2 or len(vectors)<4 or len(vectors)%2 or target.shape!=(len(vectors)//2,len(vectors)//2):raise ValueError('Invalid episodic batch')
    a,b=F.normalize(vectors[::2],dim=1),F.normalize(vectors[1::2],dim=1)
    return (F.cross_entropy(a@b.T/.1,target)+F.cross_entropy(b@a.T/.1,target))*.5


def embed(model,manifest,pixels,sources):
    if not sources or len(set(sources))!=len(sources):raise ValueError('Expected unique embedding sources')
    mapping={s:i for i,s in enumerate(sources)}
    ids=[i for i,w in enumerate(manifest['windows']) if w['source'] in mapping]
    sums=np.zeros((len(sources),model.head.out_features),dtype=np.float32);counts=np.zeros(len(sources),dtype=np.int32)
    model.eval();device=next(model.parameters()).device
    with torch.no_grad():
        for start in range(0,len(ids),128):
            indexes=ids[start:start+128]
            vectors=unit(model(*(t.to(device) for t in batch(pixels,manifest['windows'],indexes))).cpu().numpy())
            owners=[mapping[manifest['windows'][i]['source']] for i in indexes]
            np.add.at(sums,owners,vectors);np.add.at(counts,owners,1)
    if np.any(counts==0):raise ValueError('Missing source windows')
    return unit(sums/counts[:,None])


def references(vectors,samples,families,count=1,quantized=False):
    if count not in [1,4] or len(vectors)!=len(samples):raise ValueError('Invalid reference budget')
    rows=[];owners=[];pools=defaultdict(list)
    for i,s in enumerate(samples):
        length=None if count==1 else '8+short' if len(s['text'])==8 else '8+long' if len(s['text'])>=12 else s['length']
        pools[s['family'],length].append(i)
    for label,family in enumerate(families):
        for length in [None] if count==1 else ['2-3','4-7','8+short','8+long']:
            ids=pools[family,length]
            if not ids:raise ValueError('Missing reference coverage: '+family)
            rows.append(vectors[ids].mean(0));owners.append(label)
    rows=unit(rows);scale=np.maximum(np.abs(rows).max(1,keepdims=True)/127,1e-12)
    packed=np.round(rows/scale).clip(-127,127).astype(np.int8)
    return unit(packed.astype(np.float32)*scale) if quantized else rows,np.asarray(owners),packed,scale.astype(np.float32)


def rank(vectors,refs,owners,families):
    if refs.ndim!=2 or vectors.ndim!=2 or refs.shape[1]!=vectors.shape[1] or len(refs)!=len(owners) or not len(families) or len(set(families))!=len(families):raise ValueError('Invalid catalog dimensions')
    if set(owners.tolist())!=set(range(len(families))):raise ValueError('Missing catalog owner')
    # Bounded batches avoid a queries × references allocation for the complete corpus.
    scores=np.empty((len(vectors),len(families)),dtype=np.float32)
    order=np.argsort(owners,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(owners[order]))+1]
    for start in range(0,len(vectors),256):
        similarity=vectors[start:start+256]@refs[order].T
        scores[start:start+256]=np.maximum.reduceat(similarity,starts,axis=1)
    return scores


def metrics(scores,samples,families):
    positions={f:i for i,f in enumerate(families)};groups={};confusions=Counter()
    ranking=np.argsort(-scores,axis=1,kind='stable')[:,:min(5,len(families))]
    for row,s in zip(ranking,samples):
        if s['family'] not in positions:continue
        label=positions[s['family']];hit=int(row[0]==label);top5=int(label in row)
        if not hit:confusions[s['family'],families[row[0]]]+=1
        for key in ['all','split/'+s['split'],'renderer/'+s['renderer'],s['split']+'/'+s['renderer'],'length/'+s['length'],'condition/'+s['condition'],'script/'+s['script'],'family/'+s['family']]:
            g=groups.setdefault(key,{'count':0,'correct':0,'top5':0});g['count']+=1;g['correct']+=hit;g['top5']+=top5
    for g in groups.values():g['top1']=g['correct']/g['count'];g['top5Accuracy']=g['top5']/g['count']
    for split in ['train','development','test']:
        fs=sorted({s['family'] for s in samples if s['split']==split and s['family'] in positions})
        if fs:
            g=groups['split/'+split];g['macroTop1']=float(np.mean([groups['family/'+f]['top1'] for f in fs]));g['macroTop5']=float(np.mean([groups['family/'+f]['top5Accuracy'] for f in fs]))
    return {'catalogFamilies':len(families),'groups':groups,'confusions':[{'true':a,'predicted':b,'count':n} for (a,b),n in confusions.most_common(30)]}


def selection_sources(manifest,quick=False):
    samples=manifest['samples'];refs=[i for i,s in enumerate(samples) if s['role']=='reference']
    known=sorted({s['family'] for s in samples if s['split']=='train'},key=lambda f:hashlib.sha256(f.encode()).hexdigest())[:80]
    queries=[i for i,s in enumerate(samples) if s['role'] in ['validation','test'] and (not quick or (s['condition']=='clean' and s['length']=='8+' and (s['split']!='train' or s['family'] in known)))]
    return refs,queries


def assess(model,manifest,pixels,quick=False,counts=(1,),quantized=False):
    rids,qids=selection_sources(manifest,quick);rs=[manifest['samples'][i] for i in rids];qs=[manifest['samples'][i] for i in qids]
    families=sorted({s['family'] for s in rs});rv=embed(model,manifest,pixels,rids);qv=embed(model,manifest,pixels,qids);results={}
    for count in counts:
        refs,owners,_,_=references(rv,rs,families,count,quantized)
        results[str(count)]=metrics(rank(qv,refs,owners,families),qs,families)
    centers,_,_,_=references(rv,rs,families)
    trained={s['family'] for s in rs if s['split']=='train'}
    train=[i for i,f in enumerate(families) if f in trained]
    neighbor_scores=centers[train]@centers[train].T;np.fill_diagonal(neighbor_scores,-np.inf)
    nearest=np.argsort(-neighbor_scores,axis=1,kind='stable')[:,:24]
    neighbors={families[train[i]]:[families[train[j]] for j in row] for i,row in enumerate(nearest)}
    return results,neighbors,(families,rs,rv,qs,qv)


def training_pins():
    return {p:sha(ROOT/p) for p in ['train/encoder.py','train/ten_model.py','train/ten.py','bench/encoder-split.json','.data/encoder/development.json']}


def train(method,steps=20000,resume=False):
    if method not in ['episodic','classification'] or type(steps) is not int or steps<1:raise ValueError('Invalid training settings')
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    manifest,pixels,split=load_data();pools=training_pools(manifest,split);families=sorted(pools)
    aliases={(f,g['script']):g['families'] for g in split['renderingAliases'] for f in g['families']}
    model=Classifier(DIMENSIONS,context=True,wide=True,dilations=[1,1,2,2,1]).to('mps')
    classifier=nn.Linear(DIMENSIONS,len(families)).to('mps')
    parameters=list(model.parameters())+(list(classifier.parameters()) if method=='classification' else [])
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.00003)
    rng=np.random.default_rng(SEED);out=DATA/method;out.mkdir(parents=True,exist_ok=True)
    pins=training_pins();history=[];best=-1;neighbors=None;first=1;elapsed=0
    if resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if state['pins']!=pins or state['steps']!=steps or state['method']!=method:raise ValueError('Changed resume experiment')
        model.load_state_dict(state['state']);classifier.load_state_dict(state['classifier']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state=state['rng'];history=state['history'];best=state['best'];neighbors=state['neighbors'];first=state['step']+1;elapsed=state['seconds']
    elif (out/'last.pt').exists():raise ValueError('Existing experiment: use --resume or a new experiment directory')
    start=time.perf_counter();losses=[]
    def checkpoint(step):
        state={'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'classifier':{k:v.detach().cpu().clone() for k,v in classifier.state_dict().items()},
               'step':step,'steps':steps,'method':method,'pins':pins,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng.bit_generator.state,
               'history':history,'best':best,'neighbors':neighbors,'seconds':elapsed+time.perf_counter()-start}
        torch.save(state,out/'last.part');(out/'last.part').replace(out/'last.pt')
    def retain(step):
        nonlocal best,neighbors
        result,neighbors,_=assess(model,manifest,pixels,quick=True)
        groups=result['1']['groups'];known=groups['split/train']['macroTop5'];unseen=groups['split/development']['macroTop5'];value=2*known*unseen/max(known+unseen,1e-12)
        history.append({'step':step,'known':groups['split/train'],'unseen':groups['split/development'],'selection':value,'seconds':elapsed+time.perf_counter()-start})
        if value>best:
            best=value;torch.save({'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'step':step,'pins':pins,'method':method},out/'best.pt')
        save(out/'progress.json',{'method':method,'steps':steps,'history':history,'best':best})
        print(f'{method} {step}: known top5 {known:.2%}, unseen top5 {unseen:.2%}',flush=True)
    if first==1:retain(0);checkpoint(0)
    for step in range(first,steps+1):
        model.train();classifier.train();ids,labels,scripts=episode(pools,rng,neighbors)
        optimizer.zero_grad(set_to_none=True)
        vectors=model(*(t.to('mps') for t in batch(pixels,manifest['windows'],ids)))
        if method=='episodic':
            targets=alias_targets(labels[::2],scripts[::2],labels[::2],aliases).to('mps');loss=episodic_loss(vectors,targets)
        else:
            targets=alias_targets(labels,scripts,families,aliases).to('mps');loss=F.cross_entropy(classifier(vectors),targets,label_smoothing=.02)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,10);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
        if step%1000==0 or step==steps:
            print(f'{method} {step}/{steps}: loss {np.mean(losses):.4f}, {elapsed+time.perf_counter()-start:.0f}s',flush=True);losses=[]
        if step%4000==0 or step==steps:retain(step)
        if step%1000==0 or step==steps:checkpoint(step)
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.cpu().load_state_dict(selected['state'])
    float_results,_,_=assess(model.to('mps'),manifest,pixels,counts=(1,4));model.cpu()
    preparation={'method':'deskew-windows','width':128,'height':48,'windows':3,'sha256':sha(ROOT/'src/input.mjs'),'normalizerSha256':sha(ROOT/'src/prepare.mjs'),'lineSha256':sha(ROOT/'src/line.mjs')}
    artifact,restored=export(model,[str(i) for i in range(DIMENSIONS)],preparation)
    del artifact['fonts'];artifact.update(kind='font-encoder',dimensions=DIMENSIONS,normalization='l2')
    save(out/'encoder.json',artifact)
    quant_results,_,_=assess(restored.to('mps'),manifest,pixels,counts=(1,4),quantized=True)
    report={'method':method,'seed':SEED,'steps':steps,'selectedStep':selected['step'],'seconds':elapsed+time.perf_counter()-start,'pins':pins,'history':history,
            'families':split['counts'],'encoderParameters':sum(p.numel() for p in model.parameters()),'encoderBytes':(out/'encoder.json').stat().st_size,
            'encoderSha256':sha(out/'encoder.json'),'validationFloat':float_results,'validationInt8':quant_results,
            'scope':'Fresh weights, grouped held-out families, disjoint strings, Pillow and Chromium in both arms. Synthetic development only; no test-family data used.'}
    save(ROOT/f'bench/encoder-{method}.json',report)
    print(method,'selected',selected['step'],'complete',flush=True)


def valid_dimensions(d):
    """An embedding has a multiple of 16 numbers, 16 to 1024, as src/catalog.mjs reads them."""
    return type(d) is int and 16<=d<=1024 and d%16==0


def load_encoder(path):
    artifact=read(path)
    if artifact.get('kind')!='font-encoder' or not valid_dimensions(artifact.get('dimensions')) or artifact.get('normalization')!='l2' or 'fonts' in artifact:raise ValueError('Invalid encoder schema')
    return load_export({**artifact,'fonts':[str(i) for i in range(artifact['dimensions'])]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=['episodic','classification']);p.add_argument('--steps',type=int,default=20000);p.add_argument('--resume',action='store_true');a=p.parse_args();train(a.method,a.steps,a.resume)
