"""Continue the encoder with cross-text retrieval loss; select on development only."""
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train.encoder import (DATA, ROOT, DIMENSIONS, load_data, training_pools,
                          episode, alias_targets, episodic_loss, embed,
                          selection_sources, rank, metrics)
from train.encoder_data import SEED, save
from train.encoder_references import prototypes
from train.ten import batch
from train.ten_model import Classifier, export
from train.robustness import read, sha


def refine(steps=12000,case_training=False):
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    out=DATA/('case-refine' if case_training else 'retrieval-refine');out.mkdir(parents=True,exist_ok=True)
    if (out/'progress.json').exists():raise ValueError('Existing refinement experiment')
    manifest,pixels,split=load_data();pools=training_pools(manifest,split);families=sorted(pools)
    aliases={(f,g['script']):g['families'] for g in split['renderingAliases'] for f in g['families']}
    source=DATA/'classification/last.pt';state=torch.load(source,map_location='cpu',weights_only=False)
    for path,expected in state['pins'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Changed initialization dependency: '+path)
    model=Classifier(DIMENSIONS,context=True,wide=True,dilations=[1,1,2,2,1]).to('mps')
    classifier=nn.Linear(DIMENSIONS,len(families)).to('mps')
    model.load_state_dict(state['state']);classifier.load_state_dict(state['classifier'])
    parameters=list(model.parameters())+list(classifier.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.0002,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.00001)
    rng=np.random.default_rng(SEED+1);neighbors=state['neighbors'];history=[];best=-1;start=time.perf_counter()
    pins={path:sha(ROOT/path) for path in ['train/encoder_refine.py','train/encoder_references.py','train/encoder.py','bench/encoder-split.json','.data/encoder/development.json']}
    pins[str(source.relative_to(ROOT))]=sha(source)
    if case_training:
        from train.encoder_quality import with_words
        from train.encoder_data import validate_shard
        case_path=DATA/'case-training';cases=read(case_path/'manifest.json');case_pixels=np.memmap(case_path/'pixels.u8',dtype=np.uint8,mode='r')
        if sha(case_path/'pixels.u8')!=cases['sha256']:raise ValueError('Changed case-training pixels')
        for file,expected in cases['pins'].items():
            if sha(ROOT/file)!=expected:raise ValueError('Changed case-training dependency')
        validate_shard(cases,len(case_pixels))
        if any(s['role']!='train' or s['split']!='train' or split['families'].get(s['family'])!='train' for s in cases['samples']):raise ValueError('Held-out family in case training')
        reserved={s['text'].casefold() for s in manifest['samples'] if s['role']!='train'}
        if any(s['text'].casefold() in reserved for s in cases['samples']):raise ValueError('Case training text leakage')
        case_pools=training_pools(cases,{'families':{s['family']:'train' for s in cases['samples']}})
        phrase=read(ROOT/'.data/detection-quality/phrases.json');phrase_path=ROOT/'.data/detection-quality/phrases.u8'
        if sha(phrase_path)!=phrase['sha256']:raise ValueError('Changed phrase pixels')
        phrase_pixels=np.memmap(phrase_path,dtype=np.uint8,mode='r')
        phrase_ids=[i for i,s in enumerate(phrase['samples']) if s['split']!='test'];phrase_samples=[phrase['samples'][i] for i in phrase_ids]
        word_path=ROOT/'.data/detection-quality/word-references/development';words=read(word_path/'manifest.json')
        if sha(word_path/'pixels.u8')!=words['sha256']:raise ValueError('Changed word reference pixels')
        word_pixels=np.memmap(word_path/'pixels.u8',dtype=np.uint8,mode='r')
        for file in [case_path/'manifest.json',ROOT/'.data/detection-quality/phrases.json',word_path/'manifest.json',ROOT/'train/encoder_quality.py']:
            pins[str(file.relative_to(ROOT))]=sha(file)
    rids,qids=selection_sources(manifest,quick=True)
    rs=[manifest['samples'][i] for i in rids];qs=[manifest['samples'][i] for i in qids];candidates=sorted({s['family'] for s in rs})
    def retain(step):
        nonlocal best
        rv=embed(model,manifest,pixels,rids);qv=embed(model,manifest,pixels,qids)
        refs,owners,_,_=prototypes(rs,rv,candidates,'scripts')
        phrase_result=None
        if case_training:
            refs,owners,packed,scales=prototypes(rs,rv,candidates,'scripts')
            wv=embed(model,words,word_pixels,list(range(len(words['samples']))))
            refs,owners,_,_=with_words(refs,owners,packed,scales,candidates,words['samples'],wv)
            pv=embed(model,phrase,phrase_pixels,phrase_ids)
            phrase_result=metrics(rank(pv,refs,owners,candidates),phrase_samples,candidates)
        result=metrics(rank(qv,refs,owners,candidates),qs,candidates)
        known=result['groups']['split/train']['macroTop5'];unseen=result['groups']['split/development']['macroTop5'];value=2*known*unseen/(known+unseen)
        if phrase_result:
            a=phrase_result['groups']['split/train']['macroTop5'];b=phrase_result['groups']['split/development']['macroTop5']
            value=(value+2*a*b/max(a+b,1e-12))*.5
        history.append({'step':step,'selection':value,'results':result,'phraseDevelopment':phrase_result,'seconds':time.perf_counter()-start})
        if value>best:
            best=value;torch.save({'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'step':step,'pins':pins},out/'best.pt')
        save(out/'progress.json',{'steps':steps,'pins':pins,'history':history})
        print('Refine',step,'known',round(known,4),'unseen',round(unseen,4),'selection',round(value,4),'phraseTop1',phrase_result['groups']['all']['top1'] if phrase_result else None,'seconds',round(time.perf_counter()-start),flush=True)
    retain(0);losses=[]
    for step in range(1,steps+1):
        model.train();classifier.train()
        extra=case_training and step%2==0
        ids,labels,scripts=episode(case_pools if extra else pools,rng,neighbors)
        optimizer.zero_grad(set_to_none=True)
        vectors=model(*(t.to('mps') for t in batch(case_pixels if extra else pixels,cases['windows'] if extra else manifest['windows'],ids)))
        targets=alias_targets(labels,scripts,families,aliases).to('mps')
        pairs=alias_targets(labels[::2],scripts[::2],labels[::2],aliases).to('mps')
        loss=F.cross_entropy(classifier(vectors),targets,label_smoothing=.02)+.5*episodic_loss(vectors,pairs)
        if not torch.isfinite(loss):raise ValueError('Nonfinite refinement loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,10);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
        if step%1000==0 or step==steps:
            print('Refine',step,'loss',float(np.mean(losses)),'seconds',round(time.perf_counter()-start),flush=True);losses=[]
        if step%2000==0 or step==steps:retain(step)
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.cpu().load_state_dict(selected['state'])
    preparation=read(ROOT/'models/encoder/encoder.json')['preparation']
    artifact,_=export(model,[str(i) for i in range(DIMENSIONS)],preparation)
    del artifact['fonts'];artifact.update(kind='font-encoder',dimensions=DIMENSIONS,normalization='l2')
    save(out/'encoder.json',artifact)
    save(out/'selection.json',{'step':selected['step'],'encoderSha256':sha(out/'encoder.json'),'pins':pins,'seconds':time.perf_counter()-start,
         'criterion':'Equal weight to known/unseen harmonic macro top-5 on existing clean development and opened phrase queries restricted to train/development families; script/case references. Final families excluded.' if case_training else 'Harmonic known/unseen macro family top-5, existing clean development subset; per-script int8 references. Final queries and demo examples excluded.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=12000);parser.add_argument('--case-training',action='store_true');args=parser.parse_args()
    if args.steps<1:parser.error('steps must be positive')
    refine(args.steps,args.case_training)
