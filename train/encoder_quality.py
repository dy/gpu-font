"""Measure a refinement before publishing its encoder and rebuilt references."""
import argparse
import base64
from pathlib import Path

import numpy as np
import torch

from train.encoder import ROOT, load_data, load_encoder, embed, selection_sources, rank, metrics, unit
from train.encoder_data import save, validate_shard
from train.encoder_references import OUT, prototypes
from train.encoder_catalog import preparation_hash, read_catalog
from train.robustness import read, sha

BASE=ROOT/'.data/encoder/classification/encoder.json'
CANDIDATE=ROOT/'.data/encoder/retrieval-refine/encoder.json'
SELECTION=ROOT/'bench/encoder-quality-selection.json'


def vectors(path,phase):
    target=OUT/sha(path)[:16]/phase;target.mkdir(parents=True,exist_ok=True)
    manifest,pixels,_=load_data(phase);rids,qids=selection_sources(manifest)
    binding={'encoderSha256':sha(path),'dataSha256':sha(ROOT/f'.data/encoder/{phase}.json')}
    record=target/'cache.json'
    if record.exists():
        previous=read(record)
        if previous['binding']!=binding or any(sha(target/name)!=value for name,value in previous['files'].items()):raise ValueError('Changed embedding cache')
        rv=np.load(target/'reference.npy',allow_pickle=False);qv=np.load(target/'query.npy',allow_pickle=False)
    else:
        model=load_encoder(path).to('mps')
        rv=embed(model,manifest,pixels,rids);print(path.parent.name,phase,'references',len(rv),flush=True)
        qv=embed(model,manifest,pixels,qids);print(path.parent.name,phase,'queries',len(qv),flush=True)
        np.save(target/'reference.npy',rv,allow_pickle=False);np.save(target/'query.npy',qv,allow_pickle=False)
        save(record,{'binding':binding,'files':{name:sha(target/name) for name in ['reference.npy','query.npy']}})
    rs=[manifest['samples'][i] for i in rids];qs=[manifest['samples'][i] for i in qids]
    if rv.shape!=(len(rs),128) or qv.shape!=(len(qs),128):raise ValueError('Wrong cached embedding dimensions')
    return rs,rv,qs,qv


def summary(result):
    return {key:group for key,group in result['groups'].items() if not key.startswith('family/')}


def reference_vectors(path,phase,rs,rv,families,method):
    refs,owners,packed,scales=prototypes(rs,rv,families,'mean' if method=='mean' else 'scripts')
    if method!='script-words':return refs,owners,packed,scales
    source=OUT/'word-references'/phase;manifest=read(source/'manifest.json');pixels_path=source/'pixels.u8'
    if sha(pixels_path)!=manifest['sha256']:raise ValueError('Changed word reference pixels')
    for file,expected in manifest['pins'].items():
        if sha(ROOT/file)!=expected:raise ValueError('Changed word reference dependency')
    validate_shard(manifest,pixels_path.stat().st_size)
    target=OUT/sha(path)[:16]/phase;cache=target/'words.npy';meta=target/'words.json'
    binding={'encoderSha256':sha(path),'manifestSha256':sha(source/'manifest.json')}
    if meta.exists():
        old=read(meta)
        if old['binding']!=binding or old['sha256']!=sha(cache):raise ValueError('Changed word embedding cache')
        vectors=np.load(cache,allow_pickle=False)
    else:
        vectors=embed(load_encoder(path).to('mps'),manifest,np.memmap(pixels_path,dtype=np.uint8,mode='r'),list(range(len(manifest['samples']))))
        np.save(cache,vectors,allow_pickle=False);save(meta,{'binding':binding,'sha256':sha(cache)})
    groups={};positions={f:i for i,f in enumerate(families)}
    for i,s in enumerate(manifest['samples']):
        if s['family'] in positions:groups.setdefault((s['family'],s['referenceGroup']),[]).append(i)
    rows=unit(np.array([vectors[ids].mean(0) for _,ids in sorted(groups.items())]))
    ws=np.maximum(np.abs(rows).max(1)/127,1e-12);wp=np.round(rows/ws[:,None]).clip(-127,127).astype(np.int8)
    return (np.concatenate([refs,unit(wp.astype(np.float32)*ws[:,None])]),np.r_[owners,[positions[f] for f,_ in sorted(groups)]],
            np.concatenate([packed,wp]),np.r_[scales,ws])


def select():
    if SELECTION.exists():raise ValueError('Selection already frozen')
    baseline=read(OUT/'reference-comparison.json')
    if baseline['binding']['encoderSha256']!=sha(BASE):raise ValueError('Changed baseline encoder')
    candidates=[]
    def add(name,path,method,report):
        known=report['groups']['split/train']['macroTop5'];unseen=report['groups']['split/development']['macroTop5']
        candidates.append({'name':name,'encoder':str(path.relative_to(ROOT)),'encoderSha256':sha(path),'referenceMethod':method,'selection':2*known*unseen/(known+unseen),'results':summary(report)})
        print(name,'selection',candidates[-1]['selection'],'unseen',report['groups']['split/development'],flush=True)
    for method in ['mean','scripts']:add('original-'+method,BASE,method,baseline['methods'][method]['results'])
    for name,path in [('original',BASE),('refined',CANDIDATE)]:
        rs,rv,qs,qv=vectors(path,'development');families=sorted({r['family'] for r in rs})
        for method in (['script-words'] if name=='original' else ['scripts','script-words']):
            refs,owners,_,_=reference_vectors(path,'development',rs,rv,families,method)
            result=metrics(rank(qv,refs,owners,families),qs,families);add(name+'-'+method,path,method,result)
            save(OUT/f'{name}-{method}-development.json',result)
    winner=max(candidates,key=lambda c:c['selection'])
    save(SELECTION,{'winner':winner['name'],'encoder':winner['encoder'],'encoderSha256':winner['encoderSha256'],'referenceMethod':winner['referenceMethod'],
                    'criterion':'Harmonic known/unseen macro top-5 across all development queries. Fresh phrases and historical final queries excluded.',
                    'refinement':read(CANDIDATE.parent/'selection.json'),'candidates':candidates})
    print('Selected',winner['name'],[(c['name'],c['selection']) for c in candidates],flush=True)


def catalog(path,rs,rv,method):
    original=read(ROOT/'models/encoder/google-fonts.json');families=[f['familyId'] for f in original['faces']]
    refs,owners,packed,scales=reference_vectors(path,'final',rs,rv,families,method)
    result={k:v for k,v in original.items() if k not in ['referencesPerFace','vectors']}
    result.update(version=3,encoderSha256=sha(path),preparationSha256=preparation_hash(read(path)['preparation']),referenceMethod=method)
    result['vectors']={'encoding':'int8-base64','shape':list(packed.shape),'data':base64.b64encode(packed.tobytes()).decode(),'scales':scales.tolist(),'owners':owners.tolist()}
    decoded,decoded_owners,_=read_catalog(result,path)
    np.testing.assert_allclose(decoded,refs,atol=1e-6);np.testing.assert_array_equal(decoded_owners,owners)
    return result,refs,owners,families


def confirm():
    selected=read(SELECTION);path=ROOT/selected['encoder']
    if sha(path)!=selected['encoderSha256']:raise ValueError('Changed selected encoder')
    phrase_path=OUT/'phrases.json';phrases=read(phrase_path);pixels_path=OUT/'phrases.u8'
    if phrases['sha256']!=sha(pixels_path):raise ValueError('Changed phrase pixels')
    for file,expected in phrases['pins'].items():
        if sha(ROOT/file)!=expected:raise ValueError('Changed phrase benchmark dependency')
    validate_shard(phrases,pixels_path.stat().st_size);pixels=np.memmap(pixels_path,dtype=np.uint8,mode='r')
    reports={};cases=read(OUT/'demo-before.json')
    for name,encoder,method in [('before',BASE,'mean'),('after',path,selected['referenceMethod'])]:
        rs,rv,qs,qv=vectors(encoder,'final');data,refs,owners,families=catalog(encoder,rs,rv,method)
        historical=metrics(rank(qv,refs,owners,families),qs,families)
        model=load_encoder(encoder).to('mps');pv=embed(model,phrases,pixels,list(range(len(phrases['samples']))))
        score=rank(pv,refs,owners,families);fresh=metrics(score,phrases['samples'],families)
        regression=[]
        for case in cases:
            arrays=[];windows=[];offset=0
            for window in case['inputs']:
                values=np.round(np.array(window['pixels'])*255).astype(np.uint8)
                windows.append({'source':0,'offset':offset,'width':window['width'],'height':window['height']});arrays.append(values);offset+=len(values)
            vector=embed(model,{'windows':windows},np.concatenate(arrays),[0])
            scores=rank(vector,refs,owners,families)[0];order=np.argsort(-scores,kind='stable');family=case['source']['known'].replace('-','')
            regression.append({'family':family,'rank':int(np.flatnonzero(np.array(families)[order]==family)[0]+1),'matches':[{'family':families[i],'score':float(scores[i])} for i in order[:5]]})
        reports[name]={'encoderSha256':sha(encoder),'referenceMethod':method,'historical':summary(historical),'phrases':summary(fresh),'regressions':regression}
        save(OUT/f'{name}-phrases.json',fresh)
        print(name,'phrases',fresh['groups']['all'],'regressions',[(c['family'],c['rank']) for c in regression],flush=True)
        if name=='after':
            save(OUT/'google-fonts.json',data)
            reports[name]['catalogSha256']=sha(OUT/'google-fonts.json');reports[name]['catalogBytes']=(OUT/'google-fonts.json').stat().st_size
    save(ROOT/'bench/encoder-quality.json',{'selectionSha256':sha(SELECTION),'phraseManifestSha256':sha(phrase_path),'reports':reports,
         'scope':'Frozen development selection, then fresh synthetic phrase confirmation. Historical final results are a regression check, not a new untouched test. Reported demo examples are explicit regressions. No calibrated certainty or real-photo accuracy claim.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['select','confirm']);args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    if args.command=='select':select()
    else:confirm()
