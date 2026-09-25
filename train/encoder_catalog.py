"""Select on development, freeze the encoder, then index/evaluate final families."""
import argparse
import base64
import gzip
import hashlib
import json
import shutil

import numpy as np
import torch

from train.encoder import DATA, ROOT, SPLIT, DIMENSIONS, valid_dimensions, load_data, load_encoder, assess, references, rank, metrics, unit
from train.encoder_data import save
from train.robustness import read,sha
from train.ten_model import unpack_bits

SELECTION=ROOT/'bench/encoder-selection.json'


def preparation_hash(preparation):
    return hashlib.sha256(json.dumps(preparation,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def select():
    if SELECTION.exists():
        selected=read(SELECTION)
        if sha(DATA/selected['method']/'encoder.json')!=selected['encoderSha256']:raise ValueError('Changed frozen encoder')
        print('Existing frozen selection retained',selected,flush=True);return
    if (DATA/'final.json').exists():raise ValueError('Final data was already opened before selection')
    candidates=[]
    for method in ['episodic','classification']:
        path=ROOT/f'bench/encoder-{method}.json';report=read(path)
        if report['steps']!=20000 or report['encoderSha256']!=sha(DATA/method/'encoder.json') or report['pins']['bench/encoder-split.json']!=sha(SPLIT):raise ValueError('Incomplete or changed comparison')
        for count in [1,4]:
            groups=report['validationInt8'][str(count)]['groups'];known=groups['split/train']['macroTop5'];unseen=groups['split/development']['macroTop5']
            candidates.append({'method':method,'referencesPerFace':count,'knownMacroTop5':known,'unseenMacroTop5':unseen,'selection':2*known*unseen/max(known+unseen,1e-12),
                               'encoderSha256':report['encoderSha256'],'reportSha256':sha(path),'selectedStep':report['selectedStep']})
    selected=max(candidates,key=lambda c:(c['selection'],-c['referencesPerFace']))
    save(SELECTION,{**selected,'splitSha256':sha(SPLIT),'candidates':candidates,'criterion':'Harmonic mean of known/unseen macro family top-5 on full mixed-renderer development queries; ties prefer fewer references.'})
    print('Frozen selection',selected,flush=True)


def make_catalog(encoder_path,families,samples,vectors,count,inventory):
    artifact=read(encoder_path);refs,owners,packed,scales=references(vectors,samples,families,count,True)
    by_id={f['id']:f for f in inventory['families']};faces=[]
    for name in families:
        family=by_id[name];face=next(f for f in family['faces'] if f['path']==family['selected']);axes=family['trainingAxes']
        faces.append({'id':name+'/normal','familyId':name,'family':family['family'],'weight':axes.get('wght',face['weight']),
                      'style':'italic' if face['italic'] else 'normal','axes':axes,'scripts':sorted(family['alphabets']),
                      'sourcePath':family['selected'],'sourceBlob':face['blob']})
    return {'version':1,'kind':'font-catalog','encoderSha256':sha(encoder_path),'preparationSha256':preparation_hash(artifact['preparation']),
            'dimensions':DIMENSIONS,'referencesPerFace':count,'sourceCommit':inventory['commit'],'faces':faces,
            'vectors':{'encoding':'int8-base64','shape':list(packed.shape),'data':base64.b64encode(packed.tobytes()).decode(),'scales':scales[:,0].tolist(),'owners':owners.tolist()}},refs,owners


def read_catalog(catalog,encoder_path):
    encoder=read(encoder_path)
    if not isinstance(catalog,dict) or type(catalog.get('version'))is not int or catalog['version'] not in [1,2,3] or catalog.get('kind')!='font-catalog' or not valid_dimensions(catalog.get('dimensions')) or catalog['dimensions']!=encoder.get('dimensions',DIMENSIONS):raise ValueError('Invalid catalog schema')
    variable=catalog['version']==3
    if variable and 'referencesPerFace' in catalog:raise ValueError('Variable references must omit the fixed budget')
    if not variable and (type(catalog.get('referencesPerFace'))is not int or catalog['referencesPerFace'] not in [1,4]):raise ValueError('Invalid reference budget')
    if catalog.get('encoderSha256')!=sha(encoder_path) or catalog.get('preparationSha256')!=preparation_hash(encoder['preparation']):raise ValueError('Incompatible encoder/preparation')
    faces=catalog.get('faces')
    if not isinstance(faces,list) or not 1<=len(faces)<=100000 or any(not isinstance(f,dict) or any(not isinstance(f.get(k),str) or not f[k] for k in ['id','familyId','family']) for f in faces) or len({f['id'] for f in faces})!=len(faces):raise ValueError('Expected unique catalog faces')
    if catalog['version']==1 and len({f['familyId'] for f in faces})!=len(faces):raise ValueError('Version 1 requires unique families')
    names={}
    for face in faces:
        if names.setdefault(face['familyId'],face['family'])!=face['family']:raise ValueError('Conflicting family names')
    vectors=catalog.get('vectors')
    if not isinstance(vectors,dict) or not isinstance(vectors.get('shape'),list) or len(vectors['shape'])!=2:raise ValueError('Invalid vector shape')
    rows=vectors['shape'][0] if catalog['version']==3 else len(faces)*catalog['referencesPerFace']
    if type(rows)is not int or not len(faces)<=rows<=640000:raise ValueError('Invalid reference count')
    bits={'int8-base64':8,'int4-base64':4}.get(vectors.get('encoding'))  # src/catalog.mjs reads the same two
    dimensions=catalog['dimensions']
    if not bits or vectors.get('shape')!=[rows,dimensions] or any(type(n)is not int for n in vectors['shape']):raise ValueError('Invalid vector shape')
    try:raw=base64.b64decode(vectors['data'],validate=True)
    except (KeyError,TypeError,ValueError) as error:raise ValueError('Invalid vector bytes') from error
    if len(raw)!=rows*dimensions*bits//8:raise ValueError('Truncated or trailing vectors')
    scales=vectors.get('scales');owners=vectors.get('owners')
    if not isinstance(scales,list) or len(scales)!=rows or any(type(s) not in [float,int] or not np.isfinite(s) or s<=0 for s in scales):raise ValueError('Invalid vector scales')
    if not isinstance(owners,list) or len(owners)!=rows or any(type(i)is not int or not 0<=i<len(faces) for i in owners):raise ValueError('Invalid vector owners')
    counts=np.bincount(owners,minlength=len(faces))
    invalid=counts<1 if variable else counts!=catalog['referencesPerFace']
    if np.any(invalid):raise ValueError('Missing catalog owner')
    decoded=unpack_bits(raw,bits,rows*dimensions).reshape(rows,dimensions).astype(np.float32)*np.array(scales,dtype=np.float32)[:,None]
    # V1 scores unique families; V2 scores individual faces, grouped by family by the caller.
    return unit(decoded),np.array(owners),[f['familyId' if catalog['version']==1 else 'id'] for f in faces]


def rejection(scores,samples,families,threshold):
    positions={f:i for i,f in enumerate(families)};predicted=scores.argmax(1);top=scores.max(1);present=np.array([s['family'] in positions for s in samples])
    correct=np.array([positions.get(s['family'],-1)==i for s,i in zip(samples,predicted)]);accepted=np.zeros(len(top),dtype=bool) if threshold is None else top>=threshold
    return {'threshold':None if threshold is None else float(threshold),'presentQueries':int(present.sum()),'absentQueries':int((~present).sum()),'acceptedPresent':int((accepted&present).sum()),
            'presentCoverage':float((accepted&present).sum()/max(1,present.sum())),
            'acceptedPresentAccuracy':float((correct&accepted).sum()/max(1,(accepted&present).sum())),
            'absentAcceptance':float((accepted&~present).sum()/max(1,(~present).sum()))}


def absent_catalog(families,split,role):
    candidates=[g for g in split['groups'] if split['families'][g[0]]==role]
    groups=sorted(candidates,key=lambda g:hashlib.sha256(('unknown:'+g[0]).encode()).hexdigest())
    absent=set()
    for g in groups:
        absent.update(g)
        if len(absent)>=max(1,round(split['counts'][role]*.2)):break
    return [f for f in families if f not in absent],sorted(absent)


def calibrate(scores,samples,families):
    positions={f:i for i,f in enumerate(families)};top=scores.max(1);predicted=scores.argmax(1)
    present=np.array([s['family'] in positions for s in samples]);correct=np.array([positions.get(s['family'],-1)==i for s,i in zip(samples,predicted)])
    if not present.any() or present.all():raise ValueError('Rejection needs present and absent queries')
    order=np.argsort(-top,kind='stable');ends=np.r_[np.flatnonzero(np.diff(top[order])),len(top)-1]
    n=np.cumsum(present[order])[ends];hits=np.cumsum(correct[order])[ends];unknown=np.cumsum(~present[order])[ends]
    good=(n>0)&(hits/np.maximum(n,1)>=.95)&(unknown/(~present).sum()<=.05)
    threshold=float(top[order[ends[np.flatnonzero(good)[-1]]]]) if good.any() else None
    return rejection(scores,samples,families,threshold)


def confidence_intervals(report,split):
    """Resample whole lineage groups, keeping equal weight per family within a draw."""
    result={};rng=np.random.default_rng(62922)
    for role in ['train','development','test']:
        groups=[[report['groups']['family/'+f] for f in group if 'family/'+f in report['groups']]
                for group in split['groups'] if split['families'][group[0]]==role]
        groups=[g for g in groups if g]
        if not groups:continue
        counts=np.array([len(g) for g in groups])
        totals=np.array([[sum(f[key] for f in g) for key in ['top1','top5Accuracy']] for g in groups])
        draws=rng.integers(len(groups),size=(2000,len(groups)))
        values=totals[draws].sum(1)/counts[draws].sum(1)[:,None]
        result[role]={'families':int(counts.sum()),'lineageGroups':len(groups),
                      'macroTop1':np.quantile(values[:,0],[.025,.975]).tolist(),
                      'macroTop5':np.quantile(values[:,1],[.025,.975]).tolist()}
    return {'method':'95% percentile intervals, 2,000 lineage-group bootstrap draws, seed 62922; conditional on these synthetic texts and this trained seed.','splits':result}


def evaluate_final():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    selected=read(SELECTION);method=selected['method'];encoder_path=DATA/method/'encoder.json';original_hash=sha(encoder_path)
    if original_hash!=selected['encoderSha256'] or sha(SPLIT)!=selected['splitSha256'] or sha(ROOT/f'bench/encoder-{method}.json')!=selected['reportSha256']:raise ValueError('Changed frozen selection')
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    model=load_encoder(encoder_path).to('mps');count=selected['referencesPerFace'];inventory=read(ROOT/'bench/corpus.json')
    dev,dp,split=load_data();quantized,_,data=assess(model,dev,dp,counts=(count,),quantized=True)
    families,rs,rv,qs,qv=data
    float_refs,float_owners,_,_=references(rv,rs,families,count,False)
    weights_only=metrics(rank(qv,float_refs,float_owners,families),qs,families)
    float_result=read(ROOT/f'bench/encoder-{method}.json')['validationFloat'][str(count)]
    quantization={name:{key:report['groups'][key] for key in ['all','split/train','split/development']} for name,report in
                  [('float',float_result),('int8Weights',weights_only),('int8WeightsAndReferences',quantized[str(count)])]}
    kept,absent=absent_catalog(families,split,'development')
    refs,owners,_,_=references(rv,rs,kept,count,True);calibration=calibrate(rank(qv,refs,owners,kept),qs,kept)
    calibration.update(catalogFamilies=len(kept),absentFamilies=absent,scope='Synthetic development-only threshold; no real-image probability claim.')
    del dev,dp,data,rv,qv,qs,rs
    manifest,pixels,split=load_data('final');results,_,data=assess(model,manifest,pixels,counts=(count,),quantized=True)
    families,rs,rv,qs,qv=data;catalog,refs,owners=make_catalog(encoder_path,families,rs,rv,count,inventory)
    out=ROOT/'models/encoder';out.mkdir(parents=True,exist_ok=True);shutil.copyfile(encoder_path,out/'encoder.json');shutil.copyfile(DATA/method/'best.pt',out/'best.pt')
    save(out/'google-fonts.json',catalog);decoded,decoded_owners,decoded_families=read_catalog(read(out/'google-fonts.json'),out/'encoder.json')
    np.testing.assert_allclose(decoded,refs,atol=1e-6);np.testing.assert_array_equal(decoded_owners,owners)
    if decoded_families!=families:raise ValueError('Catalog round-trip labels changed')
    kept,absent=absent_catalog(families,split,'test');known_refs,known_owners,_,_=references(rv,rs,kept,count,True)
    absent_result=rejection(rank(qv,known_refs,known_owners,kept),qs,kept,calibration['threshold'])
    absent_result.update(catalogFamilies=len(kept),absentFamilies=absent)
    sizes={}
    for name in ['encoder.json','google-fonts.json']:
        raw=(out/name).read_bytes();sizes[name]={'bytes':len(raw),'gzipBytes':len(gzip.compress(raw,compresslevel=9,mtime=0)),'sha256':sha(out/name)}
    if sum(v['bytes'] for v in sizes.values())>10_000_000:raise ValueError('Encoder/catalog exceed budget before runtime')
    # The same frozen vectors rank a reduced catalog; append final families, then reorder.
    previous=[f for f in families if split['families'][f]!='test'];old_refs,old_owners,_,_=references(rv,rs,previous,count,True)
    slices={'beforeFinalAddition':metrics(rank(qv,old_refs,old_owners,previous),qs,previous),'fullCatalog':results[str(count)]}
    order=np.arange(len(families))[::-1];inverse=np.argsort(order);reordered=[families[i] for i in order]
    a=rank(qv[:64],refs,owners,families);b=rank(qv[:64],refs,inverse[owners],reordered)
    np.testing.assert_array_equal(a,b[:,inverse])
    if sha(encoder_path)!=original_hash or sha(out/'encoder.json')!=original_hash:raise ValueError('Encoder changed during indexing')
    report={'selection':selected,'encoderSha256':original_hash,'catalogSha256':sha(out/'google-fonts.json'),'finalDataSha256':sha(DATA/'final.json'),
            'results':results[str(count)],'catalogAddition':slices,'developmentQuantization':quantization,'calibration':calibration,'rejection':absent_result,'payload':sizes,
            'confidenceIntervals':confidence_intervals(results[str(count)],split),
            'encoderUnchangedAfterIndexing':True,'reorderedScoresIdentical':True,
            'scope':'All 2,004 families indexed with frozen encoder. Final 300 families never optimized or used for selection. Strict identity, synthetic mixed-renderer queries; no independent screenshot or weight/style detection claim.'}
    save(ROOT/'bench/encoder-test.json',report)
    print('Final retrieval',results[str(count)]['groups']['split/test'],'payload',sizes,'rejection',absent_result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','test']);a=p.parse_args()
    if a.command=='select':select()
    else:evaluate_final()
