"""Confirm case-diverse training after checkpoint selection; never optimize queries."""
import numpy as np
import torch

from train.encoder import ROOT, load_encoder, embed, rank, metrics
from train.encoder_data import save, validate_shard
from train.encoder_quality import OUT, BASE, vectors, reference_vectors, catalog, summary
from train.robustness import read, sha

CASE=ROOT/'.data/encoder/case-refine/encoder.json'


def confirm():
    selected=read(CASE.parent/'selection.json')
    if selected['encoderSha256']!=sha(CASE):raise ValueError('Changed selected case encoder')
    rs,rv,qs,qv=vectors(CASE,'development');families=sorted({s['family'] for s in rs})
    refs,owners,_,_=reference_vectors(CASE,'development',rs,rv,families,'script-words')
    development=metrics(rank(qv,refs,owners,families),qs,families)
    save(OUT/'case-development.json',development)
    print('Case full development',development['groups']['split/development'],flush=True)
    phrase_path=OUT/'phrases-next.json';phrase=read(phrase_path);pixel_path=OUT/'phrases-next.u8'
    if sha(pixel_path)!=phrase['sha256']:raise ValueError('Changed confirmation pixels')
    for file,expected in phrase['pins'].items():
        if sha(ROOT/file)!=expected:raise ValueError('Changed confirmation dependency')
    validate_shard(phrase,pixel_path.stat().st_size)
    pixels=np.memmap(pixel_path,dtype=np.uint8,mode='r');cases=read(OUT/'demo-before.json');reports={}
    for name,path,method in [('before',BASE,'mean'),('after',CASE,'script-words')]:
        rs,rv,qs,qv=vectors(path,'final');data,refs,owners,families=catalog(path,rs,rv,method)
        historical=metrics(rank(qv,refs,owners,families),qs,families)
        model=load_encoder(path).to('mps');pv=embed(model,phrase,pixels,list(range(len(phrase['samples']))))
        fresh=metrics(rank(pv,refs,owners,families),phrase['samples'],families);regressions=[]
        for case in cases:
            chunks=[];windows=[];offset=0
            for window in case['inputs']:
                raw=np.round(np.array(window['pixels'])*255).astype(np.uint8)
                windows.append({'source':0,'offset':offset,'width':window['width'],'height':window['height']});chunks.append(raw);offset+=len(raw)
            query=embed(model,{'windows':windows},np.concatenate(chunks),[0])
            scores=rank(query,refs,owners,families)[0];order=np.argsort(-scores,kind='stable');family=case['source']['known'].replace('-','')
            regressions.append({'family':family,'rank':int(np.flatnonzero(np.array(families)[order]==family)[0]+1),'matches':[{'family':families[i],'score':float(scores[i])} for i in order[:5]]})
        reports[name]={'encoderSha256':sha(path),'referenceMethod':method,'historical':summary(historical),'phrases':summary(fresh),'regressions':regressions}
        save(OUT/f'case-{name}-phrases.json',fresh)
        print(name,'new phrases',fresh['groups']['all'],'regressions',[(c['family'],c['rank']) for c in regressions],flush=True)
        if name=='after':
            save(OUT/'case-google-fonts.json',data)
            reports[name].update(catalogSha256=sha(OUT/'case-google-fonts.json'),catalogBytes=(OUT/'case-google-fonts.json').stat().st_size)
    save(ROOT/'bench/encoder-case-quality.json',{'checkpointSelection':selected,'development':summary(development),'phraseManifestSha256':sha(phrase_path),'reports':reports,
         'scope':'Second phrase bank frozen before case-training results. First phrase bank restricted to train/development families is now checkpoint-selection data. No final-family pixels enter optimization or selection. Historical final results and reported demo cases are regressions; synthetic evaluation is not real-image certainty.'})


if __name__=='__main__':
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    confirm()
