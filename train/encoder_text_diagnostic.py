"""Opened-development diagnostic: separate image matching from text invariance."""
import argparse
from collections import Counter

import numpy as np
import torch

from train.encoder import ROOT, rank, metrics
from train.encoder_data import save
from train.encoder_quality import vectors, summary, BASE
from train.robustness import sha


def paired_sources(samples):
    eligible=[(i,s) for i,s in enumerate(samples) if s['script']=='Latn' and s['condition']=='clean' and s['size']==56 and s['length']=='8+']
    texts=[t for t,_ in Counter(s['text'] for _,s in eligible).most_common(2)]
    if len(texts)!=2:raise ValueError('Expected two independent long strings')
    lookup={}
    for i,s in eligible:
        if s['text'] not in texts:continue
        key=(s['family'],s['text'],s['renderer'])
        if key in lookup:raise ValueError('Duplicate paired source')
        lookup[key]=i
    families=sorted(f for f in {s['family'] for _,s in eligible} if all((f,t,r) in lookup for t in texts for r in ['pillow','chromium']))
    if not families:raise ValueError('No complete family pairs')
    return texts,families,lookup


def diagnose(path):
    _,_,samples,embeddings=vectors(path,'development')
    texts,families,lookup=paired_sources(samples);owners=np.arange(len(families));results={}
    for method in ['sameText','differentText']:
        all_scores=[];queries=[]
        for i,text in enumerate(texts):
            other=texts[i if method=='sameText' else 1-i]
            qids=[lookup[f,text,'chromium'] for f in families]
            rids=[lookup[f,other,'pillow'] for f in families]
            all_scores.append(rank(embeddings[qids],embeddings[rids],owners,families))
            queries.extend(samples[j] for j in qids)
        results[method]=summary(metrics(np.concatenate(all_scores),queries,families))
    report={'encoderSha256':sha(path),'families':len(families),'texts':texts,'size':56,'results':results}
    print(path.parent.name,{k:{s:v['split/'+s] for s in ['train','development']} for k,v in results.items()},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--large',action='store_true');args=parser.parse_args();torch.set_num_threads(4)
    paths=[BASE,ROOT/'.data/encoder/case-refine/encoder.json']
    if args.large:paths.append(ROOT/'.data/encoder/large-refine/encoder.json')
    save(ROOT/'bench/encoder-text-dependence.json',{'reports':[diagnose(p) for p in paths],
         'scope':'Diagnostic on opened development queries, repurposed as references. Chromium query versus Pillow reference at 56 px, same or different strings, equal one-reference budget and candidate families. This is not a new accuracy benchmark or proof that OCR improves end-to-end recognition. Changing text also changes glyph count and crop geometry.'})
