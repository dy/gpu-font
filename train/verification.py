"""Post-selection synthetic verification; never a training/selection entry point."""
from pathlib import Path
import numpy as np
import torch
from train.robustness import read, write, sha
from train.ten_model import load_export
from train.next import predict
from train.hundred import probabilities, metrics
from train.hundred_data import band

ROOT=Path(__file__).resolve().parents[1]


def evaluate():
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    root=ROOT/'.data/verification';data=read(root/'prepared.json')
    for key,name in [('inputSha256','src/input.mjs'),('lineSha256','src/line.mjs'),('normalizerSha256','src/prepare.mjs')]:
        if data[key]!=sha(ROOT/name):raise ValueError('Changed verification preparation')
    rows=[{**s,'role':'verification','length':band(s['text'])} for s in data['samples']];results={}
    for name,folder,method in [('deployed','models/hundred','current'),('deskew','models/deskew','deskew-windows')]:
        path=ROOT/f'{folder}/model.json';artifact=read(path);calibration=read(ROOT/f'{folder}/calibration.json')
        preparation=artifact['preparation']
        if preparation['sha256']!=data['inputSha256'] or preparation['normalizerSha256']!=data['normalizerSha256']:raise ValueError('Model/verification preparation mismatch')
        if method=='deskew-windows' and (preparation.get('method')!=method or preparation.get('lineSha256')!=data['lineSha256']):raise ValueError('Model/deskew version mismatch')
        entry=data['methods'][method];tensor=root/f'{method}.u8'
        if sha(tensor)!=entry['sha256']:raise ValueError('Changed verification pixels')
        model=load_export(artifact).to('mps');pixels=np.memmap(tensor,mode='r',dtype=np.uint8)
        samples,logits,owners=predict(model,pixels,entry['windows'],rows)
        if len(samples)!=len(rows):raise ValueError('Lost verification samples')
        manifest={'samples':samples,'dataConfig':{'fonts':artifact['fonts']}}
        plain=metrics(manifest,list(range(len(samples))),probabilities(logits,owners,len(samples)))
        calibrated=metrics(manifest,list(range(len(samples))),probabilities(logits,owners,len(samples),calibration['temperature']),calibration['threshold'])
        results[name]={'modelSha256':sha(path),'calibrationSha256':sha(ROOT/f'{folder}/calibration.json'),'uncalibrated':plain,'calibrated':calibrated}
        print(name,plain['groups']['all'],calibrated['rejection'],flush=True)
    write(ROOT/'bench/verification.json',{'manifestSha256':sha(root/'prepared.json'),'scope':'New held-out synthetic texts after selecting both checkpoints. Two familiar renderers; not independent screenshots. Known families and six absent families reported separately. No thresholds fitted here.','results':results})


if __name__=='__main__':evaluate()
