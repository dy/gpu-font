"""Compile raster-only reference catalogs with the frozen encoder, without training."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from train.encoder import ROOT, load_encoder, embed
from train.ten_model import pack_bits
from train.encoder_data import save, validate_shard
from train.encoder_catalog import preparation_hash, read_catalog
from train.robustness import read, sha

RECIPES={'all':{'latin-lower-v1','latin-upper-v1','words-a-v1','words-b-v1','digits-v1'},'words':{'words-a-v1','words-b-v1'},
         'alphabet':{'latin-lower-v1','latin-upper-v1'},'provided':{'provided-v1'}}
# The catalog to search with: every face by every line it has, so a face lacking glyphs for one line stands on the others.
# The recipes above hold one fixed set of lines per face, so that they compare like with like.
PARTIAL={'captured'};RECIPES['captured']=set().union(*RECIPES.values())


def make_catalog(records,vectors,encoder_path,manifest_hash,recipe):
    """Version 3: one reference row per capture, owned by its face, so a face scores by its best capture as Google Fonts
    faces do. One averaged row per face lost 21 points of top-5 once these faces met Google's per-case rows."""
    if recipe not in RECIPES or len(records)!=len(vectors):raise ValueError('Invalid reference recipe or vectors')
    ids=[i for i,r in enumerate(records) if r['recipeId'] in RECIPES[recipe]]
    if not ids:raise ValueError('No references for recipe '+recipe)
    selected=[records[i] for i in ids];face_ids=sorted({r['faceId'] for r in selected});faces=[]
    for face_id in face_ids:
        rows=[r for r in selected if r['faceId']==face_id];first=rows[0]
        if recipe not in PARTIAL and {r['recipeId'] for r in rows}!=RECIPES[recipe]:raise ValueError('Incomplete recipe for '+face_id)
        for key in ['familyId','family','styleName','weight','slant','axes','foundry','previewFile']:
            if len({json.dumps(r.get(key),sort_keys=True) for r in rows})!=1:raise ValueError('Conflicting face metadata: '+key)
        faces.append({'id':face_id,'familyId':first['familyId'],'family':first['family'],'styleName':first['styleName'],
                      'weight':first['weight'],'style':'normal' if first['slant']=='upright' else first['slant'],
                      'axes':first['axes'],'foundry':first['foundry'],'scripts':sorted({s for r in rows for s in r['scripts']}),
                      'sourceUrl':first['sourceUrl'],**({'previewFile':first['previewFile']} if first.get('previewFile') else {}),'referenceIds':[r['id'] for r in rows]})
    rows=sorted(range(len(selected)),key=lambda k:(face_ids.index(selected[k]['faceId']),selected[k]['id']))
    # Four bits a dimension, as the shipped catalogs pack their rows (bench/style.md, Quantization).
    unit=vectors[ids][rows];unit=unit/np.linalg.norm(unit,axis=1,keepdims=True);scales=np.maximum(np.abs(unit).max(1)/7,1e-12)
    packed=np.round(unit/scales[:,None]).clip(-7,7).astype(np.int8)
    owners=np.array([face_ids.index(selected[k]['faceId']) for k in rows]);encoder=read(encoder_path)
    catalog={'version':3,'kind':'font-catalog','encoderSha256':sha(encoder_path),'preparationSha256':preparation_hash(encoder['preparation']),
             'dimensions':encoder['dimensions'],'sourceManifestSha256':manifest_hash,'referenceRecipe':recipe,
             'faces':faces,'vectors':{'encoding':'int4-base64','shape':list(packed.shape),'data':base64.b64encode(pack_bits(packed,4)).decode(),
                                       'scales':scales.tolist(),'owners':owners.tolist()}}
    decoded,decoded_owners,labels=read_catalog(catalog,encoder_path)
    rows4=packed*scales[:,None];np.testing.assert_allclose(decoded,rows4/np.linalg.norm(rows4,axis=1,keepdims=True),atol=1e-6);np.testing.assert_array_equal(decoded_owners,owners)
    if labels!=face_ids:raise ValueError('Changed face order')
    return catalog


def complete(records,recipe):
    """Faces holding every capture of the recipe; partial faces wait for the collector instead of failing the archive."""
    have={}
    for r in records:
        if r['recipeId'] in RECIPES[recipe]:have.setdefault(r['faceId'],set()).add(r['recipeId'])
    ready={f for f,s in have.items() if s==RECIPES[recipe] or recipe in PARTIAL}
    return [i for i,r in enumerate(records) if r['faceId'] in ready],sorted(set(have)-ready)


def compile_catalogs(archive,encoder_path,output,device='cpu'):
    archive=Path(archive).resolve();output=Path(output).resolve();encoder_path=Path(encoder_path).resolve()
    if output.is_relative_to(archive):raise ValueError('Outputs must stay outside the source archive')
    before=sha(encoder_path);manifest_hash=sha(archive/'manifest.jsonl')
    output.mkdir(parents=True,exist_ok=True);prefix=output/'inputs'
    subprocess.run(['node',str(ROOT/'scripts/preview-prepare.mjs'),str(archive),str(encoder_path),str(prefix)],check=True,cwd=ROOT)
    manifest=read(prefix.with_suffix('.json'));raw=prefix.with_suffix('.u8').read_bytes()
    if manifest['encoderSha256']!=before or manifest['manifestSha256']!=manifest_hash or sha(prefix.with_suffix('.u8'))!=manifest['pixelSha256']:raise ValueError('Changed compiler inputs')
    validate_shard(manifest,len(raw));torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    model=load_encoder(encoder_path).to(device);records=manifest['samples']
    vectors=embed(model,manifest,np.frombuffer(raw,dtype=np.uint8),list(range(len(records))))
    np.save(output/'embeddings.npy',vectors,allow_pickle=False)
    products=[]
    for recipe in RECIPES:
        ids,skipped=complete(records,recipe)
        if not ids:continue
        catalog=make_catalog([records[i] for i in ids],vectors[ids],encoder_path,manifest_hash,recipe)
        target=output/f'preview-pilot-{recipe}.json';save(target,catalog)
        products.append({'path':target.name,'sha256':sha(target),'bytes':target.stat().st_size,'recipe':recipe,
                         'families':len({f['familyId'] for f in catalog['faces']}),'faces':len(catalog['faces']),'skippedFaces':skipped})
    if sha(encoder_path)!=before or sha(archive/'manifest.jsonl')!=manifest_hash:raise ValueError('Changed encoder or source manifest')
    report={'encoderSha256':before,'sourceManifestSha256':manifest_hash,'inputsSha256':sha(prefix.with_suffix('.json')),'records':len(records),'windows':len(manifest['windows']),
            'embeddingSha256':sha(output/'embeddings.npy'),'catalogs':products,'encoderUnchanged':True,
            'scope':'Catalog references only, generated from archived raster regions. No font files, training, accuracy evaluation, face-prediction calibration or redistribution claim.'}
    save(output/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',nargs='?',default='.data/previews/pilot-v1')
    p.add_argument('--encoder',default='models/encoder/encoder.json');p.add_argument('--output',default='.data/catalogs/preview-pilot-v1')
    p.add_argument('--device',choices=['cpu','mps'],default='cpu');a=p.parse_args()
    compile_catalogs(a.archive,a.encoder,a.output,a.device)
