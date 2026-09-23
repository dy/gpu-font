"""Catalog swap on collected previews: a face is indexed by its captures, one phrase stays out as the query, and the
font must come back from its own source's catalog alone, from all sources, and from all of them merged with Google Fonts.
References and queries share a capture pipeline (the source's tester at one size), so this bounds screenshots from above."""
import argparse

import numpy as np

from train.encoder import ROOT
from train.encoder_catalog import read_catalog
from train.encoder_data import save
from train.robustness import read, sha
from scripts.preview_catalog import RECIPES

PHRASES=sorted(RECIPES['words'])


def load(compiled):
    """Records and normalized embeddings of compiled archives, all from one encoder."""
    records,vectors,bindings=[],[],[]
    for path in compiled:
        report=read(path/'report.json');inputs=read(path/'inputs.json')
        if sha(path/'inputs.json')!=report['inputsSha256'] or sha(path/'embeddings.npy')!=report['embeddingSha256']:raise ValueError('Changed compiled archive: '+str(path))
        records+=inputs['samples'];vectors.append(np.load(path/'embeddings.npy'))
        bindings.append({'archive':path.name,'encoderSha256':report['encoderSha256'],'sourceManifestSha256':report['sourceManifestSha256'],'embeddingSha256':report['embeddingSha256']})
    if len({b['encoderSha256'] for b in bindings})!=1:raise ValueError('Archives compiled by different encoders')
    vectors=np.concatenate(vectors).astype(np.float64);vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    return records,vectors,bindings


def ranked(query,refs,families,google=None):
    """Families by their best reference, as the browser ranks them; `google` adds a grouped catalog's families."""
    best={}
    for score,family in zip(refs@query,families):
        if score>best.get(family,-2):best[family]=score
    if google is not None:best.update(zip(google.names,np.maximum.reduceat(google.refs@query,google.starts)))
    return sorted(best,key=lambda f:-best[f])


class Grouped:
    """A catalog's rows sorted by family, so each family's best row is one reduceat."""
    def __init__(self,refs,families):
        order=np.argsort(families,kind='stable');families=np.asarray(families)[order]
        self.refs=refs[order];self.starts=np.r_[0,np.flatnonzero(families[1:]!=families[:-1])+1];self.names=families[self.starts].tolist();self.set=set(self.names)


def measure(records,vectors,lineage,scope,google=None,references=None):
    """Each face's captures (minus the query's phrase, or only `references`) index it; each phrase queries it."""
    rows=[]
    for i,query in enumerate(records):
        if query['recipeId'] not in PHRASES or not scope(query):continue
        kept=[j for j,r in enumerate(records) if scope(r) and r['recipeId']!=query['recipeId'] and r['recipeId']!='provided-v1' and (references is None or r['recipeId'] in references)]
        order=ranked(vectors[i],vectors[kept],[records[j]['familyId'] for j in kept],google);truth=query['familyId'];near={truth}|lineage.get(truth,set())
        rows.append({'top1':order[0]==truth,'top5':truth in order[:5],'lineage1':order[0] in near,'lineage5':bool(near&set(order[:5])),
                     'google1':google is not None and order[0] in google.set,'family':truth,'first':order[0],'candidates':len(order)})
    if not rows:return None
    mean=lambda k:round(float(np.mean([r[k] for r in rows])),4)
    out={'queries':len(rows),'families':len({r['family'] for r in rows}),'candidates':max(r['candidates'] for r in rows),**{k:mean(k) for k in ['top1','top5','lineage1','lineage5']}}
    if google is not None:out['firstIsGoogle']=mean('google1')
    out['misses']=sorted({(r['family'],r['first']) for r in rows if not r['lineage5']})
    return out


def swap(compiled,catalog_path,encoder_path,lineage_path,output):
    records,vectors,bindings=load(compiled)
    if bindings[0]['encoderSha256']!=sha(encoder_path):raise ValueError('Archives were compiled by another encoder; recompile them')
    lineage={}
    for group in read(lineage_path)['groups']:
        for family in group['familyIds']:lineage.setdefault(family,set()).update(set(group['familyIds'])-{family})
    catalog=read(catalog_path);refs,owners,_=read_catalog(catalog,encoder_path);refs=np.asarray(refs,np.float64);refs/=np.linalg.norm(refs,axis=1,keepdims=True)
    names=[catalog['faces'][o]['familyId'] for o in owners];google=Grouped(refs,names)
    source=lambda r:r['source']['key'];collected=lambda r:source(r)!='google-fonts'
    results={'shipped':{},'alphabet':{}}
    for protocol,references in [('shipped',None),('alphabet',RECIPES['alphabet']|{'digits-v1'})]:
        for key in sorted({source(r) for r in records if collected(r)}):
            results[protocol][key]=measure(records,vectors,lineage,lambda r,key=key:source(r)==key,references=references)
        results[protocol]['all sources']=measure(records,vectors,lineage,collected,references=references)
        results[protocol]['with Google Fonts']=measure(records,vectors,lineage,collected,google,references)
    # Control: Google families captured by the same pipeline, searched in the Google catalog itself.
    control=[]
    for i,r in enumerate(records):
        if source(r)!='google-fonts' or r['recipeId'] not in PHRASES:continue
        truth=r['familyId'].split(':')[-1].replace('-','');order=ranked(vectors[i],refs[:0],[],google)
        control.append({'family':r['family'],'recipe':r['recipeId'],'rank':order.index(truth)+1})
    report={'encoderSha256':sha(encoder_path),'googleCatalogSha256':sha(catalog_path),'lineageSha256':sha(lineage_path),'archives':bindings,
            'protocols':{'shipped':'Every capture of the face except the query phrase (both alphabets, digits, the other phrase), one row each; the demo ships all of them.',
                         'alphabet':'Alphabet and digit captures only, so no reference shares words with the query.'},
            'credit':'top1/top5: the exact family. lineage1/lineage5: also a family of the same lineage group (Helvetica with Neue Haas Grotesk, Minion from two sources).',
            'results':results,'googleControl':{'queries':len(control),'top1':round(float(np.mean([c['rank']==1 for c in control])),4),'top5':round(float(np.mean([c['rank']<=5 for c in control])),4),'ranks':control},
            'scope':'Previews captured from each source\'s own tester; references and queries share its renderer and size, so real screenshots will score lower.'}
    save(output,report)
    for protocol,groups in results.items():
        for key,r in groups.items():print(f"{protocol:<9} {key:<18} queries {r['queries']:>3} families {r['families']:>3} candidates {r['candidates']:>5} top1 {100*r['top1']:5.1f} top5 {100*r['top5']:5.1f} lineage5 {100*r['lineage5']:5.1f}",flush=True)
    print('Google control top1',report['googleControl']['top1'],'top5',report['googleControl']['top5'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('compiled',nargs='*',default=['.data/catalogs/preview-pilot-v1','.data/catalogs/preview-pilot-v2'])
    p.add_argument('--catalog',default='models/encoder/google-fonts.json');p.add_argument('--encoder',default='models/encoder/encoder.json')
    p.add_argument('--lineage',default='bench/preview-lineage.json');p.add_argument('--output',default='bench/preview-swap.json');a=p.parse_args()
    swap([ROOT/c for c in a.compiled],ROOT/a.catalog,ROOT/a.encoder,ROOT/a.lineage,ROOT/a.output)
