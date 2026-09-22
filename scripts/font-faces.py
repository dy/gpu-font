"""Import genuine normal/italic, 400/700 faces from the pinned Google Fonts tree."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
from pathlib import Path
import urllib.request
from urllib.parse import quote
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT=Path(__file__).resolve().parents[1]
FONTS=['inter','roboto','open-sans','source-sans-3','montserrat','poppins','nunito','lora','merriweather','playfair-display']

def fetch(path,commit,blob=None):
    if not re.fullmatch('[a-f0-9]{40}',commit) or '..' in Path(path).parts or not path.startswith(('ofl/','apache/','ufl/')):raise ValueError('Invalid catalog path/revision')
    dest=ROOT/'.data/google'/commit/path
    if not dest.exists():
        url='https://raw.githubusercontent.com/google/fonts/'+commit+'/'+quote(path)
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'gpu-font'}),timeout=60) as response:data=response.read()
        if blob and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=blob:raise ValueError('Git blob mismatch')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    data=dest.read_bytes()
    if blob and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=blob:raise ValueError('Cached source mismatch')
    return dest


def main():
    tree=json.loads((ROOT/'bench/google-corpus.json').read_text());commit=tree['commit']
    files=[{'path':f['path'],'sha':f['blob'],'size':f['bytes']} for f in tree['files']]
    selected=[]
    for family in FONTS:
        candidates=[f for f in files if f['path'].split('/')[1]==family.replace('-','')]
        for italic in [False,True]:
            eligible=[f for f in candidates if ('Italic' in f['path'])==italic]
            for weight in [400,700]:
                variable=[f for f in eligible if 'wght' in f['path']]
                static=[f for f in eligible if Path(f['path']).stem.endswith(('BoldItalic' if italic else 'Bold') if weight==700 else ('-Italic' if italic else '-Regular'))]
                choices=variable or static
                if not choices:raise ValueError(f'Missing source {family} {weight} {italic}')
                selected.append((family,weight,italic,sorted(choices,key=lambda f:f['path'])[0]))
    unique={s[3]['path']:s[3] for s in selected}
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda f:fetch(f['path'],commit,f['sha']),unique.values()))
    licenses={}
    for f in unique.values():
        folder=f['path'].rsplit('/',1)[0]
        if folder not in licenses:licenses[folder]=fetch(folder+'/OFL.txt',commit).read_text()
    out=ROOT/'.data/faces';out.mkdir(exist_ok=True);faces=[]
    for family,weight,italic,item in selected:
        path=fetch(item['path'],commit,item['sha'])
        with TTFont(path,recalcTimestamp=False) as font:
            axes={a.axisTag:a.defaultValue for a in font['fvar'].axes} if 'fvar' in font else {}
            if 'wght' in axes:
                axis=next(a for a in font['fvar'].axes if a.axisTag=='wght')
                if not axis.minValue<=weight<=axis.maxValue:raise ValueError('Weight out of range')
                axes['wght']=weight
            if axes:instantiateVariableFont(font,axes,inplace=True)
            actual_italic=bool(font['OS/2'].fsSelection & 1)
            if font['OS/2'].usWeightClass!=weight or actual_italic!=italic:raise ValueError(f'Incorrect face: {path}')
            if any(ord(c) not in font.getBestCmap() for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'):raise ValueError('Missing Latin glyphs')
            style='italic' if italic else 'normal';ident=f'{family}-{weight}-{style}';dest=out/f'{ident}.ttf';font.save(dest)
            faces.append({'id':ident,'family':family,'weight':weight,'style':style,'axes':axes,'source':item['path'],'blob':item['sha'],'sourceSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'instanceSha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    (ROOT/'bench/font-faces.json').write_text(json.dumps({'commit':commit,'faces':faces,'licenses':licenses,'scope':'Ten families, genuine 400/700 normal/italic faces; no synthetic slant.'},indent=2)+'\n')
    (ROOT/'bench/google-corpus.json').write_text(json.dumps({'commit':commit,'source':'https://github.com/google/fonts','scope':'TTF inventory under ofl/apache/ufl; directory count is not deduplicated family count.','files':[{'path':f['path'],'blob':f['sha'],'bytes':f['size']} for f in files]},indent=2)+'\n')
    print(f'Imported {len(faces)} true faces at {commit}',flush=True)

if __name__=='__main__':main()
