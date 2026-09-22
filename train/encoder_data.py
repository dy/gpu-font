"""Frozen family groups and fresh, renderer-balanced data for catalog retrieval."""
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

from scripts.corpus import ROOT, CACHE, blob
from train.corpus_data import banks, texts, without_hints, pins as corpus_pins
from train.hundred_data import transform, band
from train.robustness import read, sha

DATA = ROOT / '.data/encoder'
SPLIT = ROOT / 'bench/encoder-split.json'
SEED = 20260922
ROLES = ['train', 'development', 'test']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(path)


def family_groups(families, aliases, repositories):
    ids = [f['id'] for f in families]
    if not ids or len(set(ids)) != len(ids): raise ValueError('Invalid family inventory')
    parent = {i: i for i in ids}; evidence = []
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def join(members, reason):
        members = sorted(set(members))
        if any(i not in parent for i in members): raise ValueError('Unknown alias family')
        if len(members) < 2: return
        for i in members[1:]: parent[root(i)] = root(members[0])
        evidence.append({'basis': reason, 'families': members})
    names = defaultdict(list); repos = defaultdict(list); binaries = defaultdict(list)
    modifiers = {'sans','serif','mono','monospace','display','text','narrow','condensed','expanded','rounded','round','sc','small','caps','pro','variable','flex','italic','regular','light','bold','thin','black','medium','semibold','extrabold','extracondensed','semicondensed','wide'}
    for f in families:
        words = re.findall(r'[a-z]+|[0-9]+', f['family'].casefold())
        key = 'noto' if words[0] == 'noto' else ' '.join(w for w in words if w not in modifiers and not w.isdigit())
        if key: names[key].append(f['id'])
        repository = repositories.get(f['id'])
        if repository: repos[repository].append(f['id'])
        selected = next(face for face in f['faces'] if face['path'] == f['selected'])
        binaries[selected['blob']].append(f['id'])
    for key, members in names.items(): join(members, 'name-lineage:' + key)
    for key, members in repos.items(): join(members, 'source-repository:' + key)
    for key, members in binaries.items(): join(members, 'identical-font-blob:' + key)
    for g in aliases: join(g['families'], 'identical-rendering:' + g['script'])
    groups = defaultdict(list)
    for i in ids: groups[root(i)].append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0]), evidence


def assign_groups(groups, seed=SEED):
    ids = [i for g in groups for i in g]
    if not groups or any(not g for g in groups) or len(ids) != len(set(ids)): raise ValueError('Invalid family groups')
    targets = dict(zip(ROLES, np.array([.7,.15,.15]) * len(ids)))
    counts = Counter(); result = {}
    ordered = sorted(groups, key=lambda g: (-len(g), hashlib.sha256(f'{seed}:{sorted(g)}'.encode()).hexdigest()))
    for g in ordered:
        role = max(ROLES, key=lambda r: targets[r] - counts[r])
        for i in g: result[i] = role
        counts[role] += len(g)
    return result


def freeze():
    if SPLIT.exists():
        split = read(SPLIT)
        if split['catalogSha256'] != sha(ROOT/'bench/corpus.json'): raise ValueError('Changed frozen inventory')
        print('Existing family split retained', split['counts'], flush=True); return
    catalog = read(ROOT/'bench/corpus.json'); families = [f for f in catalog['families'] if not f['excluded']]
    old = read(ROOT/'.data/corpus/prepared.json')
    if old['pins'] != corpus_pins(): raise ValueError('Stale rendering aliases')
    aliases = [{'script':g['script'], 'families':[old['fonts'][i] for i in g['labels']]} for g in old['aliases']]
    repositories = {}
    for f in families:
        path = CACHE/f['folder']/'METADATA.pb'; raw = path.read_bytes()
        if blob(raw) != f['metadataBlob']: raise ValueError('Changed family metadata')
        urls = re.findall(r'repository_url:\s*"([^"]+)"', raw.decode())
        if urls:
            url = urls[0].rstrip('/').removesuffix('.git').lower()
            if url not in ['https://github.com/google/fonts', 'https://github.com/googlefonts']:
                repositories[f['id']] = url
    groups, evidence = family_groups(families, aliases, repositories)
    assignment = assign_groups(groups); scripts = {}
    for role in ROLES:
        scripts[role] = dict(sorted(Counter(s for f in families if assignment[f['id']] == role for s in f['alphabets']).items()))
    save(SPLIT, {'version':1,'seed':SEED,'catalogSha256':sha(ROOT/'bench/corpus.json'), 'sourceCommit':catalog['commit'],
        'initialization':'fresh random weights; no historical checkpoint', 'families':assignment,
        'counts':dict(Counter(assignment.values())), 'groups':groups,'evidence':evidence,'scripts':scripts,'renderingAliases':aliases,
        'limits':'Conservative name/repository/blob/rendering groups; source metadata is not a complete genealogy. Shared rare-script groups cannot appear in every split.'})
    print('Frozen family split', dict(Counter(assignment.values())), 'groups',len(groups), flush=True)


def text_plan(family, pools):
    result = []
    for script, coverage in sorted(family['alphabets'].items()):
        alphabet = ''.join(c for c in pools[script] if c in coverage)
        if len(alphabet) < 8: alphabet = ''.join(sorted(coverage))[:64]
        # Reserve every historical bank, including the old final-test revision.
        used = []
        for role in ['train','validation','test']:
            old = texts(alphabet, script, role, used); used.extend(old)
            if role == 'test': used.extend(texts(alphabet,script,role,used,revision=2))
        for role in ['train','reference','validation','test']:
            rng=random.Random(int.from_bytes(hashlib.sha256(f'encoder-v1:{script}:{role}'.encode()).digest()[:8]))
            lengths=[2 if len({c.casefold() for c in alphabet})>=12 else 3,4,8,12]
            values=[];seen={t.casefold() for t in used};count=32 if role=='train' else 8
            for _ in range(100000):
                value=''.join(rng.choices(alphabet,k=lengths[len(values)%4]))
                if value.casefold() in seen:continue
                values.append(value);seen.add(value.casefold())
                if len(values)==count:break
            if len(values)!=count:raise ValueError('Exhausted encoder text bank')
            used.extend(values)
            for i, text in enumerate(values):
                index = int.from_bytes(hashlib.sha256(f'{script}:{role}:{i}'.encode()).digest()[:4])
                result.append({'script':script,'role':role,'text':text,'index':index,'size':[24,32,44,56][(i//4+i)%4],
                               'light':bool(i%2),'length':band(text)})
    return result


def prepare_plan(phase):
    split = read(SPLIT); catalog = read(ROOT/'bench/corpus.json')
    if split['catalogSha256'] != sha(ROOT/'bench/corpus.json'): raise ValueError('Changed frozen inventory')
    families = [f for f in catalog['families'] if not f['excluded']]; pools = banks(families)
    jobs = []
    for family in families:
        group = split['families'][family['id']]
        if phase == 'development' and group == 'test': continue
        roles = ['reference','test'] if phase == 'final' else ['reference','validation'] + (['train'] if group == 'train' else [])
        plans = []
        for plan in text_plan(family,pools):
            if plan['role'] not in roles: continue
            conditions = ['clean'] if plan['role']=='reference' else ['clean' if plan['index']%2 else 'augmented'] if plan['role']=='train' else ['clean','rotate','small']
            plans.extend({**plan,'condition':condition} for condition in conditions)
        jobs.append({'family':family,'split':group,'plans':plans})
    pins = {key:sha(ROOT/key) for key in ['bench/corpus.json','bench/encoder-split.json','train/encoder_data.py','scripts/encoder-browser.mjs','scripts/corpus-prepare.mjs','src/line.mjs','src/input.mjs','src/prepare.mjs','train/hundred_data.py','requirements.txt']}
    save(DATA/f'{phase}-plan.json', {'version':1,'phase':phase,'pins':pins,'jobs':jobs})
    print('Planned',phase,len(jobs),'families',sum(len(j['plans']) for j in jobs),'images per renderer',flush=True)


def render_face(family):
    path = DATA/'faces'/f'{family["id"]}.ttf'; meta = path.with_suffix('.json')
    face = next(f for f in family['faces'] if f['path'] == family['selected'])
    expected = {'blob':face['blob'],'axes':family['trainingAxes']}
    if path.exists() and meta.exists():
        previous = read(meta)
        if previous['source'] == expected and previous['sha256'] == sha(path): return
    raw = (CACHE/family['selected']).read_bytes()
    if blob(raw) != face['blob']: raise ValueError('Changed font binary')
    if face['axes']:
        with TTFont(io.BytesIO(raw),recalcTimestamp=False) as font:
            instantiateVariableFont(font,family['trainingAxes'],inplace=True)
            stream=io.BytesIO();font.save(stream);raw=stream.getvalue()
    path.parent.mkdir(parents=True,exist_ok=True); path.with_suffix('.part').write_bytes(raw);path.with_suffix('.part').replace(path)
    save(meta,{'source':expected,'sha256':sha(path)})


def faces(phase,workers):
    jobs=read(DATA/f'{phase}-plan.json')['jobs']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[pool.submit(render_face,j['family']) for j in jobs]
        for n,future in enumerate(as_completed(tasks),1):
            future.result()
            if n%100==0 or n==len(tasks):print('Normal face instances',n,'/',len(tasks),flush=True)


def render_pillow(job):
    family, plans, prefix, expected = job
    meta = prefix.with_suffix('.json'); packed = prefix.with_suffix('.u8')
    if meta.exists() and packed.exists():
        previous = read(meta)
        if previous['pins'] == expected and previous['sha256'] == sha(packed): return
    face = next(f for f in family['faces'] if f['path'] == family['selected']); path = CACHE/family['selected']
    if blob(path.read_bytes()) != face['blob']: raise ValueError('Changed font binary')
    cache = {}; unhinted = None; images = []
    def draw(plan):
        size = plan['size']
        if size not in cache:
            font = ImageFont.truetype(io.BytesIO(unhinted) if unhinted else str(path),size,layout_engine=ImageFont.Layout.RAQM)
            if face['axes']: font.set_variation_by_axes([family['trainingAxes'][tag] for tag in face['axes']])
            cache[size] = font
        font = cache[size]; left,top,right,bottom = font.getbbox(plan['text'])
        if right <= left or bottom <= top: raise ValueError('Empty text rendering')
        image = Image.new('L',(right-left+12,bottom-top+12),255)
        ImageDraw.Draw(image).text((6-left,6-top),plan['text'],font=font,fill=0)
        return transform(image,plan,plan['condition'])
    for plan in plans:
        try: image = draw(plan)
        except OSError as error:
            if unhinted or 'execution context too long' not in str(error): raise
            unhinted = without_hints(path); cache.clear(); image = draw(plan)
        images.append({'width':image.width,'height':image.height,'pixels':base64.b64encode(image.tobytes()).decode()})
    process = subprocess.run(['node',str(ROOT/'scripts/corpus-prepare.mjs')],input=json.dumps(images),text=True,capture_output=True,check=True)
    prepared = json.loads(process.stdout)
    if len(prepared) != len(plans) or any(not ws for ws in prepared): raise ValueError('Missing prepared specimens: '+family['id'])
    windows = []; prefix.parent.mkdir(parents=True,exist_ok=True)
    with packed.with_suffix('.part').open('wb') as out:
        for owner, items in enumerate(prepared):
            for w in items:
                raw = base64.b64decode(w['pixels'],validate=True)
                if len(raw) != w['width']*w['height']: raise ValueError('Invalid prepared image')
                windows.append({'source':owner,'offset':out.tell(),'width':w['width'],'height':w['height']}); out.write(raw)
    packed.with_suffix('.part').replace(packed)
    save(meta,{'pins':expected,'sha256':sha(packed),'samples':plans,'windows':windows,'unhintedFallback':unhinted is not None})


def render(phase,workers):
    plan = read(DATA/f'{phase}-plan.json'); expected = plan['pins']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(render_pillow,(j['family'],j['plans'],DATA/phase/'pillow'/j['family']['id'],expected)) for j in plan['jobs']]
        for n, future in enumerate(as_completed(tasks),1):
            future.result()
            if n%100==0 or n==len(tasks): print('Pillow',phase,n,'/',len(tasks),flush=True)


def validate_shard(meta, size):
    if not isinstance(meta,dict) or not isinstance(meta.get('samples'),list) or not meta['samples'] or not isinstance(meta.get('windows'),list): raise ValueError('Empty/invalid shard')
    end = 0; owners = set()
    for w in meta['windows']:
        if not isinstance(w,dict) or any(type(w.get(k)) is not int for k in ['source','offset','width','height']) or w['offset']!=end or not 0<=w['source']<len(meta['samples']) or not 1<=w['width']<=128 or not 1<=w['height']<=48: raise ValueError('Invalid shard window')
        end += w['width']*w['height']; owners.add(w['source'])
    if end != size or len(owners) != len(meta['samples']): raise ValueError('Incomplete shard pixels')


def pack(phase):
    plan = read(DATA/f'{phase}-plan.json'); samples=[]; windows=[]
    with (DATA/f'{phase}.u8.part').open('wb') as out:
        for job in plan['jobs']:
            family = job['family']['id']
            for renderer in ['pillow','chromium']:
                prefix=DATA/phase/renderer/family; meta=read(prefix.with_suffix('.json')); raw=prefix.with_suffix('.u8').read_bytes()
                if meta['pins']!=plan['pins'] or meta['sha256']!=hashlib.sha256(raw).hexdigest() or meta['samples']!=job['plans']: raise ValueError('Changed renderer shard: '+family)
                validate_shard(meta,len(raw)); offset=out.tell(); owner=len(samples)
                windows.extend({**w,'offset':w['offset']+offset,'source':w['source']+owner} for w in meta['windows'])
                samples.extend({**s,'family':family,'split':job['split'],'renderer':renderer} for s in meta['samples']); out.write(raw)
    (DATA/f'{phase}.u8.part').replace(DATA/f'{phase}.u8')
    save(DATA/f'{phase}.json',{'version':1,'pins':plan['pins'],'planSha256':sha(DATA/f'{phase}-plan.json'),'sha256':sha(DATA/f'{phase}.u8'),'samples':samples,'windows':windows})
    print('Packed',phase,len(samples),'images',len(windows),'windows',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['split','plan','faces','render','pack']);p.add_argument('--phase',choices=['development','final'],default='development');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.command=='split': freeze()
    elif a.command=='plan': prepare_plan(a.phase)
    elif a.command=='render': render(a.phase,a.workers)
    elif a.command=='faces': faces(a.phase,a.workers)
    else: pack(a.phase)
