import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables.ttProgram import Program
from fontTools.ttLib.tables.TupleVariation import TupleVariation

from scripts import corpus
from train.corpus_data import texts, banks, plans, without_hints
from train.corpus import widen
import train.corpus as trainer
import train.corpus_data as corpus_data
from train.ten_model import Classifier, export, load_export
from train.faces import qat_snapshot, Int8Weights


class CorpusTests(unittest.TestCase):
    def test_normal_axes_and_rendering_do_not_use_a_variable_fonts_thin_default(self):
        axes={'wght':{'min':100,'default':100,'max':900},'wdth':{'min':75,'default':75,'max':100},'opsz':{'min':8,'default':14,'max':72}}
        self.assertEqual(corpus.normal_axes({'axes':axes}),{'wght':400,'wdth':100,'opsz':14})
        self.assertEqual(corpus.normal_axes({'axes':{'wght':{'min':500,'default':700,'max':900}}}),{'wght':500})
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus_data,'DATA',Path(tmp)), patch.object(corpus_data,'CACHE',Path(tmp)):
            root=Path(tmp); (root/'shards').mkdir(); path=root/'variable.ttf'
            fb=FontBuilder(1000,isTTF=True); fb.setupGlyphOrder(['.notdef','A'])
            pen=TTGlyphPen(None); pen.moveTo((0,0)); pen.lineTo((200,0)); pen.lineTo((100,600)); pen.closePath()
            fb.setupGlyf({'.notdef':TTGlyphPen(None).glyph(),'A':pen.glyph()})
            fb.setupHorizontalMetrics({'.notdef':(500,0),'A':(500,0)}); fb.setupHorizontalHeader(ascent=800,descent=-200)
            fb.setupCharacterMap({ord('A'):'A'}); fb.setupNameTable({'familyName':'Variable test','styleName':'Thin'})
            fb.setupOS2(usWeightClass=100); fb.setupPost(); fb.setupMaxp(); fb.setupFvar([('wght',100,100,900,'Weight')],[])
            gvar=newTable('gvar'); gvar.version=1; gvar.reserved=0
            gvar.variations={'A':[TupleVariation({'wght':(0,1,1)},[(0,0),(200,0),(100,0),(0,0),(200,0),(0,0),(0,0)])]}; fb.font['gvar']=gvar; fb.save(path)
            original=path.read_bytes(); face={'path':'variable.ttf','blob':corpus.blob(original),'axes':{'wght':axes['wght']}}
            family={'id':'normal','selected':'variable.ttf','faces':[face],'trainingAxes':{'wght':400}}
            plan={'size':40,'text':'A','role':'train','script':'Latn','index':0,'length':'1','light':False}
            with patch.object(corpus_data,'plans',return_value=[plan]):
                corpus_data.render_family((family,{},{}))
                corpus_data.render_family(({**family,'id':'thin','trainingAxes':{'wght':100}},{},{}))
            normal=json.loads((root/'shards/normal.json').read_text()); thin=json.loads((root/'shards/thin.json').read_text())
            self.assertGreater(normal['windows'][0]['width'],thin['windows'][0]['width'])
            self.assertEqual(path.read_bytes(),original)

    def test_packed_corpus_checks_splits_labels_and_both_sides_of_final_byte(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(trainer,'DATA',Path(tmp)), patch.object(trainer,'pins',return_value={'test':'pin'}):
            path=Path(tmp); raw=bytes([0,40,80,120,160,200]); (path/'prepared.u8').write_bytes(raw)
            samples=[{'label':i,'family':name,'role':role} for i,name in enumerate(['a','b']) for role in ['train','validation','test']]
            manifest={'fonts':['a','b'],'pins':{'test':'pin'},'sha256':hashlib.sha256(raw).hexdigest(),'samples':samples,
                      'windows':[{'source':i,'offset':i,'width':1,'height':1} for i in range(6)]}
            def load(value, pixels=raw):
                value=json.loads(json.dumps(value)); value['sha256']=hashlib.sha256(pixels).hexdigest()
                (path/'prepared.u8').write_bytes(pixels); (path/'prepared.json').write_text(json.dumps(value))
                return trainer.load_data()
            for _ in range(2):
                decoded,pixels=load(manifest); self.assertEqual(decoded['samples'],samples); self.assertEqual(pixels.tolist(),list(raw)); del pixels
            for pixels in [b'',raw[:-1],raw+b'\0']: self.assertRaisesRegex(ValueError,'Incomplete',load,manifest,pixels)
            for field,value in [('width',0),('height',49),('offset',0.0),('source',6)]:
                bad=json.loads(json.dumps(manifest)); bad['windows'][0][field]=value
                self.assertRaisesRegex(ValueError,'window',load,bad)
            for field,value in [('label',-1),('label',2),('label',True),('family','absent'),('role','other')]:
                bad=json.loads(json.dumps(manifest)); bad['samples'][0][field]=value
                self.assertRaisesRegex(ValueError,'sample',load,bad)
            bad=json.loads(json.dumps(manifest)); bad['samples'][0]['role']='test'
            self.assertRaisesRegex(ValueError,'split coverage',load,bad)
            bad=json.loads(json.dumps(manifest)); bad['pins']={}
            self.assertRaisesRegex(ValueError,'Changed',load,bad)
            decoded,pixels=load(manifest); self.assertEqual(pixels.tolist(),list(raw)); del pixels

    def test_pinned_fetch_reuses_exact_blob_and_rejects_corrupt_cache(self):
        data=b'source'; item={'path':'ofl/a/A.ttf','size':len(data),'sha':corpus.blob(data)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus,'CACHE',Path(tmp)), patch.object(corpus,'request',return_value=data) as request:
            a=corpus.fetch(item); self.assertEqual(a.read_bytes(),data)
            self.assertEqual(corpus.fetch(item),a); self.assertEqual(request.call_count,1)
            a.write_bytes(b'change'); self.assertRaisesRegex(ValueError,'checksum',corpus.fetch,item)
            a.write_bytes(data); self.assertEqual(corpus.fetch(item).read_bytes(),data)
            self.assertEqual(list(Path(tmp).rglob('*.part')),[])

    def test_source_paths_and_metadata_are_bounded_and_ignore_nested_names(self):
        for path in ['/ofl/a/a.ttf','ofl/../a.ttf','ofl/a/../../b.ttf','other/a/A.ttf','ofl//a/A.ttf']:
            self.assertRaises(ValueError,corpus.source_path,path)
        self.assertEqual(corpus.source_path('ofl/a/A.ttf'),corpus.CACHE/'ofl/a/A.ttf')
        text='name: "Quoted \\"face\\""\nfonts {\n  name: "Nested"\n}\nsubsets: "latin"\nsubsets: "greek"\n'
        self.assertEqual(corpus.metadata_strings(text,'name'),['Quoted "face"'])
        self.assertEqual(corpus.metadata_strings(text,'subsets'),['latin','greek'])
        self.assertEqual(corpus.metadata_strings('', 'name'),[])

    def test_cmap_excludes_notdef_and_records_true_weight_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'font.ttf'; fb=FontBuilder(1000,isTTF=True)
            fb.setupGlyphOrder(['.notdef','A','alef','blank','point'])
            glyphs={}
            for name in ['.notdef','A','alef']:
                pen=TTGlyphPen(None); pen.moveTo((0,0)); pen.lineTo((200,0)); pen.lineTo((100,600)); pen.closePath(); glyphs[name]=pen.glyph()
            glyphs['blank']=TTGlyphPen(None).glyph()
            pen=TTGlyphPen(None); pen.moveTo((0,0)); pen.lineTo((0,0)); pen.closePath(); glyphs['point']=pen.glyph()
            glyphs['A'].program=Program(); glyphs['A'].program.fromBytecode([0xB0,0,0x21])
            fb.setupGlyf(glyphs); fb.setupHorizontalMetrics({g:(500,0) for g in glyphs}); fb.setupHorizontalHeader(ascent=800,descent=-200)
            fb.setupCharacterMap({ord('A'):'A',ord('B'):'.notdef',ord('C'):'blank',ord('D'):'point',ord('א'):'alef'})
            fb.setupNameTable({'familyName':'Test','styleName':'Bold Italic'}); fb.setupOS2(usWeightClass=700,fsSelection=1); fb.setupHead(macStyle=2,created=3800000000,modified=3800000000); fb.setupPost(); fb.setupMaxp()
            fpgm=newTable('fpgm'); fpgm.program=Program(); fpgm.program.fromBytecode([0xB0,0,0x21]); fb.font['fpgm']=fpgm; fb.save(path)
            with patch.object(corpus,'source_path',return_value=path): face,groups=corpus.face_info({'path':'ofl/test/Test.ttf','sha':corpus.blob(path.read_bytes()),'size':path.stat().st_size})
            self.assertEqual(face['weight'],700); self.assertTrue(face['italic']); self.assertFalse(face['color'])
            self.assertEqual(groups,{'Latn':[ord('A')],'Hebr':[ord('א')]})
            original=path.read_bytes(); unhinted=without_hints(path)
            self.assertEqual(path.read_bytes(),original)
            with TTFont(io.BytesIO(unhinted)) as clean, TTFont(path) as source:
                self.assertNotIn('fpgm',clean); self.assertEqual(clean.getBestCmap(),source.getBestCmap())
                self.assertEqual(len(clean['glyf']['A'].program.getBytecode()),0)
                self.assertEqual(clean['glyf']['A'].getCoordinates(clean['glyf'])[0],source['glyf']['A'].getCoordinates(source['glyf'])[0])

    def test_script_texts_are_reproducible_disjoint_and_covered(self):
        families=[{'alphabets':{'Latn':'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz','Hebr':'אבגדהוזחטיכלמנסעפצקרשת'}}]
        pools=banks(families); a=plans(families[0],pools)
        self.assertEqual(a,plans(families[0],pools))
        only_latin={'alphabets':{'Latn':families[0]['alphabets']['Latn']}}
        self.assertEqual([s for s in a if s['script']=='Latn'],plans(only_latin,pools))
        split={r:{s['text'].casefold() for s in a if s['role']==r} for r in ['train','validation','test']}
        for x,y in [('train','validation'),('train','test'),('validation','test')]: self.assertFalse(split[x]&split[y])
        for sample in a: self.assertLessEqual(set(sample['text']),set(families[0]['alphabets'][sample['script']]))
        for script in families[0]['alphabets']:
            used=[s['text'] for s in a if s['script']==script and s['role']!='test']
            old=texts(pools[script],script,'test',used)
            fresh=[s['text'] for s in a if s['script']==script and s['role']=='test']
            self.assertFalse({t.casefold() for t in old}&{t.casefold() for t in fresh})
        self.assertRaises(ValueError,texts,'abc','Latn','train')
        small = {'alphabets':{'Latn':'ABCDabcd'}}
        self.assertEqual(len(plans(small,{'Latn':'ABCDabcd'})),80)

    def test_widening_preserves_logits_for_small_odd_and_maximum_inputs(self):
        torch.set_num_threads(2); torch.manual_seed(91)
        source=Classifier(3,context=True,dilations=[1,1,2,2,1]).eval(); wide=widen(source,3).eval()
        with torch.no_grad():
            wide.head.weight.copy_(source.head.weight.repeat_interleave(2,1)/2); wide.head.bias.copy_(source.head.bias)
            for h,w in [(1,1),(3,7),(48,128),(1,1)]:
                x=torch.rand(2,1,h,w)
                torch.testing.assert_close(wide(x),source(x),rtol=1e-5,atol=1e-6)
        artifact,restored=export(wide,['a','b','c'],{'width':128,'height':48,'windows':3})
        loaded=load_export(artifact)
        with torch.no_grad(): torch.testing.assert_close(loaded(x),restored(x),rtol=0,atol=0)
        self.assertEqual(loaded.head.in_features,128)

    def test_wide_qat_snapshot_preserves_live_model_and_dimensions(self):
        model=Classifier(2,training=False,context=True,wide=True)
        for layer in [*model.convs,model.head]: torch.nn.utils.parametrize.register_parametrization(layer,'weight',Int8Weights())
        for _ in range(2):
            snapshot=qat_snapshot(model)
            self.assertTrue(snapshot.wide); self.assertEqual(snapshot.head.in_features,128)
            model(torch.zeros(1,1,3,7)).sum().backward()
            self.assertIsNotNone(model.head.parametrizations.weight.original.grad)


if __name__=='__main__': unittest.main()
