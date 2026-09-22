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

from scripts import corpus
from train.corpus_data import texts, banks, plans, without_hints
from train.corpus import widen
from train.ten_model import Classifier, export, load_export
from train.faces import qat_snapshot, Int8Weights


class CorpusTests(unittest.TestCase):
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
