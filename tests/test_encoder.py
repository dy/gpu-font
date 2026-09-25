import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
import torch
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont,newTable
from fontTools.ttLib.tables.TupleVariation import TupleVariation

from train.encoder_data import family_groups,assign_groups,text_plan,validate_shard
from train.encoder import unit,episode,training_pools,alias_targets,episodic_loss,references,rank,metrics
from train.encoder import load_encoder
from train.encoder_data import save
from train.ten_model import Classifier,export
from train import encoder
from train import encoder_data
from train.robustness import sha
from scripts.corpus import blob


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_lineage_repository_binary_and_render_aliases_never_cross_splits(self):
        def font(i,name):return {'id':i,'family':name,'selected':i,'faces':[{'path':i,'blob':i}]}
        families=[font('a','Example Sans'),font('b','Example Serif'),font('c','Separate'),font('d','Another'),font('e','Else'),font('f','Unrelated')]
        families[4]['faces'][0]['blob']='d'
        groups,evidence=family_groups(families,[{'script':'Latn','families':['b','c']}],{'e':'repo','f':'repo'})
        self.assertEqual(groups,[['a','b','c'],['d','e','f']]);self.assertEqual(len(evidence),4)
        groups.extend([[str(i)] for i in range(40)])
        first=assign_groups(groups)
        self.assertEqual(first,assign_groups(list(reversed(groups))))
        self.assertEqual(set(first.values()),{'train','development','test'})
        for group in groups:self.assertEqual(len({first[f] for f in group}),1)
        for invalid in [[],[['a'],['a']],[[]]]:self.assertRaises(ValueError,assign_groups,invalid)
        self.assertRaises(ValueError,family_groups,families,[{'script':'Latn','families':['absent','a']}],{})

    def test_fresh_text_banks_cover_small_alphabets_without_cross_role_collisions(self):
        for alphabet in ['ABCDabcd','ABCDEFabcdef','ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz']:
            family={'alphabets':{'Latn':alphabet}};pools={'Latn':alphabet}
            plans=text_plan(family,pools)
            self.assertEqual(plans,text_plan(family,pools));self.assertEqual(len(plans),56)
            groups={role:{s['text'].casefold() for s in plans if s['role']==role} for role in ['train','reference','validation','test']}
            self.assertEqual([len(g) for g in groups.values()],[32,8,8,8])
            self.assertEqual(sum(map(len,groups.values())),len(set.union(*groups.values())))
            for s in plans:self.assertLessEqual(set(s['text']),set(alphabet))

    def test_shard_smallest_input_exact_final_boundary_and_repeated_recovery(self):
        a={'samples':[{'text':'a'}],'windows':[{'source':0,'offset':0,'width':1,'height':1}]}
        b={'samples':[{'text':'a'},{'text':'b'}],'windows':[a['windows'][0],{'source':1,'offset':1,'width':2,'height':1}]}
        for value,size in [(a,1),(a,1),(b,3),(a,1)]:validate_shard(value,size)
        for size in [0,2]:self.assertRaisesRegex(ValueError,'Incomplete',validate_shard,a,size)
        for size in [2,4]:self.assertRaisesRegex(ValueError,'Incomplete',validate_shard,b,size)
        for value in [None,{}, {'samples':[],'windows':[]}, {'samples':[{}],'windows':[]}]:self.assertRaises(ValueError,validate_shard,value,0)
        for key,value in [('source',True),('source',1),('offset',1),('width',0),('height',49)]:
            broken=copy.deepcopy(a);broken['windows'][0][key]=value;self.assertRaises(ValueError,validate_shard,broken,1)
        validate_shard(a,1)

    def test_loader_binds_plan_bytes_roles_renderers_and_family_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);split=root/'split.json';save(split,{'families':{'a':'train','b':'development','c':'test'}})
            save(root/'development-plan.json',{'pins':{'split.json':sha(split)}})
            samples=[]
            for family,group,roles in [('a','train',['train','reference','validation']),('b','development',['reference','validation'])]:
                for role in roles:
                    for renderer in ['pillow','chromium']:samples.append({'family':family,'split':group,'role':role,'renderer':renderer,'text':role})
            raw=bytes(range(len(samples)));path=root/'development.u8';path.write_bytes(raw)
            manifest={'pins':{'split.json':sha(split)},'planSha256':sha(root/'development-plan.json'),'sha256':sha(path),'samples':samples,
                      'windows':[{'source':i,'offset':i,'width':1,'height':1} for i in range(len(samples))]}
            def load(value,pixels=raw):
                path.write_bytes(pixels);value=copy.deepcopy(value);value['sha256']=sha(path);save(root/'development.json',value)
                return encoder.load_data()
            with patch.object(encoder,'DATA',root),patch.object(encoder,'ROOT',root),patch.object(encoder,'SPLIT',split):
                for _ in range(2):
                    actual,pixels,_=load(manifest);self.assertEqual(actual['samples'],samples);self.assertEqual(pixels.tolist(),list(raw));del pixels
                for data in [b'',raw[:-1],raw+b'\0']:self.assertRaises(ValueError,load,manifest,data)
                for key,value in [('pins',{}),('planSha256','wrong')]:self.assertRaisesRegex(ValueError,'plan',load,{**manifest,key:value})
                for key,value in [('family','c'),('role','test'),('renderer','other'),('text','reference')]:
                    bad=copy.deepcopy(manifest);bad['samples'][0][key]=value;self.assertRaises(ValueError,load,bad)
                _,pixels,_=load(manifest);self.assertEqual(pixels.tolist(),list(raw));del pixels

    def test_browser_face_freezes_variable_axes_and_recovers_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'variable.ttf';fb=FontBuilder(1000,isTTF=True)
            fb.setupGlyphOrder(['.notdef','A']);pen=TTGlyphPen(None);pen.moveTo((0,0));pen.lineTo((200,0));pen.lineTo((100,600));pen.closePath()
            fb.setupGlyf({'.notdef':TTGlyphPen(None).glyph(),'A':pen.glyph()});fb.setupHorizontalMetrics({'.notdef':(500,0),'A':(500,0)})
            fb.setupHorizontalHeader(ascent=800,descent=-200);fb.setupCharacterMap({ord('A'):'A'})
            fb.setupNameTable({'familyName':'Variable fixture','styleName':'Thin'});fb.setupOS2(usWeightClass=100);fb.setupPost();fb.setupMaxp()
            fb.setupFvar([('wght',100,100,900,'Weight')],[])
            gvar=newTable('gvar');gvar.version=1;gvar.reserved=0;gvar.variations={'A':[TupleVariation({'wght':(0,1,1)},[(0,0),(200,0),(100,0),(0,0),(200,0),(0,0),(0,0)])]};fb.font['gvar']=gvar;fb.save(path)
            original=path.read_bytes();face={'path':'variable.ttf','blob':blob(original),'axes':{'wght':{'min':100,'default':100,'max':900}}}
            family={'id':'fixture','selected':'variable.ttf','faces':[face],'trainingAxes':{'wght':400}}
            with patch.object(encoder_data,'DATA',root),patch.object(encoder_data,'CACHE',root):
                encoder_data.render_face(family);output=root/'faces/fixture.ttf';normal=output.read_bytes();time=output.stat().st_mtime_ns
                with TTFont(output) as font:
                    self.assertNotIn('fvar',font);self.assertEqual(font['OS/2'].usWeightClass,400)
                    normal_width=font['glyf']['A'].xMax
                encoder_data.render_face(family);self.assertEqual(output.stat().st_mtime_ns,time)
                output.write_bytes(b'corrupt');encoder_data.render_face(family);self.assertEqual(output.read_bytes(),normal)
                encoder_data.render_face({**family,'trainingAxes':{'wght':100}})
                with TTFont(output) as font:self.assertLess(font['glyf']['A'].xMax,normal_width)
                encoder_data.render_face(family);self.assertEqual(output.read_bytes(),normal);self.assertEqual(path.read_bytes(),original)

    def fixture(self):
        samples=[];windows=[]
        for family in ['a','b','c']:
            for text in ['first','second','third']:
                for renderer in ['pillow','chromium']:
                    windows.append({'source':len(samples)});samples.append({'family':family,'script':'Latn','role':'train','text':text,'renderer':renderer})
        split={'families':{'a':'train','b':'train','c':'train','unseen':'development'}}
        return {'samples':samples,'windows':windows},split

    def test_episode_changes_catalog_order_and_pairs_different_text_and_renderers(self):
        manifest,split=self.fixture();pools=training_pools(manifest,split);rng=np.random.default_rng(12);orders=set()
        for _ in range(12):
            indexes,labels,scripts=episode(pools,rng,{'a':['b']},count=2)
            orders.add(tuple(labels[::2]));self.assertEqual(len(indexes),4);self.assertEqual(len(set(labels)),2)
            for i in range(0,4,2):
                a,b=[manifest['samples'][manifest['windows'][j]['source']] for j in indexes[i:i+2]]
                self.assertEqual(a['family'],b['family']);self.assertNotEqual(a['text'],b['text']);self.assertNotEqual(a['renderer'],b['renderer'])
            self.assertEqual([manifest['samples'][manifest['windows'][i]['source']]['text'] for i in indexes[:2]],
                             [manifest['samples'][manifest['windows'][i]['source']]['text'] for i in indexes[2:]])
            self.assertEqual(scripts,['Latn']*4)
        self.assertGreater(len(orders),2)
        bad=copy.deepcopy(manifest);bad['samples'][0]['family']='unseen'
        self.assertRaisesRegex(ValueError,'Held-out',training_pools,bad,split)
        self.assertRaises(ValueError,episode,{'a':pools['a']},rng)

    def test_episodic_loss_semantics_gradients_and_alias_soft_targets(self):
        target=alias_targets(['a','b'],['Latn']*2,['a','b'],{})
        good=torch.tensor([[1.,0],[1.,0],[0,1.],[0,1.]],requires_grad=True)
        bad=good[[0,3,2,1]]
        self.assertLess(episodic_loss(good,target),episodic_loss(bad,target))
        episodic_loss(bad,target).backward();self.assertTrue(torch.isfinite(good.grad).all());self.assertGreater(good.grad.abs().sum(),0)
        soft=alias_targets(['a','b'],['Latn']*2,['a','b'],{('a','Latn'):['a','b'],('b','Latn'):['a','b']})
        torch.testing.assert_close(soft,torch.full((2,2),.5))
        for values in [torch.empty(0,2),good[:1],good[:3]]:self.assertRaises(ValueError,episodic_loss,values,target)
        self.assertRaises(ValueError,alias_targets,['a'],['Latn'],['a','a'],{})
        self.assertRaises(ValueError,alias_targets,['a'],['Latn'],['b'],{})

    def test_catalog_add_remove_reorder_reuses_exact_query_vectors(self):
        queries=unit([[3,0],[0,5]]);original=queries.copy();refs=unit([[2,0],[0,4],[-2,-2]])
        def names(matrix,families):return [families[i] for i in matrix.argmax(1)]
        self.assertEqual(names(rank(queries,refs[:2],np.array([0,1]),['a','b']),['a','b']),['a','b'])
        self.assertEqual(names(rank(queries,refs,np.array([0,1,2]),['a','b','new']),['a','b','new']),['a','b'])
        self.assertEqual(names(rank(queries,refs[[2,1,0]],np.array([0,1,2]),['new','b','a']),['new','b','a']),['a','b'])
        self.assertEqual(names(rank(queries,refs[1:2],np.array([0]),['b']),['b']),['b','b'])
        np.testing.assert_array_equal(queries,original)
        for vectors in [[],[[0,0]],[[float('nan'),1]]]:self.assertRaises(ValueError,unit,vectors)
        self.assertRaises(ValueError,rank,queries,refs,np.array([0,1,2]),['a','b'])

    def test_reference_quantization_and_macro_family_metrics(self):
        samples=[{'family':f,'text':text,'length':length} for f in ['a','b'] for text,length in [('ab','2-3'),('abcd','4-7'),('abcdefgh','8+'),('abcdefghijkl','8+')]]
        vectors=unit([[1,0]]*4+[[0,1]]*4)
        for count in [1,4]:
            refs,owners,packed,scales=references(vectors,samples,['a','b'],count,True)
            self.assertEqual(len(refs),2*count);self.assertEqual(packed.dtype,np.int8)
            np.testing.assert_allclose(refs,unit(packed.astype(np.float32)*scales),atol=1e-6)
        qs=[{'family':f,'split':'test','renderer':'chromium','condition':'clean','script':'Latn','length':'4-7'} for f in ['a','a','b']]
        result=metrics(np.array([[1,0],[1,0],[1,0]]),qs,['a','b'])
        self.assertEqual(result['groups']['all']['top1'],2/3);self.assertEqual(result['groups']['split/test']['macroTop1'],.5)

    def test_encoder_updates_and_exports_without_catalog_labels(self):
        torch.manual_seed(7);model=Classifier(128,context=True,wide=True,dilations=[1,1,2,2,1])
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);before=model.head.weight.detach().clone()
        pixels=torch.rand(4,1,16,24);target=torch.eye(2)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);loss=episodic_loss(model(pixels),target);loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertFalse(torch.equal(before,model.head.weight))
        artifact,restored=export(model,[str(i) for i in range(128)],{'width':128,'height':48,'windows':3})
        del artifact['fonts'];artifact.update(kind='font-encoder',dimensions=128,normalization='l2')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'encoder.json';save(path,artifact);loaded=load_encoder(path)
            with torch.no_grad():
                for height,width in [(1,1),(3,7),(16,24),(1,1)]:
                    x=torch.zeros(1,1,height,width)
                    torch.testing.assert_close(loaded(x),restored(x),rtol=0,atol=0)
                    self.assertEqual(unit(loaded(x).numpy()).shape,(1,128))
            for key,value in [('kind','classifier'),('dimensions',64),('fonts',['new-font'])]:
                save(path,{**artifact,key:value});self.assertRaises(ValueError,load_encoder,path)
            save(path,artifact);self.assertEqual(load_encoder(path).head.out_features,128)

    def test_embed_sizes_its_sums_by_the_model_not_a_constant(self):
        torch.manual_seed(7);model=Classifier(64,context=True,wide=True,dilations=[1,1,2,2,1]).eval()
        windows=[{'source':s,'offset':i*12,'width':4,'height':3} for i,s in enumerate(['a','a','b'])];pixels=np.arange(36,dtype=np.uint8)
        vectors=encoder.embed(model,{'windows':windows},pixels,['a','b'])
        self.assertEqual(vectors.shape,(2,64));np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,rtol=1e-5)
        self.assertRaises(ValueError,encoder.embed,model,{'windows':windows},pixels,['a','b','c'])  # c has no window


if __name__=='__main__':unittest.main()
