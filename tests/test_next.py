"""Source aggregation, face marginals and frozen feature retrieval regressions."""
import unittest
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import torch
from train import next as continuation
from train import faces
from train.retrieval import unit, prototypes, metrics, features
from train.ten import batch
from train.ten_model import Classifier
from torch.nn.utils import parametrize
from train.robustness import sha
from train.metric import paired_pools, paired_batch


class NextTests(unittest.TestCase):
    def test_face_manifest_rejects_wrong_attributes_leakage_and_final_byte_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'.data/faces';data.mkdir(parents=True);(root/'bench').mkdir()
            face={'id':'a-400-normal','family':'a','weight':400,'style':'normal'}
            (root/'bench/font-faces.json').write_text(json.dumps({'faces':[face]}))
            samples=[{'family':face['id'],'fontFamily':'a','weight':400,'style':'normal','role':r,'text':t} for r,t in [('train','aa'),('validation','bb'),('test','cc')]]
            manifest={'samples':samples,'windows':[{'source':i,'offset':i,'width':1,'height':1} for i in range(3)]}
            for key,name in [('facesSha256','bench/font-faces.json'),('textsSha256','bench/face-texts.json'),('tensorSha256','.data/faces/prepared.u8'),('inputSha256','src/input.mjs'),('normalizerSha256','src/prepare.mjs'),('generatorSha256','train/faces_data.py'),('runnerSha256','scripts/faces.mjs')]:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():path.write_bytes(bytes([0,128,255]) if key=='tensorSha256' else b'fixture')
                manifest[key]=sha(path)
            def save(m):(data/'prepared.json').write_text(json.dumps(m))
            with patch.object(faces,'ROOT',root),patch.object(faces,'DATA',data):
                save(manifest);_,_,pixels=faces.load_data();np.testing.assert_array_equal(pixels,[0,128,255]);del pixels
                mutations=[lambda m:m['samples'][0].update(weight=700),lambda m:m['samples'][0].update(style='italic'),lambda m:m['samples'][0].update(fontFamily='b'),lambda m:m['samples'][1].update(text='AA'),lambda m:m['samples'][2].update(role='train'),lambda m:m['windows'].__setitem__(0,None),lambda m:m['windows'][0].update(width=True),lambda m:m['windows'][1].update(offset=0),lambda m:m['windows'][-1].update(source=3),lambda m:m['windows'].pop(),lambda m:m.update(inputSha256='stale')]
                for mutate in mutations:
                    value=copy.deepcopy(manifest);mutate(value);save(value)
                    with self.assertRaises(ValueError):faces.load_data()
                for payload in [b'',bytes(2),bytes(4)]:
                    (data/'prepared.u8').write_bytes(payload);value=copy.deepcopy(manifest);value['tensorSha256']=sha(data/'prepared.u8');save(value)
                    with self.assertRaises(ValueError):faces.load_data()

    def test_metric_pairs_change_content_and_keep_unseen_fonts_out(self):
        samples=[{'family':f,'text':t,'renderer':r,'role':'train'} for f in ['a','b'] for t in ['aa','bb'] for r in ['pillow','chromium']]
        manifest={'samples':samples,'windows':[{'source':i} for i in range(len(samples))]}
        pools,texts=paired_pools(manifest,['a','b']);rng=np.random.default_rng(14)
        for _ in range(5):
            chosen,labels=paired_batch(pools,texts,['a','b'],rng,families=2)
            for i in [0,2]:
                a,b=[samples[manifest['windows'][j]['source']] for j in chosen[i:i+2]]
                self.assertEqual(a['family'],b['family']);self.assertNotEqual(a['text'],b['text']);self.assertEqual(labels[i],labels[i+1])
            self.assertNotEqual(labels[0],labels[2])
        bad=copy.deepcopy(manifest);bad['samples'][0]['family']='unknown'
        with self.assertRaisesRegex(ValueError,'Unseen'):paired_pools(bad,['a','b'])
        bad=copy.deepcopy(manifest);bad['windows'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete'):paired_pools(bad,['a','b'])

    def test_qat_snapshot_export_does_not_break_subsequent_training(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        model=Classifier(2,training=False).train()
        for layer in [*model.convs,model.head]:parametrize.register_parametrization(layer,'weight',faces.Int8Weights())
        from train.ten_model import export
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        x=torch.linspace(0,1,35).reshape(1,1,5,7)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output=model(x);loss=output.square().sum();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            snapshot=faces.qat_snapshot(model);artifact,restored=export(snapshot,['a','b'],{})
            torch.testing.assert_close(output.detach(),restored(x),atol=1e-7,rtol=1e-6)
            self.assertTrue(parametrize.is_parametrized(model.convs[0],'weight'))
            optimizer.step()
        weight=torch.tensor([[0.,0.],[.123,.456]],requires_grad=True)
        quant=faces.Int8Weights()(weight);self.assertTrue(torch.isfinite(quant).all())
        quant.sum().backward();torch.testing.assert_close(weight.grad,torch.ones_like(weight))

    def test_invalid_training_budget_fails_before_data_access(self):
        for trainer,args in [(continuation.train,('current',)),(faces.train,())]:
            for steps in [0,-1,False,1.5]:
                with self.subTest(trainer=trainer.__module__,steps=steps), patch.object(continuation,'read') as a, patch.object(faces,'load_data') as b:
                    with self.assertRaises(ValueError):trainer(*args,steps=steps)
                    a.assert_not_called();b.assert_not_called()
        with self.assertRaises(ValueError):continuation.train('missing',steps=1)

    def test_prediction_role_mapping_and_a_a_b_a_semantics(self):
        torch.set_num_threads(1);torch.manual_seed(37)
        model=Classifier(2).eval()
        samples=[{'role':'train','name':'a'},{'role':'validation','name':'b'},{'role':'train','name':'c'}]
        windows=[{'source':s,'offset':i*9,'width':3,'height':3} for i,s in enumerate([0,1,0,2])]
        pixels=np.arange(36,dtype=np.uint8)
        rows,logits,owners=continuation.predict(model,pixels,windows,samples,'train')
        self.assertEqual(rows,[samples[0],samples[2]]);np.testing.assert_array_equal(owners,[0,0,1])
        for role in ['train','validation','train']:
            a=continuation.predict(model,pixels,windows,samples,role)
            ids=[i for i,w in enumerate(windows) if samples[w['source']]['role']==role]
            with torch.no_grad():expected=model(*batch(pixels,windows,ids)).numpy()
            np.testing.assert_allclose(a[1],expected,rtol=0,atol=0)
        fake=Mock()
        with self.assertRaisesRegex(ValueError,'No prediction windows'):continuation.predict(fake,pixels,windows,samples,'absent')
        fake.eval.assert_not_called()

    def test_face_marginals_do_not_inherit_the_top_face_family(self):
        catalog=[{'id':f'{f}-{w}-{s}','family':f,'weight':w,'style':s} for f in ['a','b'] for w in [400,700] for s in ['normal','italic']]
        sample={'family':catalog[0]['id'],'length':'4-7','condition':'clean','renderer':'test'}
        p=np.array([[.18,.16,.12,.10,.30,.08,.04,.02]])
        report=faces.metrics(catalog,[sample],p)['groups']['all']
        self.assertEqual(report['faceAccuracy'],0)
        for key in ['family','weight','style','knownFamilyWeight','knownFamilyStyle']:self.assertEqual(report[key+'Accuracy'],1)
        for bad in [p[:,:-1],p*2,p*np.nan,-p]:
            with self.assertRaises(ValueError):faces.metrics(catalog,[sample],bad)

    def test_int8_prototypes_and_deterministic_ties_preserve_identity(self):
        vectors=unit([[3,4,0],[6,8,0],[0,0,5],[0,0,10]])
        samples=[{'family':f,'length':'1','condition':'clean'} for f in ['a','a','b','b']]
        refs,owners,size=prototypes(vectors,samples,['a','b'])
        self.assertEqual(size,2*(3+4));np.testing.assert_array_equal(owners,[0,1])
        np.testing.assert_allclose(np.linalg.norm(refs,axis=1),1,atol=1e-7)
        report=metrics(vectors,samples,['a','b'],refs,owners)
        self.assertEqual(report['all']['accuracy'],1)
        tie=metrics(np.zeros((1,3)),[samples[0]],['a','b'],refs,owners)
        self.assertEqual(tie['all']['accuracy'],1)
        with self.assertRaisesRegex(ValueError,'Missing prototype pool'):prototypes(vectors,samples,['a','b'],True)
        np.testing.assert_array_equal(unit([[0,0,0]]),[[0,0,0]])
        with self.assertRaises(ValueError):unit([[np.nan,0]])

    def test_frozen_features_equal_classifier_input_and_average_per_source(self):
        torch.set_num_threads(1);torch.manual_seed(51)
        model=Classifier(2).eval();head=model.head;model.head=torch.nn.Identity()
        windows=[{'source':s,'offset':i*9,'width':3,'height':3} for i,s in enumerate([5,8,5])];pixels=np.arange(27,dtype=np.uint8)
        with torch.no_grad():per_window=model(*batch(pixels,windows,[0,1,2])).numpy()
        result=features(model,pixels,windows,[8,5]);normalized=unit(per_window)
        np.testing.assert_allclose(result,unit([normalized[1],normalized[[0,2]].mean(0)]),atol=1e-7)
        model.head=head
        with torch.no_grad():
            torch.testing.assert_close(model(*batch(pixels,windows,[0,1,2])),head(torch.from_numpy(per_window)),atol=0,rtol=0)
        for sources in [[],[5,5],[9]]:
            with self.assertRaises(ValueError):features(model,pixels,windows,sources)


if __name__=='__main__':unittest.main()
