import base64
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from scripts.corpus import ROOT
from train.encoder import load_encoder
from train.encoder_data import save
from train.ten_model import Classifier, widen, export, load_export, pack_bits, unpack_bits, cluster_values, clustered, clusters_of, plain_state, export_clusters, LARGE_ARCH, WIDER_ARCH, WIDEST_ARCH, STUDENT_ARCH, CORPUS_ARCH


class CapacityTests(unittest.TestCase):
    def test_widening_preserves_projection_for_small_boundary_and_repeated_inputs(self):
        torch.manual_seed(11);torch.set_num_threads(2)
        model=Classifier(128,context=True,wide=True,dilations=[1,1,2,2,1]).eval();large=widen(model)
        self.assertEqual(large.architecture,LARGE_ARCH);self.assertGreater(sum(p.numel() for p in large.parameters()),sum(p.numel() for p in model.parameters()))
        a=torch.rand(1,1,1,1);b=torch.rand(1,1,48,128)
        with torch.no_grad():
            for x in [a,a,b,a]:torch.testing.assert_close(large(x),model(x),atol=2e-6,rtol=2e-6)
            mixed=torch.ones(2,1,48,128);mixed[0]=b[0];mixed[1,:,:1,:1]=a[0]
            sizes=torch.tensor([[48,128],[1,1]])
            torch.testing.assert_close(large(mixed,sizes),model(mixed,sizes),atol=2e-6,rtol=2e-6)
        artifact,restored=export(large,[str(i) for i in range(128)],{'width':128,'height':48,'windows':3})
        loaded=load_export(artifact)
        with torch.no_grad():torch.testing.assert_close(loaded(b),restored(b),atol=0,rtol=0)
        for bad in [-1,np.nan,np.inf]:self.assertRaises(ValueError,widen,model,bad)
        self.assertRaises(ValueError,widen,large);self.assertRaises(ValueError,widen,Classifier(2))

    def test_uneven_widening_keeps_the_function_and_the_artifact(self):
        # 1.5x: half the channels appear twice, half once; each input is divided by its own count.
        torch.manual_seed(13);torch.set_num_threads(2)
        large=widen(Classifier(128,context=True,wide=True,dilations=[1,1,2,2,1]),noise=.01).eval();wider=widen(large,architecture=WIDER_ARCH).eval()
        self.assertEqual(wider.architecture,WIDER_ARCH);self.assertEqual([c.out_channels for c in wider.convs],[96,192,288,384,384])
        x=torch.rand(3,1,48,128);sizes=torch.tensor([[48,128],[30,70],[1,1]])
        with torch.no_grad():torch.testing.assert_close(wider(x,sizes),large(x,sizes),atol=3e-6,rtol=3e-6)
        artifact,restored=export(wider,[str(i) for i in range(128)],{'width':128,'height':48,'windows':3})
        self.assertEqual(artifact['architecture'],WIDER_ARCH)
        with torch.no_grad():torch.testing.assert_close(load_export(artifact)(x),restored(x),atol=0,rtol=0)
        # 4/3: a third of the channels appear twice.
        widest=widen(wider,architecture=WIDEST_ARCH).eval();self.assertEqual([c.out_channels for c in widest.convs],[128,256,384,512,512])
        with torch.no_grad():torch.testing.assert_close(widest(x,sizes),large(x,sizes),atol=3e-6,rtol=3e-6)
        artifact,restored=export(widest,[str(i) for i in range(128)],{'width':128,'height':48,'windows':3})
        with torch.no_grad():torch.testing.assert_close(load_export(artifact)(x),restored(x),atol=0,rtol=0)
        for bad in [CORPUS_ARCH,LARGE_ARCH,'unknown']:self.assertRaises(ValueError,widen,wider,0,bad)  # never narrower, never the same, never unknown
        self.assertRaises(ValueError,Classifier,2,architecture='unknown')
        self.assertRaises(ValueError,Classifier,2,context=True,wide=True,large=True,architecture=CORPUS_ARCH)  # conflicting descriptions fail
        self.assertEqual(Classifier(2,context=True,wide=True,large=True).architecture,Classifier(2,architecture=LARGE_ARCH).architecture)

    def test_bit_streams_match_the_browser_and_six_bit_exports_restore_exactly(self):
        # The streams tests/catalog.test.mjs decodes: nine values at each width's extremes, the last byte partial.
        for bits,stream in [(4,'eRBvOg0='),(6,'4QcEvycOPQ=='),(8,'gX8AAf9+ggP9')]:
            top=2**(bits-1)-1;values=[-top,top,0,1,-1,top-1,1-top,3,-3]
            self.assertEqual(base64.b64encode(pack_bits(values,bits)).decode(),stream);self.assertEqual(unpack_bits(base64.b64decode(stream),bits,9).tolist(),values)
        self.assertRaises(ValueError,unpack_bits,base64.b64decode('4QcEvycOPQ=='),6,8)  # a byte too many for eight values
        torch.manual_seed(14);model=Classifier(128,context=True,wide=True,dilations=[1,1,2,2,1]).eval();x=torch.rand(2,1,48,128)
        eight,_=export(model,[str(i) for i in range(128)],{'width':128,'height':48,'windows':3})
        self.assertNotIn('bits',eight['layers'][0])  # 8-bit artifacts keep the format every earlier model shipped in
        six,restored=export(model,[str(i) for i in range(128)],{'width':128,'height':48,'windows':3},bits=6)
        self.assertEqual({l['bits'] for l in six['layers']},{6})
        self.assertEqual(len(base64.b64decode(six['layers'][1]['weights'])),64*32*9*6//8)
        with torch.no_grad():torch.testing.assert_close(load_export(six)(x),restored(x),atol=0,rtol=0)
        # As train.style.load_run(exported=True) reads it: an encoder file names its dimensions instead of fonts.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'encoder.json';save(path,{**{k:v for k,v in six.items() if k!='fonts'},'kind':'font-encoder','dimensions':128,'normalization':'l2'})
            with torch.no_grad():torch.testing.assert_close(load_encoder(path)(x),restored(x),atol=0,rtol=0)

    def test_a_separable_student_infers_alike_in_pytorch_and_the_browser_code(self):
        import json,subprocess,tempfile
        from pathlib import Path
        torch.manual_seed(15);model=Classifier(64,architecture=STUDENT_ARCH,dilations=[1,1,2,2,1]).eval()
        self.assertEqual([c.groups for c in model.convs],[1,96,1,192,1,288,1,384,1]);self.assertLess(sum(p.numel() for p in model.parameters()),400000)
        artifact,restored=export(model,[str(i) for i in range(64)],{'width':128,'height':48,'windows':3},bits=6)
        self.assertEqual([l['shape'] for l in artifact['layers']][:3],[[96,1,3,3],[96,1,3,3],[192,96,1,1]])
        pixels=torch.rand(1,1,30,77)
        with torch.no_grad():expected=restored(pixels)[0].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'model.json').write_text(json.dumps({**{k:v for k,v in artifact.items() if k!='fonts'},'kind':'font-encoder','dimensions':64,'normalization':'l2'}))
            Path(tmp,'window.json').write_text(json.dumps({'width':77,'height':30,'pixels':pixels.flatten().tolist()}))
            out=subprocess.run(['node','--input-type=module','-e',f"""
import {{ readFileSync }} from 'node:fs'; import {{ readNetwork, inferCPU }} from '{ROOT}/src/network.mjs'
const w = JSON.parse(readFileSync('{tmp}/window.json')), model = readNetwork(JSON.parse(readFileSync('{tmp}/model.json')))
console.log(JSON.stringify(Array.from(inferCPU(model, {{ width: w.width, height: w.height, pixels: Float32Array.from(w.pixels) }}))))"""],capture_output=True,text=True,check=True)
        ours=json.loads(out.stdout);scale=max(abs(v) for v in expected)
        self.assertLess(max(abs(a-b) for a,b in zip(ours,expected)),1e-4*max(scale,1))
        self.assertRaises(ValueError,widen,model,0,WIDER_ARCH)
        # Clustered 4-bit weights: 16 shared values per layer, indices in the file, exact on reload and in JavaScript.
        book,index=cluster_values(np.array([[-1,-.9,.2,.25,1],[0,.1,-.5,.5,.95]]),2)
        self.assertEqual(book.shape,(4,));self.assertTrue((np.diff(book)>=0).all());self.assertTrue(((index>=0)&(index<4)).all());self.assertEqual(index.shape,(2,5))
        clustered,restored=export(model,[str(i) for i in range(64)],{'width':128,'height':48,'windows':3},bits=4,codebook=True)
        self.assertEqual(len(clustered['layers'][1]['codebook']),16);self.assertEqual(len(base64.b64decode(clustered['layers'][1]['weights'])),96*9*4//8)
        with torch.no_grad():torch.testing.assert_close(load_export(clustered)(pixels),restored(pixels),atol=0,rtol=0)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'model.json').write_text(json.dumps({**{k:v for k,v in clustered.items() if k!='fonts'},'kind':'font-encoder','dimensions':64,'normalization':'l2'}))
            Path(tmp,'window.json').write_text(json.dumps({'width':77,'height':30,'pixels':pixels.flatten().tolist()}))
            out=subprocess.run(['node','--input-type=module','-e',f"""
import {{ readFileSync }} from 'node:fs'; import {{ readNetwork, inferCPU }} from '{ROOT}/src/network.mjs'
const w = JSON.parse(readFileSync('{tmp}/window.json')), model = readNetwork(JSON.parse(readFileSync('{tmp}/model.json')))
console.log(JSON.stringify(Array.from(inferCPU(model, {{ width: w.width, height: w.height, pixels: Float32Array.from(w.pixels) }}))))"""],capture_output=True,text=True,check=True)
            with torch.no_grad():expected=restored(pixels)[0].tolist()
            self.assertLess(max(abs(a-b) for a,b in zip(json.loads(out.stdout),expected)),1e-4*max(max(abs(v) for v in expected),1))


        torch.manual_seed(12);model=Classifier(128,context=True,wide=True)
        large=widen(model,noise=.0001)
        self.assertFalse(torch.equal(large.convs[0].weight[:32],large.convs[0].weight[32:]))
        output=large(torch.rand(4,1,24,50));loss=output.square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(large.convs[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in large.parameters() if p.grad is not None))

    def test_a_clustered_model_trains_only_its_codebooks_scales_and_biases_and_exports_exactly(self):
        torch.manual_seed(4);model=Classifier(64,architecture=STUDENT_ARCH,dilations=[1,1,2,2,1]);q=clustered(model,4).eval();x=torch.rand(2,1,40,90)
        names=[n for n,p in q.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.endswith(('.bias','.codebook','.scale')) for n in names));self.assertLess(sum(p.numel() for p in q.parameters() if p.requires_grad),5000)
        # The materialized state is an ordinary folded checkpoint; the artifact from the clusters reproduces it bit for bit.
        state=plain_state(q);plain=Classifier(64,training=False,architecture=STUDENT_ARCH,dilations=[1,1,2,2,1]).eval();plain.load_state_dict(state)
        with torch.no_grad():torch.testing.assert_close(plain(x),q(x),atol=1e-5,rtol=1e-5)
        biases=[state[f'{n}.bias'] for n in [*(f'convs.{i}' for i in range(len(q.convs))),'head']]
        artifact,restored=export_clusters(clusters_of(q),biases,q.architecture,q.dilations,[str(i) for i in range(64)],{'width':128,'height':48,'windows':3})
        self.assertEqual([len(l['codebook']) for l in artifact['layers']],[16]*10)
        with torch.no_grad():torch.testing.assert_close(load_export(artifact)(x),restored(x),atol=0,rtol=0);torch.testing.assert_close(restored(x),q(x),atol=1e-5,rtol=1e-5)
        # One training step moves the codebooks, never the indices.
        before=q.convs[1].parametrizations.weight[0].index.clone();opt=torch.optim.SGD([p for p in q.parameters() if p.requires_grad],lr=.1)
        q.train();q(x).sum().backward();opt.step()
        self.assertTrue(torch.equal(q.convs[1].parametrizations.weight[0].index,before));self.assertFalse(torch.allclose(q.eval()(x),restored(x)))

    def test_duplicated_features_can_learn_different_filters(self):
        torch.manual_seed(12);model=Classifier(128,context=True,wide=True)
        large=widen(model,noise=.0001)
        self.assertFalse(torch.equal(large.convs[0].weight[:32],large.convs[0].weight[32:]))
        output=large(torch.rand(4,1,24,50));loss=output.square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(large.convs[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in large.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
