import base64
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from train.encoder import load_encoder
from train.encoder_data import save
from train.ten_model import Classifier, widen, export, load_export, pack_bits, unpack_bits, LARGE_ARCH, WIDER_ARCH, WIDEST_ARCH, CORPUS_ARCH


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

    def test_duplicated_features_can_learn_different_filters(self):
        torch.manual_seed(12);model=Classifier(128,context=True,wide=True)
        large=widen(model,noise=.0001)
        self.assertFalse(torch.equal(large.convs[0].weight[:32],large.convs[0].weight[32:]))
        output=large(torch.rand(4,1,24,50));loss=output.square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(large.convs[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in large.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
