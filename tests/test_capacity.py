import unittest
import numpy as np
import torch

from train.ten_model import Classifier, widen, export, load_export, LARGE_ARCH, WIDER_ARCH, CORPUS_ARCH


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
        for bad in [CORPUS_ARCH,LARGE_ARCH,'unknown']:self.assertRaises(ValueError,widen,wider,0,bad)  # never narrower, never the same, never unknown
        self.assertRaises(ValueError,Classifier,2,architecture='unknown')
        self.assertRaises(ValueError,Classifier,2,context=True,wide=True,large=True,architecture=CORPUS_ARCH)  # conflicting descriptions fail
        self.assertEqual(Classifier(2,context=True,wide=True,large=True).architecture,Classifier(2,architecture=LARGE_ARCH).architecture)

    def test_duplicated_features_can_learn_different_filters(self):
        torch.manual_seed(12);model=Classifier(128,context=True,wide=True)
        large=widen(model,noise=.0001)
        self.assertFalse(torch.equal(large.convs[0].weight[:32],large.convs[0].weight[32:]))
        output=large(torch.rand(4,1,24,50));loss=output.square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(large.convs[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in large.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
