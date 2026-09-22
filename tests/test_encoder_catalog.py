import base64
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from train.encoder_catalog import read_catalog,preparation_hash,calibrate,rejection,absent_catalog,confidence_intervals
from train.encoder_data import save
from train.robustness import sha


class EncoderCatalogTests(unittest.TestCase):
    def test_catalog_binding_exact_packed_boundaries_and_repeated_decodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'encoder.json';preparation={'width':128,'height':48,'windows':3};save(path,{'preparation':preparation})
            raw=bytes([127]+[0]*127+[0,127]+[0]*126)
            catalog={'version':1,'kind':'font-catalog','dimensions':128,'referencesPerFace':1,'encoderSha256':sha(path),'preparationSha256':preparation_hash(preparation),
                     'faces':[{'id':f,'familyId':f,'family':f} for f in ['a','b']],
                     'vectors':{'encoding':'int8-base64','shape':[2,128],'data':base64.b64encode(raw).decode(),'scales':[1/127]*2,'owners':[0,1]}}
            for _ in range(2):
                vectors,owners,families=read_catalog(catalog,path);np.testing.assert_array_equal(vectors[:,:2],np.eye(2));np.testing.assert_array_equal(vectors[:,2:],0)
                self.assertEqual(families,['a','b']);self.assertEqual(owners.tolist(),[0,1])
            smallest=copy.deepcopy(catalog);smallest['faces']=smallest['faces'][:1]
            smallest['vectors'].update(shape=[1,128],data=base64.b64encode(raw[:128]).decode(),scales=[1/127],owners=[0])
            self.assertEqual(read_catalog(smallest,path)[0][0].tolist(),[1.]+[0.]*127)
            for value in [None,{},[],{'faces':None}]:self.assertRaises(ValueError,read_catalog,value,path)
            for content in [b'',raw[:-1],raw+b'\0']:
                bad=copy.deepcopy(catalog);bad['vectors']['data']=base64.b64encode(content).decode();self.assertRaisesRegex(ValueError,'vectors',read_catalog,bad,path)
            for key,value in [('encoderSha256','wrong'),('preparationSha256','wrong'),('dimensions',64),('dimensions',128.0),('version',True),('faces',[]),('referencesPerFace',2),('referencesPerFace',True)]:
                self.assertRaises(ValueError,read_catalog,{**catalog,key:value},path)
            for key,value in [('data','???'),('shape',[1,128]),('scales',[1,float('nan')]),('scales',[0,1]),('owners',[True,1]),('owners',[0,0])]:
                bad=copy.deepcopy(catalog);bad['vectors'][key]=value;self.assertRaises(ValueError,read_catalog,bad,path)
            bad=copy.deepcopy(catalog);bad['faces'][1]['id']='a';self.assertRaises(ValueError,read_catalog,bad,path)
            bad=copy.deepcopy(catalog);bad['faces'][1]['familyId']='a';self.assertRaises(ValueError,read_catalog,bad,path)
            other=copy.deepcopy(catalog);other['faces'][1]={'id':'c','familyId':'c','family':'C'}
            self.assertEqual(read_catalog(other,path)[2],['a','c']);self.assertEqual(read_catalog(catalog,path)[2],['a','b'])
            save(path,{'preparation':{**preparation,'method':'other'}});self.assertRaisesRegex(ValueError,'Incompatible',read_catalog,catalog,path)

    def test_rejection_threshold_is_selected_only_at_complete_score_ties(self):
        scores=np.array([[.95,.1],[.92,.2],[.8,.93],[.8,.91],[.82,.7],[.65,.81]],dtype=np.float32)
        samples=[{'family':f} for f in ['a','a','b','b','unknown','unknown']]
        result=calibrate(scores,samples,['a','b'])
        self.assertEqual(result['acceptedPresent'],4);self.assertEqual(result['acceptedPresentAccuracy'],1);self.assertEqual(result['absentAcceptance'],0)
        self.assertEqual(result,rejection(scores,samples,['a','b'],result['threshold']))
        tied=calibrate(np.array([[.9],[.9]]),[{'family':'a'},{'family':'unknown'}],['a'])
        self.assertEqual(tied['acceptedPresent'],0);self.assertEqual(tied['absentAcceptance'],0)
        self.assertIsNone(tied['threshold'])
        # Normalized float32 dot products can round slightly above one. A missing
        # operating point must reject explicitly, not use a near-one sentinel.
        rounded=rejection(np.array([[1.0000003],[1.0000003]]),[{'family':'a'},{'family':'unknown'}],['a'],tied['threshold'])
        self.assertEqual(rounded['acceptedPresent'],0);self.assertEqual(rounded['absentAcceptance'],0)
        self.assertRaises(ValueError,calibrate,scores[:1],samples[:1],['a','b'])

    def test_absent_catalog_removes_complete_family_groups(self):
        split={'groups':[['a','b'],['c','d'],['known']], 'families':{'a':'development','b':'development','c':'development','d':'development','known':'train'},'counts':{'development':4}}
        kept,absent=absent_catalog(['a','b','c','d','known'],split,'development')
        self.assertEqual(len(absent),2);self.assertIn('known',kept);self.assertFalse(set(kept)&set(absent))
        self.assertEqual(set(kept)|set(absent),set(split['families']))
        for group in split['groups']:self.assertTrue(set(group)<=set(kept) or set(group)<=set(absent))

    def test_intervals_resample_lineages_and_weight_families_not_query_counts(self):
        split={'groups':[['a','b'],['c']], 'families':{'a':'test','b':'test','c':'development'}}
        report={'groups':{'family/a':{'count':1000,'top1':0.,'top5Accuracy':.5},
                          'family/b':{'count':1,'top1':1.,'top5Accuracy':1.}}}
        result=confidence_intervals(report,split)
        self.assertEqual(result,confidence_intervals(report,split))
        self.assertEqual(result['splits'],{'test':{'families':2,'lineageGroups':1,'macroTop1':[.5,.5],'macroTop5':[.75,.75]}})
        self.assertEqual(confidence_intervals({'groups':{}},split)['splits'],{})


if __name__=='__main__':unittest.main()
