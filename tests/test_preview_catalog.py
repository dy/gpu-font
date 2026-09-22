import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.preview_catalog import make_catalog
from train.encoder_catalog import read_catalog
from train.encoder_data import save


class PreviewCatalogTests(unittest.TestCase):
    def test_image_references_keep_regular_bold_italic_faces_and_exact_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=Path(tmp)/'encoder.json';save(encoder,{'preparation':{'width':128,'height':48,'windows':3}})
            records=[];vectors=[]
            for i,(style,weight,slant) in enumerate([('Regular',400,'upright'),('Bold',540,'upright'),('Italic',400,'italic')]):
                for recipe in ['words-a-v1','words-b-v1']:
                    records.append({'id':style+recipe,'recipeId':recipe,'faceId':style,'familyId':'a','family':'A','styleName':style,'weight':weight,'slant':slant,'axes':{},'foundry':'Foundry','sourceUrl':'https://example.test/a','scripts':['Latn']})
                    v=np.zeros(128,dtype=np.float32);v[i]=1;vectors.append(v)
            vectors=np.array(vectors)
            a=make_catalog(records,vectors,encoder,'manifest','words')
            decoded,owners,labels=read_catalog(a,encoder)
            self.assertEqual(a['version'],2);self.assertEqual(a['sourceManifestSha256'],'manifest')
            self.assertEqual(labels,['Bold','Italic','Regular']);self.assertEqual(owners.tolist(),[0,1,2])
            np.testing.assert_array_equal(decoded[:,:3],np.eye(3)[[1,2,0]])
            self.assertEqual(a['faces'][0]['weight'],540);self.assertEqual(a['faces'][1]['style'],'italic')
            self.assertEqual(a['faces'][0]['referenceIds'],['Boldwords-a-v1','Boldwords-b-v1'])
            for _ in range(2):self.assertEqual(make_catalog(records,vectors,encoder,'manifest','words'),a)
            # A different set indexes with the same frozen encoder and no font paths.
            b=make_catalog(records[:2],vectors[:2],encoder,'other','words')
            self.assertEqual(b['encoderSha256'],a['encoderSha256']);self.assertEqual(len(b['faces']),1)
            self.assertEqual(make_catalog(records,vectors,encoder,'manifest','words'),a)
            self.assertRaisesRegex(ValueError,'unique families',read_catalog,{**a,'version':1},encoder)
            bad=copy.deepcopy(a);bad['faces'][1]['family']='Other'
            self.assertRaisesRegex(ValueError,'Conflicting',read_catalog,bad,encoder)
            self.assertRaisesRegex(ValueError,'Incomplete',make_catalog,records[:-1],vectors[:-1],encoder,'manifest','words')
            bad=copy.deepcopy(records);bad[1]['weight']=700
            self.assertRaisesRegex(ValueError,'Conflicting',make_catalog,bad,vectors,encoder,'manifest','words')
            for recipe in ['alphabet','unknown']:
                self.assertRaises(ValueError,make_catalog,records,vectors,encoder,'manifest',recipe)
            self.assertRaises(ValueError,make_catalog,[],np.empty((0,128)),encoder,'manifest','words')
            self.assertRaises(ValueError,make_catalog,records,vectors[:-1],encoder,'manifest','words')
            self.assertRaises(ValueError,make_catalog,records,np.zeros_like(vectors),encoder,'manifest','words')


if __name__=='__main__':unittest.main()
