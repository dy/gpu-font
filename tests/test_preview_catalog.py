import base64
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.preview_catalog import make_catalog, complete
from train.encoder_catalog import read_catalog
from train.encoder_data import save
from train.ten_model import pack_bits


class PreviewCatalogTests(unittest.TestCase):
    def test_compiled_rows_are_four_bit_and_read_as_their_values_times_their_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=Path(tmp)/'encoder.json';save(encoder,{'dimensions':128,'preparation':{'width':128,'height':48,'windows':3}})
            records=[{'id':f'r{i}{recipe}','recipeId':recipe,'faceId':f'f{i}','familyId':'a','family':'A','styleName':f'S{i}','weight':400,'slant':'upright','axes':{},'foundry':'F','sourceUrl':'https://example.test/a','scripts':['Latn']} for i in range(3) for recipe in ['words-a-v1','words-b-v1']]
            vectors=np.random.default_rng(3).normal(size=(6,128)).astype(np.float32);a=make_catalog(records,vectors,encoder,'manifest','words')
            self.assertEqual((a['vectors']['encoding'],a['dimensions'],a['vectors']['shape']),('int4-base64',128,[6,128]))
            unit=vectors/np.linalg.norm(vectors,axis=1,keepdims=True);scales=np.abs(unit).max(1)/7;four=np.round(unit/scales[:,None]).clip(-7,7).astype(np.int8)
            self.assertEqual(a['vectors']['data'],base64.b64encode(pack_bits(four,4)).decode());np.testing.assert_allclose(a['vectors']['scales'],scales,rtol=1e-6)
            decoded,owners,labels=read_catalog(a,encoder)
            # Each row is its 4-bit values times its scale, normalized; an 8-bit copy of the same values reads the same.
            np.testing.assert_allclose(decoded*np.linalg.norm(four*scales[:,None],axis=1,keepdims=True),four*scales[:,None],rtol=1e-5,atol=1e-7)
            b=copy.deepcopy(a);b['vectors'].update(encoding='int8-base64',data=base64.b64encode(four.tobytes()).decode())
            np.testing.assert_allclose(read_catalog(b,encoder)[0],decoded,atol=1e-7);self.assertEqual((owners.tolist(),labels),(lambda r:(r[1].tolist(),r[2]))(read_catalog(b,encoder)))
            for data,message in [(a['vectors']['data'][:-4],'Truncated'),(base64.b64encode(pack_bits(four,4)+b'\0').decode(),'Truncated')]:
                self.assertRaisesRegex(ValueError,message,read_catalog,{**a,'vectors':{**a['vectors'],'data':data}},encoder)
            self.assertRaisesRegex(ValueError,'shape',read_catalog,{**a,'vectors':{**a['vectors'],'encoding':'int5-base64'}},encoder)

    def test_a_face_carries_the_preview_file_its_records_name_and_they_must_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=Path(tmp)/'encoder.json';save(encoder,{'dimensions':128,'preparation':{'width':128,'height':48,'windows':3}})
            record=lambda i,**extra:{'id':f'r{i}','recipeId':'provided-v1','faceId':f'f{i}','familyId':f'a{i}','family':f'A{i}','styleName':None,'weight':None,'slant':None,'axes':{},'foundry':'F','sourceUrl':'https://example.test/a','scripts':['Latn'],**extra}
            faces=make_catalog([record(0,previewFile=3),record(1)],np.eye(2,128,dtype=np.float32),encoder,'manifest','captured')['faces']
            self.assertEqual(faces[0]['previewFile'],3);self.assertNotIn('previewFile',faces[1])  # absent means 0, the usual one
            clash=[{**record(0,previewFile=3),'recipeId':'words-a-v1'},{**record(0),'id':'r0b','recipeId':'words-b-v1'}]
            self.assertRaisesRegex(ValueError,'previewFile',make_catalog,clash,np.eye(2,128,dtype=np.float32),encoder,'manifest','captured')

    def test_image_references_keep_regular_bold_italic_faces_and_exact_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=Path(tmp)/'encoder.json';save(encoder,{'dimensions':128,'preparation':{'width':128,'height':48,'windows':3}})
            records=[];vectors=[]
            for i,(style,weight,slant) in enumerate([('Regular',400,'upright'),('Bold',540,'upright'),('Italic',400,'italic')]):
                for recipe in ['words-a-v1','words-b-v1']:
                    records.append({'id':style+recipe,'recipeId':recipe,'faceId':style,'familyId':'a','family':'A','styleName':style,'weight':weight,'slant':slant,'axes':{},'foundry':'Foundry','sourceUrl':'https://example.test/a','scripts':['Latn']})
                    v=np.zeros(128,dtype=np.float32);v[i]=1;vectors.append(v)
            vectors=np.array(vectors)
            a=make_catalog(records,vectors,encoder,'manifest','words')
            decoded,owners,labels=read_catalog(a,encoder)
            self.assertEqual(a['version'],3);self.assertEqual(a['sourceManifestSha256'],'manifest')
            # One row per capture, owned by its face: a face scores by its best capture, never by an average.
            self.assertEqual(labels,['Bold','Italic','Regular']);self.assertEqual(owners.tolist(),[0,0,1,1,2,2])
            np.testing.assert_array_equal(decoded[:,:3],np.eye(3)[[1,1,2,2,0,0]])
            self.assertEqual(a['faces'][0]['weight'],540);self.assertEqual(a['faces'][1]['style'],'italic')
            self.assertEqual(a['faces'][0]['referenceIds'],['Boldwords-a-v1','Boldwords-b-v1'])
            for _ in range(2):self.assertEqual(make_catalog(records,vectors,encoder,'manifest','words'),a)
            # A different set indexes with the same frozen encoder and no font paths.
            b=make_catalog(records[:2],vectors[:2],encoder,'other','words')
            self.assertEqual(b['encoderSha256'],a['encoderSha256']);self.assertEqual(len(b['faces']),1)
            self.assertEqual(make_catalog(records,vectors,encoder,'manifest','words'),a)
            self.assertRaisesRegex(ValueError,'unique families',read_catalog,{**a,'version':1,'referencesPerFace':1},encoder)
            bad=copy.deepcopy(a);bad['faces'][1]['family']='Other'
            self.assertRaisesRegex(ValueError,'Conflicting',read_catalog,bad,encoder)
            self.assertRaisesRegex(ValueError,'Incomplete',make_catalog,records[:-1],vectors[:-1],encoder,'manifest','words')
            # A live archive may hold a face whose captures are unfinished: index the complete faces, report the rest.
            ids,skipped=complete(records[:-1],'words')
            self.assertEqual(skipped,['Italic']);self.assertEqual(ids,[0,1,2,3])
            self.assertEqual(make_catalog([records[i] for i in ids],vectors[ids],encoder,'manifest','words')['faces'],[f for f in a['faces'] if f['id']!='Italic'])
            self.assertEqual(complete(records,'alphabet'),([],[]))
            # The catalog searched with keeps every face by every line it has: Italic stands on its one line.
            ids,skipped=complete(records[:-1],'captured');partial=make_catalog([records[i] for i in ids],vectors[ids],encoder,'manifest','captured')
            self.assertEqual((ids,skipped),([0,1,2,3,4],[]))
            self.assertEqual([(f['id'],f['referenceIds']) for f in partial['faces']],[('Bold',['Boldwords-a-v1','Boldwords-b-v1']),('Italic',['Italicwords-a-v1']),('Regular',['Regularwords-a-v1','Regularwords-b-v1'])])
            bad=copy.deepcopy(records);bad[1]['weight']=700
            self.assertRaisesRegex(ValueError,'Conflicting',make_catalog,bad,vectors,encoder,'manifest','words')
            # The shipped recipe: every controlled capture of a face is its own row, never averaged away.
            shots=['latin-lower-v1','latin-upper-v1','words-a-v1','words-b-v1','digits-v1']
            every=[{**records[0],'id':'Regular'+r,'recipeId':r} for r in shots];rows=np.eye(128,dtype=np.float32)[:5]
            c=make_catalog(every,rows,encoder,'manifest','all');decoded,owners,_=read_catalog(c,encoder)
            self.assertEqual(owners.tolist(),[0]*5);np.testing.assert_array_equal(decoded[:,:5],np.eye(5)[[4,0,1,2,3]])  # rows follow record ids: digits first
            self.assertEqual(sorted(c['faces'][0]['referenceIds']),sorted('Regular'+r for r in shots))
            for recipe in ['alphabet','all','unknown']:
                self.assertRaises(ValueError,make_catalog,records,vectors,encoder,'manifest',recipe)
            self.assertRaises(ValueError,make_catalog,[],np.empty((0,128)),encoder,'manifest','words')
            self.assertRaises(ValueError,make_catalog,records,vectors[:-1],encoder,'manifest','words')
            self.assertRaises(ValueError,make_catalog,records,np.zeros_like(vectors),encoder,'manifest','words')


    def test_a_catalog_holds_up_to_100000_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=Path(tmp)/'encoder.json';save(encoder,{'dimensions':128,'preparation':{'width':128,'height':48,'windows':3}})
            face=lambda i:{'id':f'f{i}','familyId':f'f{i}','family':f'F{i}','referenceIds':[f'f{i}-words']}
            records=[{'id':f'f{i}-words','recipeId':'words-a-v1','faceId':f'f{i}','familyId':f'f{i}','family':f'F{i}','styleName':'S','weight':400,'slant':'upright','axes':{},'foundry':'F','sourceUrl':'https://example.test/a','scripts':['Latn']} for i in range(10001)]
            vectors=np.zeros((10001,128),np.float32);vectors[np.arange(10001),np.arange(10001)%128]=1
            many=make_catalog(records,vectors,encoder,'manifest','captured');self.assertEqual(len(read_catalog(many,encoder)[2]),10001)
            self.assertRaisesRegex(ValueError,'faces',read_catalog,{**many,'faces':[face(i) for i in range(100001)]},encoder)


if __name__=='__main__':unittest.main()
