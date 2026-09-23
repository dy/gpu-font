import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from train import encoder_words
from train.encoder_quality import with_words
from train.encoder_references import prototypes


class CaseTrainingTests(unittest.TestCase):
    def test_plan_keeps_held_out_families_and_reserved_text_out_of_case_training(self):
        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        families=[{'id':f,'excluded':False,'alphabets':{'Latn':alphabet if f!='caps' else alphabet.upper()}} for f in ['train','caps','dev','test']]
        data={'corpus.json':{'families':families},'encoder-split.json':{'families':{'train':'train','caps':'train','dev':'development','test':'test'}},
              'development.json':{'samples':[{'text':'ri'}]},'final.json':{'samples':[{'text':'ov'}]},'phrases.json':{'samples':[{'text':'Quiet rivers flow'}]}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(encoder_words,'OUT',Path(tmp)), patch.object(encoder_words,'sha',return_value='fixture'):
            def read(path):
                return json.loads(path.read_text()) if path.exists() and path.parent==Path(tmp) else data[path.name]
            with patch.object(encoder_words,'read',side_effect=read):
                encoder_words.plan();first=(Path(tmp)/'plan.json').read_bytes();encoder_words.plan()
                self.assertEqual(first,(Path(tmp)/'plan.json').read_bytes())
                result=json.loads(first)
                self.assertEqual([j['family']['id'] for j in result['jobs']],['train'])
                self.assertEqual(len(set(t.casefold() for t in result['texts'])),32)
                self.assertTrue(all(t.islower() for t in result['texts'][:16]))
                self.assertTrue(all(t[0].isupper() and t[1:].islower() for t in result['texts'][16:24]))
                self.assertTrue(all(t.isupper() for t in result['texts'][24:]))
                self.assertFalse(set(t.casefold() for t in result['texts'])&{'ri','ov','quiet rivers flow'})
                self.assertTrue(all(p['role']=='train' for j in result['jobs'] for p in j['plans']))

    def test_word_references_keep_case_groups_and_correct_family_owners(self):
        samples=[{'family':'a','script':'Latn'},{'family':'b','script':'Latn'}];rv=np.array([[1,0,0],[0,1,0]],dtype=np.float32)
        base=prototypes(samples,rv,['a','b'],'scripts')
        words=[{'family':'a','referenceGroup':'lower'},{'family':'a','referenceGroup':'upper'},{'family':'outside','referenceGroup':'lower'}]
        vectors=np.array([[0,0,1],[1,1,0],[1,0,0]],dtype=np.float32)
        rows,owners,packed,scales=with_words(*base,['a','b'],words,vectors)
        self.assertEqual(owners.tolist(),[0,1,0,0]);self.assertEqual(len(rows),4)
        np.testing.assert_array_equal(rows[:3],np.eye(3))
        np.testing.assert_allclose(rows[3],[2**-.5,2**-.5,0],atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(rows,axis=1),1,atol=1e-6)
        self.assertEqual(packed.shape,rows.shape);self.assertEqual(scales.shape,(4,))


if __name__=='__main__':unittest.main()
