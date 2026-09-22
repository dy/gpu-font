import unittest

import numpy as np

from train.encoder_references import cluster, prototypes


class EncoderReferenceTests(unittest.TestCase):
    def test_script_means_do_not_mix_distinct_glyph_distributions(self):
        samples=[{'family':'a','script':'Latn'},{'family':'a','script':'Grek'},{'family':'a','script':'Latn'},{'family':'b','script':'Latn'}]
        vectors=np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1]],dtype=np.float32)
        before=vectors.copy()
        for order in [np.arange(4),np.arange(4),np.array([3,2,1,0]),np.arange(4)]:
            rows,owners,packed,scales=prototypes([samples[i] for i in order],vectors[order],['a','b'],'scripts')
            np.testing.assert_array_equal(rows,[[0,1,0],[1,0,0],[0,0,1]])
            np.testing.assert_array_equal(owners,[0,0,1]);np.testing.assert_allclose(packed*scales[:,None],rows,atol=1e-6)
        np.testing.assert_array_equal(vectors,before)
        mean,owners,_,_=prototypes(samples,vectors,['a','b'],'mean')
        np.testing.assert_allclose(mean[0,:2],[2/np.sqrt(5),1/np.sqrt(5)],atol=.004)
        self.assertEqual(owners.tolist(),[0,1])
        single=prototypes(samples[:1],vectors[:1],['a'],'scripts')
        np.testing.assert_array_equal(single[0],[[1,0,0]])
        for args in [([],[],[],'mean'),(samples,vectors,['missing'],'scripts'),(samples,vectors,['a'],'unknown')]:
            self.assertRaises(ValueError,prototypes,*args)

    def test_spherical_clusters_are_deterministic_bounded_and_unit_length(self):
        vectors=np.eye(3,dtype=np.float32);before=vectors.copy()
        np.testing.assert_array_equal(cluster(vectors[:1],8),vectors[:1])
        np.testing.assert_array_equal(cluster(vectors,8),vectors)
        np.testing.assert_array_equal(cluster(vectors,8),cluster(vectors,8))
        repeated=cluster(np.repeat(vectors[:1],4,axis=0),4)
        np.testing.assert_array_equal(repeated,np.repeat(vectors[:1],4,axis=0))
        self.assertEqual(len(cluster(vectors,2)),2)
        np.testing.assert_allclose(np.linalg.norm(cluster(vectors,2),axis=1),1,atol=1e-6)
        np.testing.assert_array_equal(vectors,before)
        for bad in [[],[[0,0]],[[np.nan,1]]]:self.assertRaises(ValueError,cluster,bad,4)


if __name__=='__main__':unittest.main()
