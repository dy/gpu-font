import unittest

from train.encoder_text_diagnostic import paired_sources


class TextPairTests(unittest.TestCase):
    def test_only_complete_same_size_cross_renderer_pairs_enter_comparison(self):
        samples=[{'family':f,'text':t,'renderer':r,'script':'Latn','condition':'clean','size':56,'length':'8+'}
                 for f in ['b','a'] for t in ['abcdefgh','mnopqrstuvwx'] for r in ['chromium','pillow']]
        incomplete={**samples[0],'family':'incomplete'};wrong_size={**samples[0],'family':'incomplete','size':24}
        texts,families,ids=paired_sources(samples+[incomplete,wrong_size])
        self.assertEqual(families,['a','b']);self.assertEqual(set(texts),{'abcdefgh','mnopqrstuvwx'})
        for f in families:
            for t in texts:
                for r in ['chromium','pillow']:
                    s=samples[ids[f,t,r]];self.assertEqual((s['family'],s['text'],s['renderer']),(f,t,r))
        self.assertRaises(ValueError,paired_sources,samples+[samples[0]])
        self.assertRaises(ValueError,paired_sources,[])
        self.assertRaises(ValueError,paired_sources,[s for s in samples if s['renderer']=='pillow'])
        self.assertEqual(paired_sources(samples)[:2],paired_sources(samples)[:2])


if __name__=='__main__':unittest.main()
