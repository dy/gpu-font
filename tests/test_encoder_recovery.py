import unittest
from train.encoder_recovery import choose


class RecoverySelectionTests(unittest.TestCase):
    def test_quality_gate_rejects_regressions_missing_cases_and_average_only_wins(self):
        baseline={'selection':.6,'ranks':{'lora':2,'montserrat':29,'other':5}}
        good={'name':'small','selection':.65,'bytes':490000,'ranks':{'lora':1,'montserrat':12,'other':3}}
        bad=[{**good,'selection':.59},{**good,'ranks':{**good['ranks'],'lora':2}},
             {**good,'ranks':{**good['ranks'],'montserrat':29}},
             {**good,'selection':.8,'ranks':{**good['ranks'],'other':6}},
             {**good,'ranks':{'lora':1,'montserrat':1}}]
        for candidate in bad:self.assertRaises(ValueError,choose,[candidate],baseline)
        self.assertRaises(ValueError,choose,[],baseline)
        self.assertEqual(choose(bad+[good],baseline),good)
        larger={**good,'name':'large','selection':.659,'bytes':1800000}
        self.assertEqual(choose([larger,good],baseline),good)
        self.assertEqual(choose([good,larger],baseline),good)
        larger['selection']=.67;self.assertEqual(choose([good,larger],baseline),larger)


if __name__=='__main__':unittest.main()
