"""Hand-calculable scoring checks for the sealed executive-study reanalysis."""
import copy,unittest
from cookbook.executive.analyze import summarize


def row(cid,order,mean):
    values=list(range(6));probs=[int(v==mean) for v in values]
    return {'case_id':cid,'rotation':order,'study_id':'S','values':values,'probabilities':probs,
            'mean':mean,'top_legal':True,'label_mass':1.,'prompt_sha256':cid}


class ExecutiveTests(unittest.TestCase):
    def test_group_weights_and_reverse_coding(self):
        ref={'study_id':'S','title':'fixture','evidence':'software only','kind':'calibration','orders':[0],
             'cases':[{'id':'a','group':'A','item':1,'group_n':10},
                      {'id':'b','group':'B','item':1,'group_n':30},
                      {'id':'c','group':'C','item':1,'group_n':40}],
             'effects':[{'effect_id':'E','groups1':['A','B'],'groups2':['C'],'items':[1],'human_delta':2,'span':5}],
             'probability_quality':.9}
        rows=[row('a',0,1),row('b',0,5),row('c',0,2)]
        result,details=summarize(ref,{'prompt':rows,'sae':copy.deepcopy(rows)})
        self.assertEqual(details[0]['mean_model_delta'],2)
        self.assertEqual(result['methods']['prompt']['raw_effect_mae'],0)
        self.assertIsNone(result['relative_error_reduction'])
        ref['cases'][0].update(score_offset=6,score_multiplier=-1)
        ref['effects'][0].update(groups1=['A'],groups2=['C'],human_delta=3)
        _,details=summarize(ref,{'prompt':rows,'sae':copy.deepcopy(rows)})
        self.assertEqual(details[0]['mean_model_delta'],3)

    def test_transfer_averages_order_before_absolute_error(self):
        ref={'study_id':'S','title':'fixture','evidence':'software only','kind':'frozen_transfer','orders':[0,1],
             'cases':[{'id':'a'},{'id':'b'}],
             'effects':[{'effect_id':'E','coefficients':{'a':1,'b':-1},'human_delta':1,'span':2}],
             'probability_quality':.9}
        rows=[row('a',0,0),row('a',1,2),row('b',0,0),row('b',1,0)]
        result,details=summarize(ref,{'prompt':rows,'sae':copy.deepcopy(rows)})
        self.assertEqual(result['methods']['prompt']['raw_effect_mae'],0)
        self.assertEqual(result['methods']['prompt']['mean_per_order_scale_error'],.5)
        self.assertEqual(details[0]['order_deltas'],[0,2])

    def test_calibration_defaults_to_same_pooled_policy_as_transfer(self):
        ref={'study_id':'S','title':'fixture','evidence':'software only','kind':'calibration','orders':[0,1],
             'cases':[{'id':'a'},{'id':'b'}],
             'effects':[{'effect_id':'E','coefficients':{'a':1,'b':-1},'human_delta':1,'span':2}],
             'probability_quality':.9}
        rows=[row('a',0,0),row('a',1,2),row('b',0,0),row('b',1,0)]
        result,_=summarize(ref,{'prompt':rows,'sae':copy.deepcopy(rows)})
        self.assertEqual(result['aggregation'],'pooled_orders')
        self.assertEqual(result['methods']['prompt']['raw_effect_mae'],0)

    def test_missing_duplicate_and_invalid_probabilities_rejected(self):
        ref={'study_id':'S','title':'fixture','evidence':'software only','kind':'frozen_transfer','orders':[0],
             'cases':[{'id':'a'}],'effects':[{'effect_id':'E','coefficients':{'a':1},'human_delta':1,'span':5}],
             'probability_quality':.9}
        r=row('a',0,2)
        with self.assertRaises(ValueError):summarize(ref,{'prompt':[],'sae':[r]})
        with self.assertRaises(ValueError):summarize(ref,{'prompt':[r,r],'sae':[r]})
        bad=copy.deepcopy(r);bad['probabilities']=[.2]*6
        with self.assertRaises(AssertionError):summarize(ref,{'prompt':[bad],'sae':[r]})

    def test_source_calibration_can_report_pooled_and_per_order_errors(self):
        ref={'study_id':'S','title':'source','evidence':'calibration','kind':'source_calibration',
             'aggregation':'pooled_orders','orders':[0,1],'cases':[{'id':'a'},{'id':'b'}],
             'effects':[{'effect_id':'E','coefficients':{'a':1,'b':-1},'human_delta':1,'span':2}],
             'probability_quality':.9}
        rows=[row('a',0,0),row('a',1,2),row('b',0,0),row('b',1,0)]
        result,_=summarize(ref,{'prompt':rows,'sae':copy.deepcopy(rows)})
        self.assertEqual(result['kind'],'source_calibration')
        self.assertEqual(result['methods']['prompt']['raw_effect_mae'],0)
        self.assertEqual(result['methods']['prompt']['mean_per_order_scale_error'],.5)
        ref['aggregation']='per_order'
        result,_=summarize(ref,{'prompt':rows,'sae':copy.deepcopy(rows)})
        self.assertEqual(result['methods']['prompt']['raw_effect_mae'],1)


if __name__=='__main__':unittest.main()
