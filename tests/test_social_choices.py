"""Selected-recipe arithmetic, stochastic decoding and complete-data contracts."""
from pathlib import Path
import json,unittest
import numpy as np
import torch
from cookbook.social_choices.run import SamplingBackend,RecordedSampler,uniforms,parse_response
from cookbook.social_choices.numerics import NumericIntervention
from cookbook.social_choices.analyze import summarize,paired_probability


class SocialChoiceTests(unittest.TestCase):
    def test_bf16_matches_frozen_original_outputs(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/social_choice_bf16_parity.json').read_text())
        h=torch.tensor(fixture['hidden'],dtype=torch.bfloat16)
        for c in fixture['cases']:
            op=NumericIntervention(0,c['mode'],torch.tensor(fixture['direction']),strength=c['strength'],
                                   target=torch.tensor(fixture['target']) if c['mode']=='swap' else None,
                                   position=c['position'])
            self.assertTrue(torch.equal(op.apply(h)[0].float(),torch.tensor(c['expected'])),c)

    def test_config_uses_model_specific_layers_and_selected_dose(self):
        root=Path(__file__).resolve().parents[1]/'cookbook/social_choices/configs'
        for model,dim,layers,dose in [('9b',4096,[16,20,24],11.30625),('27b',5120,[32,40,48],15.075)]:
            p=json.loads((root/f'{model}.json').read_text())
            backend=SamplingBackend.__new__(SamplingBackend)
            backend.config=p['studies']['reporting']['backend_config']
            backend.vectors={i:{'anger':torch.arange(1,dim+1).float()} for i in layers}
            ops=backend.interventions('anger')
            self.assertEqual([x.layer for x in ops],layers)
            self.assertTrue(all(x.strength==dose and x.direction.numel()==dim for x in ops))

    def test_inverse_cdf_is_sampling_not_argmax(self):
        sampler=RecordedSampler(((np.arange(1000)+.5)/1000)[:,None],3,1.,2,1.)
        sampled=sampler(torch.ones(1000,3,dtype=torch.long),torch.tensor([.2,.8]).log().repeat(1000,1)).argmax(-1)
        self.assertEqual(int((sampled==0).sum()),200)
        self.assertEqual(int((sampled==1).sum()),800)
        all_draws=uniforms(12,1,list(range(40)),4)
        np.testing.assert_array_equal(all_draws[3::4],uniforms(12,1,list(range(3,40,4)),4))
        self.assertEqual(parse_response('anger',['Report','Silent']),'Other')

    def test_analysis_does_not_drop_invalid_or_accept_duplicate_samples(self):
        p={'n_per_arm':4,'studies':{'s':{'target_choice':'A','conditions':['baseline','calm'],
            'source_conditions':{'baseline':'baseline','calm':'old_calm'},'backend_config':{'choices':['A','B'],
            'operations':{'calm':{'strength':1}}},'original_search_family_sizes':{'calm':7}}}}
        rows=[]
        for condition,answers in [('baseline',['B','B','A','A']),('old_calm',['A','B','A','Other'])]:
            rows.extend({'study':'s','condition':condition,'sample_id':i,'response':a,'reached_token_limit':False} for i,a in enumerate(answers))
        r=summarize(p,rows)['s']['calm']
        self.assertEqual(r['target_rate'],.5)
        self.assertEqual(r['other_rate'],.25)
        self.assertEqual(r['paired_increase'],1)
        self.assertEqual(r['paired_decrease'],1)
        self.assertEqual(r['conservative_correction_factor'],56)
        self.assertEqual(paired_probability(5,0),1/16)
        with self.assertRaises(ValueError):summarize(p,rows[:-1])
        with self.assertRaises(ValueError):summarize(p,rows+[rows[-1]])


if __name__=='__main__':unittest.main()
