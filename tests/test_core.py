import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from repsteer import Intervention, InterventionSession, JSpace, TopKSAE, FrozenController
from repsteer.demo import prepare
from repsteer.runner import run
from repsteer.config import load_config, load_cases, validate_config
from repsteer.metrics import choice_probabilities, effect_scores, sample_choices
from repsteer.io import read_json, write_json
from repsteer.backends import ToyBackend, capture_prefix


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.h = torch.tensor([[[1.,2.,3.],[2.,4.,6.]]])
        self.v = torch.tensor([1.,0.,0.])

    def test_ablation_and_immutability(self):
        before = self.h.clone()
        edited, _ = Intervention(0,"ablate",self.v).apply(self.h)
        self.assertTrue(torch.equal(self.h,before))
        self.assertTrue(torch.equal(edited[0,0],before[0,0]))
        self.assertEqual(edited[0,-1,0],0)

    def test_swap_preserves_legacy_projection_formula(self):
        v2 = torch.tensor([0.,1.,0.])
        edited,_ = Intervention(0,"swap",self.v,target=v2,strength=.5).apply(self.h)
        torch.testing.assert_close(edited[0,-1],torch.tensor([1.,5.,6.]))

    def test_relative_norm_and_sham(self):
        edited, record = Intervention(0,"add",self.v,strength=-.2,scale="residual_norm").apply(self.h)
        self.assertAlmostEqual(record["mean_actual_norm_ratio"],.2,places=6)
        self.assertLess(edited[0,-1,0],self.h[0,-1,0])
        sham,_ = Intervention(0,"add",self.v,strength=0).apply(self.h)
        self.assertTrue(torch.equal(sham,self.h))

    def test_dtype_and_arbitrary_site(self):
        edited,_ = Intervention(0,"add",self.v,strength=.5,position=0).apply(self.h.bfloat16())
        self.assertEqual(edited.dtype,torch.bfloat16)
        torch.testing.assert_close(edited[0,-1],self.h[0,-1].bfloat16())

    def test_all_positions(self):
        edited,_ = Intervention(0,"ablate",self.v,position="all").apply(self.h)
        self.assertTrue(torch.equal(edited[...,0],torch.zeros(1,2)))

    def test_invalid_directions_and_sites(self):
        for kwargs in [{"mode":"add","direction":torch.zeros(3)},
                       {"mode":"swap","direction":self.v},
                       {"mode":"ablate","direction":self.v,"scale":"residual_norm"}]:
            with self.assertRaises(ValueError): Intervention(0,**kwargs)
        with self.assertRaises(IndexError): Intervention(0,"add",self.v,position=2).apply(self.h)

    def test_zero_gated_controller_delta_is_allowed(self):
        edited,_ = Intervention(0,"add",torch.zeros(3),normalize=False).apply(self.h)
        self.assertTrue(torch.equal(edited,self.h))

    def test_tuple_hook_and_prefill_one_token(self):
        class TupleLayer(nn.Module):
            def forward(self,h): return h,"cache"
        layer = TupleLayer()
        one = self.h[:,:1]
        with InterventionSession([layer],[Intervention(0,"ablate",self.v)]) as s:
            out, cache = layer(one)
            second,_ = layer(one)
        self.assertEqual(cache,"cache")
        self.assertEqual(out[0,0,0],0)
        torch.testing.assert_close(second,one)
        self.assertEqual(len(s.records),1)
        self.assertEqual(len(layer._forward_hooks),0)

    def test_cleanup_when_forward_raises(self):
        layer=nn.Identity()
        with self.assertRaises(RuntimeError):
            with InterventionSession([layer],[Intervention(0,"add",self.v)]):
                layer(self.h)
                raise RuntimeError("test")
        self.assertEqual(len(layer._forward_hooks),0)

    def test_all_forwards_and_ordered_composition(self):
        layer=nn.Identity()
        ops=[Intervention(0,"ablate",self.v),Intervention(0,"add",self.v,strength=2)]
        with InterventionSession([layer],ops,scope="all_forwards") as s:
            self.assertEqual(layer(self.h)[0,-1,0],2)
            self.assertEqual(layer(self.h)[0,-1,0],2)
        self.assertEqual(len(s.records),4)
        with self.assertRaises(ValueError):
            with InterventionSession([layer],[Intervention(0,"add",self.v,position=0)],scope="all_forwards"): pass


class RepresentationTests(unittest.TestCase):
    def test_jacobian_orientation(self):
        J=torch.tensor([[1.,2.],[3.,5.]])
        w=torch.tensor([[2.,-1.]])
        h=torch.tensor([.5,3.])
        lens=JSpace(J)
        torch.testing.assert_close(h@(w[0]@J),w[0]@lens.transport(h))
        torch.testing.assert_close(lens.direction(w),(w[0]@J)/(w[0]@J).norm())

    def test_sae_gate_before_selecting_features(self):
        sae=TopKSAE(torch.eye(3),torch.zeros(3),torch.eye(3),1)
        h=torch.tensor([[3.,2.,1.]])
        torch.testing.assert_close(sae.encode(h,[1]),torch.tensor([[0.]]))
        torch.testing.assert_close(sae.encode(h),torch.tensor([[3.,0.,0.]]))

    def test_raw_topk_and_relu_are_explicit(self):
        args=(torch.eye(2),torch.zeros(2),torch.eye(2),1)
        raw=TopKSAE(*args,"raw_topk"); relu=TopKSAE(*args,"relu_topk")
        self.assertEqual(float(raw.encode(torch.tensor([[-1.,-2.]])).sum()),-1.)
        self.assertEqual(float(relu.encode(torch.tensor([[-1.,-2.]])).sum()),0.)

    def test_sae_dimension_and_id_validation(self):
        with self.assertRaises(ValueError): TopKSAE(torch.zeros(3,2),torch.zeros(3),torch.zeros(3,2),1)
        sae=TopKSAE(torch.eye(3),torch.zeros(3),torch.eye(3),1)
        with self.assertRaises(ValueError): sae.direction([3])
        with self.assertRaises(ValueError): sae.direction([0,0])

    def test_legacy_context_controller_formula_both_formats(self):
        c={"mode":"context","fids":[1,3],"vectors":torch.tensor([[1.,0.],[0.,1.]]),
           "weights":torch.tensor([2.,-1.]),"rms":.5}
        center=torch.arange(4,dtype=torch.float32); scale=torch.ones(4)*2
        for payload in [{"controllers":{"x":c},"center":center,"scale":scale},
                        {"controllers":{"x":c},"centers":{"context":center},"scales":{"context":scale}}]:
            ctrl=FrozenController.from_legacy(payload,"x")
            z=torch.tensor([5.,-5.])
            expected=(((z-center[c["fids"]])/scale[c["fids"]]).clamp(-2,2)*c["weights"])@c["vectors"]/c["rms"]
            expected*=min(1.,2./float(expected.norm()))
            torch.testing.assert_close(ctrl.direction(z),expected)

    def test_controller_roundtrip_and_model_check(self):
        ctrl=FrozenController(torch.eye(2),torch.ones(2),[],rms=2.)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"controller.pt"; torch.save(ctrl.to_payload("model",1),path)
            torch.testing.assert_close(FrozenController.load(path,"model",1).direction(),ctrl.direction())
            with self.assertRaises(ValueError): FrozenController.load(path,"other-model",1)

    def test_context_prefix_is_independent_of_question_and_options(self):
        backend=ToyBackend({},0)
        case={"event":"The shared event ends here.","question":"First question?",
              "choices":[{"id":"a","text":"Yes","value":1},{"id":"b","text":"No","value":0}]}
        a=backend.encode(case,[0,1])
        case["question"]="A very different and longer second question?"
        b=backend.encode(case,[1,0])
        torch.testing.assert_close(capture_prefix(backend,a,0,8),capture_prefix(backend,b,0,8),rtol=0,atol=0)


class MetricTests(unittest.TestCase):
    def test_legal_normalization_and_low_mass(self):
        r=choice_probabilities(torch.tensor([1000.,0.,0.,0.]),[[1,2],[3]])
        self.assertAlmostEqual(sum(r["probabilities"]),1,places=6)
        self.assertAlmostEqual(r["probabilities"][0],2/3,places=6)
        self.assertFalse(r["top_legal"])
        self.assertEqual(r["label_mass"],0)

    def test_aliases_must_not_overlap(self):
        with self.assertRaises(ValueError): choice_probabilities(torch.ones(3),[[0,1],[1,2]])

    def test_effect_metric_and_zero_baseline(self):
        rows=[{"method":m,"case_id":cid,"order_index":0,"mean":x}
            for m,pair in [("baseline",(3,1)),("sae",(2,1))] for cid,x in zip(["t","c"],pair)]
        effects=[{"id":"e","coefficients":{"t":1,"c":-1},"human_delta":1,"span":4}]
        result=effect_scores(rows,effects)
        self.assertEqual(result["baseline"]["scale_mae"],.25)
        self.assertEqual(result["sae"]["relative_error_reduction"],1)
        effects[0]["human_delta"]=2
        self.assertIsNone(effect_scores(rows,effects)["sae"]["relative_error_reduction"])
        with self.assertRaises(ValueError): effect_scores(rows+rows[:1],effects)
        with self.assertRaises(ValueError): effect_scores(rows[:-1],effects)

    def test_effect_errors_follow_pooled_order_policy(self):
        rows=[{"method":"baseline","case_id":cid,"order_index":order,"mean":value}
              for order,t in enumerate([0,2]) for cid,value in [("t",t),("c",0)]]
        effects=[{"id":"e","coefficients":{"t":1,"c":-1},"human_delta":1,"span":2}]
        result=effect_scores(rows,effects)["baseline"]
        self.assertEqual(result["scale_mae"],0)
        self.assertEqual(result["effect_details"][0]["model_delta"],1)
        self.assertEqual([r["absolute_error"] for r in result["details"]],[1,1])

    def test_monte_carlo_reproducible(self):
        a=sample_choices([.2,.8],5000,42); b=sample_choices([.2,.8],5000,42)
        self.assertTrue(torch.equal(a,b))
        self.assertLess(abs(float(a.float().mean())-.8),.04)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=prepare(Path(self.tmp.name)/"fixture")
    def tearDown(self): self.tmp.cleanup()

    def test_all_three_real_hook_workflows_and_repeatability(self):
        for name,count in [("jspace",16),("sae",8),("controller",8)]:
            summary=run(self.root/(name+".json"))
            self.assertEqual(summary["rows"],count)
            path=self.root/"results"/name
            rows=[json.loads(x) for x in (path/"trials.jsonl").read_text().splitlines()]
            hashes={}
            for r in rows:
                key=(r["case_id"],r["order_index"])
                hashes.setdefault(key,set()).add(r["prompt_sha256"])
                self.assertAlmostEqual(sum(r["probabilities"]),1,places=6)
                self.assertEqual(len(r["interventions"]),0 if r["method"]=="baseline" else 1)
            self.assertTrue(all(len(s)==1 for s in hashes.values()))
            self.assertTrue(summary["synthetic"])
        repeat=self.root/"repeat"
        run(self.root/"jspace.json",repeat)
        self.assertEqual((repeat/"trials.jsonl").read_bytes(),(self.root/"results/jspace/trials.jsonl").read_bytes())

    def test_fail_on_existing_output(self):
        run(self.root/"sae.json")
        with self.assertRaises(FileExistsError): run(self.root/"sae.json")

    def test_config_unknown_key_and_model_mismatch(self):
        cfg=read_json(self.root/"sae.json")
        cfg["interventions"][0]["strenght"]=1
        with self.assertRaises(ValueError): validate_config(cfg)
        del cfg["interventions"][0]["strenght"]
        cfg["directions"]["sae_combo"]["model_id"]="wrong"
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_thinking_is_not_silently_scored_as_a_choice(self):
        cfg=read_json(self.root/"sae.json")
        cfg["model"]["enable_thinking"]=True
        with self.assertRaisesRegex(ValueError,"enable_thinking=false"): validate_config(cfg)

    def test_relative_paths_resolved_from_config(self):
        cfg=load_config(self.root/"sae.json")
        self.assertEqual(Path(cfg["cases"]),self.root/"cases.json")

    def test_invalid_case_values_and_contrasts(self):
        payload=read_json(self.root/"cases.json")
        payload["effects"][0]["coefficients"]["unequal"]=1
        write_json(self.root/"bad_cases.json",payload)
        with self.assertRaises(ValueError): load_cases(self.root/"bad_cases.json")

    def test_failed_run_retains_status(self):
        cfg=read_json(self.root/"sae.json")
        cfg["directions"]["sae_combo"]["feature_ids"]=[999]
        write_json(self.root/"bad.json",cfg)
        with self.assertRaises(ValueError): run(self.root/"bad.json")
        self.assertEqual(read_json(self.root/"results/sae/status.json")["state"],"failed")


if __name__ == "__main__": unittest.main()
