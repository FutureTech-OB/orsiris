"""Tiny deterministic fixtures exercise the real hook/SAE/JSpace path on CPU."""
from pathlib import Path
import torch
from .io import write_json
from .representations import FrozenController


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    gen = torch.Generator().manual_seed(2718)
    J = torch.eye(16) + torch.randn(16,16,generator=gen)*.02
    torch.save({"J":{0:J},"d_model":16},root/"lens.pt")
    enc = torch.randn(32,16,generator=gen)*.2
    dec = torch.randn(16,32,generator=gen)*.2
    torch.save({"W_enc":enc,"b_enc":torch.zeros(32),"W_dec":dec},root/"sae.pt")
    vectors = torch.stack([dec[:,i]/dec[:,i].norm() for i in [0,1]])
    ctrl = FrozenController(vectors,torch.tensor([.8,-.3]),[0,1],"context",torch.zeros(2),torch.ones(2),1.)
    torch.save(ctrl.to_payload("toy-v1",0),root/"controller.pt")
    choices = [{"id":"accept","text":"Accept the allocation.","value":1},
               {"id":"reject","text":"Reject; both participants receive zero.","value":0}]
    write_json(root/"cases.json",{"schema_version":1,"description":"Synthetic smoke fixtures, not research evidence.",
        "cases":[{"id":"unequal","event":"Another participant keeps 90 units and offers you 10 units.",
                  "question":"Would you accept?","choices":choices,"orders":[[0,1],[1,0]]},
                 {"id":"equal","event":"Another participant keeps 50 units and offers you 50 units.",
                  "question":"Would you accept?","choices":choices,"orders":[[0,1],[1,0]]}],
        "effects":[{"id":"illustrative_only","coefficients":{"equal":1,"unequal":-1},
                    "human_delta":0.2,"span":1,"split":"synthetic_fixture"}]})
    definitions = {
        "j_from":{"kind":"jspace","model_id":"toy-v1","layer":0,"checkpoint":"lens.pt","token_ids":[1,2]},
        "j_to":{"kind":"jspace","model_id":"toy-v1","layer":0,"checkpoint":"lens.pt","token_ids":[3]},
        "sae_combo":{"kind":"sae","model_id":"toy-v1","layer":0,"checkpoint":"sae.pt",
                     "feature_ids":[0,1],"weights":[1,-.5],"top_k":4,"activation":"raw_topk"},
        "frozen":{"kind":"controller","model_id":"toy-v1","layer":0,"checkpoint":"controller.pt",
                  "sae_checkpoint":"sae.pt","top_k":4,"activation":"raw_topk","context_last_n":8}}
    configs = {
        "jspace":[{"name":"j_ablate","mode":"ablate","direction":"j_from","strength":1,"site":"last"},
                  {"name":"j_swap","mode":"swap","direction":"j_from","target":"j_to","strength":1,"site":"last"},
                  {"name":"j_add_event","mode":"add","direction":"j_from","strength":.15,"scale":"residual_norm","site":"event_end"}],
        "sae":[{"name":"sae_combo","mode":"add","direction":"sae_combo","strength":.1,"scale":"residual_norm","site":"last"}],
        "controller":[{"name":"frozen_controller","mode":"add","direction":"frozen","strength":.1,"scale":"residual_norm","site":"last"}]}
    for name,methods in configs.items():
        used = {m["direction"] for m in methods} | {m["target"] for m in methods if "target" in m}
        write_json(root/(name+".json"),{"schema_version":1,"seed":42,"model":{"backend":"toy","model_id":"toy-v1"},
            "cases":"cases.json","output":f"results/{name}","directions":{k:definitions[k] for k in sorted(used)},"interventions":methods})
    return root
