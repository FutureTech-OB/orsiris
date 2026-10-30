from pathlib import Path
import hashlib
import json
import platform
import time
import torch
from . import __version__
from .backends import ToyBackend, HFBackend, capture_prefix
from .config import load_config, load_cases
from .interventions import Intervention, InterventionSession
from .representations import JSpace, TopKSAE, FrozenController
from .metrics import choice_probabilities, effect_scores
from .io import write_json, sha256


class DirectionRegistry:
    def __init__(self, definitions, backend, model_id):
        self.definitions, self.backend = definitions, backend
        self.assets, self.resolved, self.token_ids = {}, {}, {}
        self.model_id = model_id
        for name,d in definitions.items():
            if d["layer"] >= len(backend.layers):
                raise ValueError(f"Layer unavailable for direction {name}")
            kind = d["kind"]
            if kind == "vector":
                self.resolved[name] = torch.tensor(d["vector"],dtype=torch.float32)
            elif kind == "jspace":
                key = ("jspace", d["checkpoint"], d["layer"])
                if key not in self.assets:
                    self.assets[key] = JSpace.load(d["checkpoint"],d["layer"])
                lens = self.assets[key]
                ids = d.get("token_ids") or backend.concept_ids(d["tokens"])
                if any(type(i) is not int or i<0 or i>=len(backend.unembedding) for i in ids):
                    raise ValueError("Invalid concept token ids")
                self.token_ids[name] = ids
                rows = backend.unembedding[ids].detach().float().cpu()
                self.resolved[name] = lens.direction(rows,d.get("weights"))
            elif kind == "sae":
                sae = self.sae(d["checkpoint"],d)
                self.resolved[name] = sae.direction(d["feature_ids"],d.get("weights"))
            else:
                ctrl = FrozenController.load(d["checkpoint"],model_id,d["layer"])
                if ctrl.mode == "context" and not d.get("sae_checkpoint"):
                    raise ValueError("A context controller requires its SAE checkpoint")
                self.resolved[name] = ctrl
        for value in self.resolved.values():
            width = value.vectors.shape[1] if isinstance(value,FrozenController) else value.numel()
            if width != backend.unembedding.shape[1]:
                raise ValueError("Representation hidden width differs from model")

    def sae(self,path,d):
        key = (path,d["top_k"],d["activation"])
        if key not in self.assets:
            self.assets[key] = TopKSAE.load(*key)
        return self.assets[key]

    def direction(self,name,encoded,cache):
        obj = self.resolved[name]
        if not isinstance(obj,FrozenController):
            return obj
        d = self.definitions[name]
        codes = None
        if obj.mode == "context":
            # Cache only within the case; never cache choices across agents/tasks.
            key = (d["layer"],d.get("context_last_n",16))
            if key not in cache:
                cache[key] = capture_prefix(self.backend,encoded,*key)
            codes = self.sae(d["sae_checkpoint"],d).encode(cache[key],obj.feature_ids).mean(0)
        return obj.direction(codes)

    def intervention(self,method,encoded,cache):
        name = method["direction"]
        d = self.definitions[name]
        position = method.get("site","last")
        if position == "event_end":
            position = encoded.event_end
        return Intervention(layer=d["layer"],mode=method["mode"],
            direction=self.direction(name,encoded,cache),strength=method.get("strength",1.),
            target=self.direction(method["target"],encoded,cache) if method["mode"]=="swap" else None,
            position=position,scale=method.get("scale","absolute"),normalize=d["kind"]!="controller",
            name=method["name"])


def run(config_path, output=None):
    cfg = load_config(config_path)
    protocol = load_cases(cfg["cases"])
    out = Path(output).resolve() if output else Path(cfg["output"])
    # Never overwrite an earlier run, including a failed/partial run.
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    write_json(out/"config.resolved.json",cfg)
    write_json(out/"protocol.json",protocol)
    write_json(out/"status.json",{"state":"running"})
    try:
        torch.manual_seed(cfg.get("seed",0))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.get("seed",0))
        torch.backends.cuda.matmul.allow_tf32 = False
        backend = (ToyBackend if cfg["model"]["backend"]=="toy" else HFBackend)(cfg["model"],cfg.get("seed",0))
        registry = DirectionRegistry(cfg["directions"],backend,cfg["model"]["model_id"])
        asset_paths = {d[k] for d in cfg["directions"].values() for k in ("checkpoint","sae_checkpoint") if k in d}
        source_files = sorted(Path(__file__).parent.glob("*.py"))
        metadata = {"package_version":__version__,"python":platform.python_version(),"torch":torch.__version__,
            "synthetic":backend.synthetic,"model_id":cfg["model"]["model_id"],"seed":cfg.get("seed",0),
            "config_sha256":sha256(config_path),"cases_sha256":sha256(cfg["cases"]),
            "checkpoint_sha256":{str(p):sha256(p) for p in sorted(asset_paths)},
            "code_sha256":{p.name:sha256(p) for p in source_files},"concept_token_ids":registry.token_ids,
            "probability_definition":"next-token probability conditioned on disjoint legal letter aliases",
            "aggregation":"model effects averaged across option orders, then absolute error normalized by scale span, then equally averaged across effects"}
        if cfg["model"]["backend"]=="hf":
            import transformers
            metadata["transformers"] = transformers.__version__
            metadata["model_local_files"] = {str(p.relative_to(cfg["model"]["path"])):sha256(p)
                for p in Path(cfg["model"]["path"]).glob("*.json") if p.stat().st_size < 50_000_000}
        write_json(out/"metadata.json",metadata)
        rows = []
        with (out/"trials.jsonl").open("w",encoding="utf-8") as f:
            for case in protocol["cases"]:
                for order_index,order in enumerate(case.get("orders",[list(range(len(case["choices"])))])):
                    encoded = backend.encode(case,order)
                    aliases = backend.label_ids(len(order))
                    cache = {}
                    for method in [None]+cfg["interventions"]:
                        interventions = [] if method is None else [registry.intervention(method,encoded,cache)]
                        with InterventionSession(backend.layers,interventions) as session, torch.inference_mode():
                            logits = backend.forward(encoded.ids)
                        values = [case["choices"][j]["value"] for j in order]
                        scores = choice_probabilities(logits,aliases)
                        probs = scores["probabilities"]
                        row = {"case_id":case["id"],"method":"baseline" if method is None else method["name"],
                            "order_index":order_index,"choice_order":[case["choices"][j]["id"] for j in order],
                            "values":values,**scores,"mean":sum(v*p for v,p in zip(values,probs)),
                            "action_probabilities":{case["choices"][j]["id"]:p for j,p in zip(order,probs)},
                            "argmax_choice":case["choices"][order[max(range(len(probs)),key=probs.__getitem__) ]]["id"],
                            "prompt_sha256":hashlib.sha256(encoded.text.encode()).hexdigest(),
                            "input_ids_sha256":hashlib.sha256(json.dumps(encoded.ids[0].tolist()).encode()).hexdigest(),
                            "formatted_prompt":encoded.text,"n_tokens":encoded.ids.shape[1],"event_end":encoded.event_end,
                            "interventions":session.records,"synthetic":backend.synthetic}
                        rows.append(row)
                        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n"); f.flush()
        summary = {"synthetic":backend.synthetic,"rows":len(rows),"cases":len(protocol["cases"]),
            "min_label_mass":min(r["label_mass"] for r in rows),
            "effects":effect_scores(rows,protocol.get("effects",[])),
            "seconds":time.monotonic()-started}
        write_json(out/"summary.json",summary)
        write_json(out/"status.json",{"state":"complete","rows":len(rows)})
        return summary
    except BaseException as exc:
        write_json(out/"status.json",{"state":"failed","error_type":type(exc).__name__,"error":str(exc)})
        raise
