from pathlib import Path
import argparse
import json
import os
import time
import torch
from domain_engine import DomainEngine, encode
from run_sae_chat_v4 import load, save, sha, emit

ROOT = Path(__file__).resolve().parent

def run(shard):
    d = load(ROOT / "protocol.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(d["gpus"][shard])
    torch.set_num_threads(2)
    torch.manual_seed(d["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = ROOT / "results" / str(shard)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def log(stage, **kw):
        print(json.dumps(dict(shard=shard, stage=stage,
            seconds=round(time.monotonic()-start, 1), **kw)), flush=True)
    engine = DomainEngine(d)
    log("model_ready")
    blob = torch.load(Path(d["sae_root"]) / "layer48.sae.pt", weights_only=True,
                      map_location="cpu", mmap=True)
    studies = {s["study_id"]: s for s in d["studies"]}
    controllers = {}
    metadata = {}
    for sid in {c["study_id"] for c in d["cases"] if c["shard"] == shard}:
        s = studies[sid]
        prior = Path(s["controller_directory"])
        selection = load(prior / "primary_frozen.json")["configuration"]
        assert selection is not None
        payload = torch.load(prior / "controllers.pt", weights_only=True, map_location="cpu")
        if s["controller_format"] == "v10":
            ctrl = payload["controllers"][selection["direction"]]
            center, scale = payload["centers"]["context"], payload["scales"]["context"]
        else:
            ctrl = payload["controllers"][selection["controller"]]
            center, scale = payload["center"], payload["scale"]
        assert ctrl["mode"] == "context"
        controllers[sid] = (selection, ctrl, center, scale)
        metadata[sid] = dict(controller_sha256=sha(prior / "controllers.pt"),
            selection_sha256=sha(prior / "primary_frozen.json"), configuration=selection)
    save(out / "metadata.json", dict(script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(ROOT / "protocol.json"), controllers=metadata))
    prefixes = {}
    cc = [c for c in d["cases"] if c["shard"] == shard]
    count, formats = 0, []
    with (out / "readouts.jsonl").open("w") as f:
        for ci, c in enumerate(cc):
            selection, ctrl, center, scale = controllers[c["study_id"]]
            fids = ctrl["fids"]
            for order in studies[c["study_id"]]["orders"]:
                row, _, _ = engine.state(c, order, False)
                ph = row["context_sha256"]
                if ph not in prefixes:
                    prefixes[ph] = encode(engine.prefix_context(c, order), blob).mean(0)
                gate = ((prefixes[ph][fids]-center[fids])/scale[fids]).clamp(-2, 2)
                direction = (gate*ctrl["weights"]) @ ctrl["vectors"] / ctrl["rms"]
                direction *= min(1., 2./max(float(direction.norm()), 1e-12))
                delta = {48: direction*(selection["dose"]*row["h_norms"]["48"])}
                steered, _ = engine.read(c, order, delta=delta)
                assert steered["prompt_sha256"] == row["prompt_sha256"]
                assert steered["values"] == row["values"]
                for method, r in [("prompt", row), ("sae", steered)]:
                    r.update(method=method, persona_id=c["persona_id"],
                        source_case_id=c["source_case_id"], shard=shard)
                    assert abs(sum(r["probabilities"])-1) < 1e-8
                    emit(f, r)
                    count += 1
                if ci == 0 and order == 0:
                    for method, dd in [("prompt", None), ("sae", delta)]:
                        g = engine.generate(c, order, dd)
                        g["method"] = method
                        formats.append(g)
            if (ci+1) % 10 == 0 or ci+1 == len(cc):
                log("progress", cases_done=ci+1, cases_total=len(cc), rows=count)
    assert count == d["expected_reads_by_shard"][shard]
    save(out / "format.json", formats)
    save(out / "complete.json", dict(status="complete", rows=count, cases=len(cc),
        seconds=time.monotonic()-start, shard=shard))
    log("complete", rows=count)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    run(p.parse_args().shard)
