import argparse
from pathlib import Path
import json
import torch
from .config import load_config, load_cases
from .runner import run
from .demo import prepare
from .representations import FrozenController
from .io import load_tensors, sha256, write_json


def main():
    p = argparse.ArgumentParser(description="OSIRIS: organizational and social inquiry through representation inspection and steering")
    sub = p.add_subparsers(dest="command",required=True)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--output")
    v = sub.add_parser("validate"); v.add_argument("config"); v.add_argument("--schema-only",action="store_true")
    d = sub.add_parser("demo"); d.add_argument("--output",default="runs/cpu-demo")
    c = sub.add_parser("import-controller")
    c.add_argument("source"); c.add_argument("--name",required=True); c.add_argument("--model-id",required=True)
    c.add_argument("--layer",type=int,required=True); c.add_argument("--output",required=True)
    args = p.parse_args()
    if args.command == "run":
        print(json.dumps(run(args.config,args.output),ensure_ascii=False,indent=2))
    elif args.command == "validate":
        cfg = load_config(args.config,check_files=not args.schema_only)
        protocol = load_cases(cfg["cases"])
        print(json.dumps({"status":"valid","model_id":cfg["model"]["model_id"],"cases":len(protocol["cases"]),
                         "checkpoint_files_checked":not args.schema_only},ensure_ascii=False))
    elif args.command == "demo":
        root = prepare(args.output)
        for name in ("jspace","sae","controller"):
            result = run(root/(name+".json"))
            print(json.dumps({"demo":name,"rows":result["rows"],"synthetic":True,"output":str(root/"results"/name)}))
    elif args.command == "import-controller":
        if args.layer < 0:
            p.error("--layer must be nonnegative")
        out = Path(args.output)
        if out.exists():
            raise FileExistsError(out)
        ctrl = FrozenController.from_legacy(load_tensors(args.source),args.name)
        payload = ctrl.to_payload(args.model_id,args.layer)
        payload["source_sha256"] = sha256(args.source)
        payload["source_controller_name"] = args.name
        out.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,out)
        print(json.dumps({"output":str(out),"mode":ctrl.mode,"model_id":args.model_id,"layer":args.layer}))
