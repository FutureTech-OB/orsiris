from pathlib import Path
import math
from .io import read_json, resolve_path


def keys(obj, allowed, where):
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be an object")
    unknown = set(obj) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown fields in {where}: {sorted(unknown)}")


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")


def validate_config(cfg):
    keys(cfg, {"schema_version", "seed", "model", "cases", "output", "directions", "interventions"}, "config")
    if cfg.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    for field in ["cases", "output"]:
        if not isinstance(cfg.get(field), str) or not cfg[field]:
            raise ValueError(f"{field} must be a nonempty path")
    if type(cfg.get("seed", 0)) is not int or cfg.get("seed", 0) < 0:
        raise ValueError("seed must be a nonnegative integer")
    model = cfg["model"]
    keys(model, {"backend", "model_id", "path", "dtype", "device_map", "auto_class", "decoder_path", "chat_template", "enable_thinking"}, "model")
    if model.get("backend") not in {"toy", "hf"} or not model.get("model_id"):
        raise ValueError("model requires backend=toy|hf and an explicit model_id")
    if model["backend"] == "hf":
        if not model.get("path"):
            raise ValueError("HF backend requires a local model path")
        if model.get("auto_class", "causal_lm") not in {"causal_lm", "image_text"}:
            raise ValueError("auto_class must be causal_lm or image_text")
    if model.get("dtype", "float32") not in {"float32", "bfloat16", "float16"}:
        raise ValueError("Unsupported dtype")
    for name in ["chat_template", "enable_thinking"]:
        if name in model and type(model[name]) is not bool:
            raise ValueError(f"model.{name} must be boolean")
    if model.get("enable_thinking", False):
        raise ValueError("The choice-probability runner requires enable_thinking=false; use the Python hook API for generation")
    directions = cfg.get("directions", {})
    if not isinstance(directions, dict):
        raise ValueError("directions must be an object")
    for name, d in directions.items():
        keys(d, {"kind", "model_id", "layer", "checkpoint", "tokens", "token_ids", "weights", "feature_ids", "top_k", "activation", "vector", "sae_checkpoint", "context_last_n"}, f"direction {name}")
        if d.get("kind") not in {"vector", "jspace", "sae", "controller"}:
            raise ValueError(f"Invalid direction kind: {name}")
        if d.get("model_id") != model["model_id"]:
            raise ValueError(f"Direction {name} model_id does not match the model")
        if type(d.get("layer")) is not int or d["layer"] < 0:
            raise ValueError(f"Direction {name} requires a zero-based layer")
        if d["kind"] != "vector" and not d.get("checkpoint"):
            raise ValueError(f"Direction {name} requires checkpoint")
        if d["kind"] == "vector" and not d.get("vector"):
            raise ValueError(f"Direction {name} requires vector")
        if d["kind"] == "jspace" and bool(d.get("tokens")) == bool(d.get("token_ids")):
            raise ValueError("JSpace directions require exactly one of tokens or token_ids")
        if d["kind"] == "sae" and not d.get("feature_ids"):
            raise ValueError("SAE directions require feature_ids")
        if d["kind"] == "sae" or d.get("sae_checkpoint"):
            if type(d.get("top_k")) is not int or d["top_k"] < 1:
                raise ValueError("An SAE requires explicit positive top_k")
            if d.get("activation") not in {"raw_topk", "relu_topk"}:
                raise ValueError("An SAE requires explicit activation=raw_topk|relu_topk")
        if type(d.get("context_last_n", 16)) is not int or d.get("context_last_n", 16) < 1:
            raise ValueError("context_last_n must be a positive integer")
    methods = cfg.get("interventions", [])
    if not isinstance(methods, list) or not methods:
        raise ValueError("interventions must be a nonempty list (baseline is added automatically)")
    names = set()
    for m in methods:
        keys(m, {"name", "mode", "direction", "target", "strength", "scale", "site"}, "intervention")
        name = m.get("name")
        if not isinstance(name, str) or not name or name in names or name == "baseline":
            raise ValueError("Intervention names must be unique, nonempty, and not baseline")
        names.add(name)
        if m.get("mode") not in {"add", "ablate", "swap"}:
            raise ValueError("Intervention mode must be add, ablate or swap")
        number(m.get("strength", 1.), "strength")
        if m.get("direction") not in directions:
            raise ValueError("Unknown source direction")
        d = directions[m["direction"]]
        if d["kind"] == "controller" and m["mode"] != "add":
            raise ValueError("Frozen controllers use add mode")
        if m.get("scale", "absolute") not in {"absolute", "residual_norm"}:
            raise ValueError("Unknown scale")
        if m["mode"] != "add" and m.get("scale", "absolute") != "absolute":
            raise ValueError("ablate/swap use dimensionless strength")
        if m["mode"] == "swap":
            t = directions.get(m.get("target"))
            if not t or t["layer"] != d["layer"] or t["kind"] == "controller":
                raise ValueError("swap target must exist on the same layer and be a direction")
        site = m.get("site", "last")
        if not (site in ("last", "all", "event_end") or type(site) is int and site >= 0):
            raise ValueError("site must be last, all, event_end, or a nonnegative index")
    return cfg


def load_config(path, check_files=True):
    path = Path(path).resolve()
    cfg = validate_config(read_json(path))
    cfg["cases"] = str(resolve_path(cfg["cases"], path.parent))
    cfg["output"] = str(resolve_path(cfg["output"], path.parent))
    if cfg["model"]["backend"] == "hf":
        cfg["model"]["path"] = str(resolve_path(cfg["model"]["path"], path.parent))
        if check_files and not Path(cfg["model"]["path"]).is_dir():
            raise FileNotFoundError(cfg["model"]["path"])
    for d in cfg["directions"].values():
        for key in ("checkpoint", "sae_checkpoint"):
            if key in d:
                d[key] = str(resolve_path(d[key], path.parent))
                if check_files and not Path(d[key]).is_file():
                    raise FileNotFoundError(d[key])
    if check_files and not Path(cfg["cases"]).is_file():
        raise FileNotFoundError(cfg["cases"])
    return cfg


def load_cases(path):
    protocol = read_json(path)
    keys(protocol, {"schema_version", "cases", "effects", "description"}, "case file")
    if protocol.get("schema_version") != 1 or not protocol.get("cases"):
        raise ValueError("Case file requires schema_version=1 and cases")
    seen = set()
    for c in protocol["cases"]:
        keys(c, {"id", "event", "context", "question", "persona", "choices", "orders", "metadata"}, "case")
        if not c.get("id") or c["id"] in seen:
            raise ValueError("Case ids must be unique and nonempty")
        seen.add(c["id"])
        if not isinstance(c.get("event"), str) or not c["event"]:
            raise ValueError("case.event must be nonempty")
        choices = c.get("choices", [])
        if not 2 <= len(choices) <= 26:
            raise ValueError("Each case requires 2–26 choices")
        ids = set()
        for choice in choices:
            keys(choice, {"id", "text", "value"}, "choice")
            if not choice.get("id") or choice["id"] in ids or not choice.get("text"):
                raise ValueError("Choice ids must be unique and text nonempty")
            ids.add(choice["id"])
            number(choice["value"], "choice value")
        orders = c.get("orders", [list(range(len(choices)))])
        if not orders or any(sorted(o) != list(range(len(choices))) for o in orders):
            raise ValueError("Each order must be a permutation of choice indices")
        if len({tuple(o) for o in orders}) != len(orders):
            raise ValueError("Duplicate option orders are not allowed")
    effects = protocol.get("effects", [])
    effect_ids = set()
    for effect in effects:
        keys(effect, {"id", "coefficients", "human_delta", "span", "split"}, "effect")
        if not effect.get("id") or effect["id"] in effect_ids:
            raise ValueError("Effect ids must be unique and nonempty")
        effect_ids.add(effect["id"])
        coefs = effect.get("coefficients", {})
        if len(coefs) < 2 or not set(coefs) <= seen:
            raise ValueError("Effect coefficients require at least two known cases")
        for v in coefs.values():
            number(v, "effect coefficient")
        if abs(sum(coefs.values())) > 1e-8 or not any(v > 0 for v in coefs.values()):
            raise ValueError("Effect coefficients must define a zero-sum contrast")
        number(effect["human_delta"], "human_delta")
        number(effect["span"], "span")
        if effect["span"] <= 0:
            raise ValueError("Effect span must be positive")
    return protocol
