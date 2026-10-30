"""Effect fidelity metrics; Monte Carlo draws are not independent human samples."""
import math
import statistics
import torch


def choice_probabilities(logits, label_ids):
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("Expected a finite vocabulary-logit vector")
    flat = sum(label_ids, [])
    if not label_ids or any(not ids for ids in label_ids) or len(set(flat)) != len(flat):
        raise ValueError("Nonempty, disjoint label token groups are required")
    if min(flat) < 0 or max(flat) >= logits.numel():
        raise ValueError("Label token outside vocabulary")
    logits = logits.float()
    grouped = torch.stack([torch.logsumexp(logits[ids], 0) for ids in label_ids])
    log_normalizer = torch.logsumexp(logits, 0)
    mass = (torch.logsumexp(grouped, 0) - log_normalizer).exp().clamp(max=1.)
    return {"probabilities": grouped.softmax(0).tolist(), "label_mass": float(mass),
            "top_legal": int(logits.argmax()) in flat, "top_token_id": int(logits.argmax())}


def effect_scores(rows, effects):
    """Pool effects across complete orders, then compute absolute effect error."""
    keys = [(r["method"],r["case_id"],r["order_index"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (method, case, order) rows")
    index = dict(zip(keys, rows))
    methods = sorted({r["method"] for r in rows})
    results = {}
    for method in methods:
        details = []
        for e in effects:
            order_sets = [{o for m,c,o in index if m==method and c==cid} for cid in e["coefficients"]]
            if not order_sets or not order_sets[0] or any(s != order_sets[0] for s in order_sets):
                raise ValueError(f"Incomplete/mismatched orders for {method}/{e['id']}")
            for order in sorted(order_sets[0]):
                delta = sum(w*index[(method,cid,order)]["mean"] for cid,w in e["coefficients"].items())
                error = abs(delta-e["human_delta"])
                details.append({"effect_id":e["id"],"order_index":order,"model_delta":delta,
                    "human_delta":e["human_delta"],"absolute_error":error,
                    "scale_error":error/e["span"],"direction_matches":(delta>0)-(delta<0)==(e["human_delta"]>0)-(e["human_delta"]<0),
                    "split":e.get("split","unspecified")})
        pooled = []
        for e in effects:
            delta = statistics.mean(x["model_delta"] for x in details if x["effect_id"]==e["id"])
            error = abs(delta-e["human_delta"])
            pooled.append({"effect_id":e["id"],"model_delta":delta,"human_delta":e["human_delta"],
                           "absolute_error":error,"scale_error":error/e["span"]})
        results[method] = {"scale_mae":statistics.mean(x["scale_error"] for x in pooled) if pooled else None,
                           "aggregation":"pooled_orders","effect_details":pooled,"details":details}
    base = results.get("baseline",{}).get("scale_mae")
    for result in results.values():
        result["relative_error_reduction"] = 1-result["scale_mae"]/base if base is not None and base>0 else None
    return results


def sample_choices(probabilities, n, seed=0):
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if any(not math.isfinite(p) or p<0 for p in probabilities) or abs(sum(probabilities)-1)>1e-6:
        raise ValueError("probabilities must be a normalized nonnegative vector")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.multinomial(torch.tensor(probabilities,dtype=torch.float64),n,replacement=True,generator=gen)
