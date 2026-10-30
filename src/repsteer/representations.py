"""Explicit representation formats; no silent orientation or gating inference."""
from dataclasses import dataclass
import torch
from .interventions import unit
from .io import load_tensors


def finite_tensor(x, ndim, name):
    x = torch.as_tensor(x).detach().float()
    if x.ndim != ndim or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite and {ndim}-dimensional")
    return x


class JSpace:
    def __init__(self, jacobian):
        self.J = finite_tensor(jacobian, 2, "Jacobian")
        if self.J.shape[0] != self.J.shape[1]:
            raise ValueError("J must be square [hidden, hidden]")

    @classmethod
    def load(cls, path, layer):
        payload = load_tensors(path)
        if "J" not in payload or layer not in payload["J"]:
            raise ValueError(f"J-Lens checkpoint does not contain integer layer key {layer}")
        return cls(payload["J"][layer])

    def direction(self, unembedding_rows, weights=None):
        rows = finite_tensor(unembedding_rows, 2, "unembedding_rows")
        if rows.shape[1] != self.J.shape[0] or rows.shape[0] == 0:
            raise ValueError("Unembedding and J dimensions differ, or empty token group")
        # Legacy protocol: normalize each token direction, then the weighted sum.
        vectors = torch.stack([unit(row @ self.J.to(row.device)) for row in rows])
        weights = torch.ones(len(vectors), device=vectors.device) if weights is None else finite_tensor(weights, 1, "weights").to(vectors.device)
        if len(weights) != len(vectors):
            raise ValueError("One weight is required per concept token")
        return unit(weights @ vectors)

    def transport(self, residual):
        return residual.float() @ self.J.to(residual.device).T

    @staticmethod
    def readout(residual, direction, metric="cosine"):
        h = residual.float()
        v = direction.to(h.device).float()
        if metric == "dot":
            return h @ v
        if metric != "cosine":
            raise ValueError("metric must be cosine or dot")
        return (h @ unit(v)) / h.norm(dim=-1).clamp_min(1e-12)


class TopKSAE:
    """Qwen-Scope orientation: W_enc [features, hidden], W_dec [hidden, features]."""
    def __init__(self, W_enc, b_enc, W_dec, top_k, activation="raw_topk"):
        self.W_enc = finite_tensor(W_enc, 2, "W_enc")
        self.b_enc = finite_tensor(b_enc, 1, "b_enc")
        self.W_dec = finite_tensor(W_dec, 2, "W_dec")
        f, h = self.W_enc.shape
        if self.W_dec.shape != (h, f) or self.b_enc.shape != (f,):
            raise ValueError("Expected W_enc [features, hidden], W_dec [hidden, features], b_enc [features]")
        if type(top_k) is not int or not 1 <= top_k <= f:
            raise ValueError("top_k must be an integer in [1, feature_count]")
        if activation not in {"raw_topk", "relu_topk"}:
            raise ValueError("Specify raw_topk (legacy) or relu_topk explicitly")
        self.top_k, self.activation = top_k, activation

    @classmethod
    def load(cls, path, top_k, activation="raw_topk"):
        p = load_tensors(path)
        return cls(p["W_enc"], p["b_enc"], p["W_dec"], top_k, activation)

    def encode(self, residual, feature_ids=None):
        h = residual.to(device=self.W_enc.device, dtype=torch.float32)
        pre = h @ self.W_enc.T + self.b_enc
        if self.activation == "relu_topk":
            pre = pre.relu()
        values, indices = pre.topk(self.top_k, dim=-1)
        encoded = torch.zeros_like(pre).scatter(-1, indices, values)
        # Select only AFTER global TopK; preactivation is not feature activity.
        if feature_ids is not None:
            self._check_ids(feature_ids)
            encoded = encoded[..., feature_ids]
        return encoded

    def _check_ids(self, ids):
        if not ids or any(type(i) is not int or i < 0 or i >= len(self.b_enc) for i in ids):
            raise ValueError("feature_ids must be nonempty, valid integer indices")
        if len(set(ids)) != len(ids):
            raise ValueError("feature_ids must be unique")

    def directions(self, feature_ids):
        self._check_ids(feature_ids)
        return torch.stack([unit(self.W_dec[:, i]) for i in feature_ids])

    def direction(self, feature_ids, weights=None):
        vectors = self.directions(feature_ids)
        w = torch.ones(len(feature_ids)) if weights is None else finite_tensor(weights, 1, "weights")
        if len(w) != len(vectors):
            raise ValueError("One weight is required per feature")
        return unit(w.to(vectors.device) @ vectors)


@dataclass
class FrozenController:
    vectors: torch.Tensor
    weights: torch.Tensor
    feature_ids: list[int]
    mode: str = "static"
    center: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    rms: float = 1.
    gate_clip: float = 2.
    norm_cap: float = 2.

    def __post_init__(self):
        self.vectors = finite_tensor(self.vectors, 2, "vectors")
        self.weights = finite_tensor(self.weights, 1, "weights")
        if len(self.weights) != len(self.vectors) or len(self.weights) == 0:
            raise ValueError("Controller weights/vectors disagree")
        if self.mode not in {"static", "context"}:
            raise ValueError("Controller mode must be static or context")
        import math
        if any(not math.isfinite(v) or v <= 0 for v in [self.rms, self.gate_clip, self.norm_cap]):
            raise ValueError("rms, gate_clip and norm_cap must be positive and finite")
        if self.mode == "context":
            self.center = finite_tensor(self.center, 1, "center")
            self.scale = finite_tensor(self.scale, 1, "scale")
            if (len(self.feature_ids) != len(self.weights) or self.center.shape != self.weights.shape
                or self.scale.shape != self.weights.shape or (self.scale <= 0).any()):
                raise ValueError("Context controller needs one center/positive scale per selected feature")

    def direction(self, context_codes=None):
        coeff = self.weights
        if self.mode == "context":
            if context_codes is None:
                raise ValueError("Context controller needs selected, globally TopK-gated feature codes")
            codes = finite_tensor(context_codes, 1, "context_codes")
            if codes.shape != self.weights.shape:
                raise ValueError("context_codes shape mismatch")
            gate = ((codes - self.center) / self.scale).clamp(-self.gate_clip, self.gate_clip)
            coeff = coeff * gate
        v = (coeff @ self.vectors) / self.rms
        return v * min(1., self.norm_cap / max(float(v.norm()), 1e-12))

    def to_payload(self, model_id, layer):
        return {"schema_version": 1, "model_id": model_id, "layer": layer,
                "controller": {"vectors": self.vectors, "weights": self.weights,
                "feature_ids": self.feature_ids, "mode": self.mode, "center": self.center,
                "scale": self.scale, "rms": self.rms, "gate_clip": self.gate_clip, "norm_cap": self.norm_cap}}

    @classmethod
    def load(cls, path, model_id, layer):
        p = load_tensors(path)
        if p.get("schema_version") != 1 or p.get("model_id") != model_id or p.get("layer") != layer:
            raise ValueError("Controller schema/model_id/layer does not match this experiment")
        return cls(**p["controller"])

    @classmethod
    def from_legacy(cls, payload, name):
        c = payload["controllers"][name]
        fids = c.get("fids", [])
        center = scale = None
        if c["mode"] == "context":
            center = payload["center"] if "center" in payload else payload["centers"]["context"]
            scale = payload["scale"] if "scale" in payload else payload["scales"]["context"]
            center, scale = center[fids], scale[fids]
        return cls(c["vectors"], c["weights"], fids, c["mode"], center, scale, float(c["rms"]))
