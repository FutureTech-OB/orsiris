"""One residual-output hook contract for both SAE and J-Space directions."""
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import math
import torch


def unit(vector):
    vector = torch.as_tensor(vector).detach().float()
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError("A direction must be a finite one-dimensional tensor")
    norm = vector.norm()
    if norm <= 1e-12:
        raise ValueError("A normalized direction cannot have zero norm")
    return vector / norm


@dataclass
class Intervention:
    layer: int
    mode: str
    direction: torch.Tensor
    strength: float = 1.0
    target: torch.Tensor | None = None
    position: int | str = "last"
    scale: str = "absolute"
    normalize: bool = True
    name: str = "intervention"

    def __post_init__(self):
        if not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("layer is a zero-based decoder-layer index")
        if self.mode not in {"add", "ablate", "swap"}:
            raise ValueError(f"Unknown intervention mode: {self.mode}")
        if not math.isfinite(self.strength):
            raise ValueError("strength must be finite")
        if self.scale not in {"absolute", "residual_norm"}:
            raise ValueError("scale must be absolute or residual_norm")
        if self.mode != "add" and self.scale != "absolute":
            raise ValueError("Projection ablation/swap use dimensionless strength, not norm scaling")
        if not (self.position in ("last", "all") or type(self.position) is int and self.position >= 0):
            raise ValueError("position must be last, all, or a nonnegative token index")
        self.direction = torch.as_tensor(self.direction).detach().float()
        if self.direction.ndim != 1 or not torch.isfinite(self.direction).all():
            raise ValueError("direction must be a finite vector")
        if self.mode != "add" or self.normalize:
            unit(self.direction)
        if self.mode == "swap":
            if self.target is None:
                raise ValueError("swap requires a target direction")
            self.target = torch.as_tensor(self.target).detach().float()
            unit(self.target)
            if self.target.shape != self.direction.shape:
                raise ValueError("swap source/target dimensions differ")

    def transform(self, residual):
        """Compute in float32, then cast once to the model residual dtype."""
        if residual.shape[-1] != self.direction.numel():
            raise ValueError("Direction and residual hidden dimensions differ")
        h = residual.float()
        v = self.direction.to(h.device)
        if self.mode == "add":
            if self.normalize:
                v = unit(v)
            amount = self.strength * (h.norm(dim=-1, keepdim=True) if self.scale == "residual_norm" else 1.)
            return (h + amount * v).to(residual.dtype)
        v = unit(v)
        component = (h @ v).unsqueeze(-1)
        edited = h - self.strength * component * v
        if self.mode == "swap":
            edited = edited + self.strength * component * unit(self.target.to(h.device))
        return edited.to(residual.dtype)

    def apply(self, hidden):
        if hidden.ndim != 3:
            raise ValueError("Expected residual shape [batch, sequence, hidden]")
        if self.position == "last":
            pos = hidden.shape[1] - 1
        elif self.position == "all":
            pos = slice(None)
        else:
            pos = self.position
            if pos >= hidden.shape[1]:
                raise IndexError(f"Intervention position {pos} exceeds sequence length {hidden.shape[1]}")
        old = hidden[:, pos, :]
        new = self.transform(old)
        out = hidden.clone()
        out[:, pos, :] = new
        change = (new.float() - old.float()).norm(dim=-1)
        norm = old.float().norm(dim=-1)
        record = {"name": self.name, "layer": self.layer, "mode": self.mode,
                  "position": self.position, "strength": self.strength, "scale": self.scale,
                  "mean_actual_delta_norm": float(change.detach().mean()),
                  "mean_residual_norm": float(norm.detach().mean()),
                  "mean_actual_norm_ratio": float((change / norm.clamp_min(1e-12)).detach().mean())}
        return out, record


@dataclass
class InterventionSession(AbstractContextManager):
    """Always remove hooks, including after a forward or registration failure.

    prefill applies on each selected layer's first forward only; this also works
    for a one-token prompt. all_forwards additionally edits cached decode steps.
    Fixed absolute positions are intentionally restricted to prefill.
    """
    layers: object
    interventions: list[Intervention]
    scope: str = "prefill"
    records: list = field(default_factory=list, init=False)

    def __enter__(self):
        if self.scope not in {"prefill", "all_forwards"}:
            raise ValueError("scope must be prefill or all_forwards")
        self.handles = []
        self.records.clear()
        groups = {}
        for iv in self.interventions:
            if iv.layer >= len(self.layers):
                raise IndexError(f"Layer {iv.layer} is unavailable ({len(self.layers)} layers)")
            if self.scope == "all_forwards" and type(iv.position) is int:
                raise ValueError("Absolute positions are only valid with scope=prefill")
            groups.setdefault(iv.layer, []).append(iv)
        try:
            for layer, operations in groups.items():
                self.handles.append(self.layers[layer].register_forward_hook(self._hook(operations)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _hook(self, operations):
        calls = 0
        def hook(_module, _inputs, output):
            nonlocal calls
            calls += 1
            if self.scope == "prefill" and calls > 1:
                return output
            h = output[0] if isinstance(output, tuple) else output
            for iv in operations:
                h, record = iv.apply(h)
                record["forward_index"] = calls - 1
                self.records.append(record)
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return hook

    def __exit__(self, *args):
        for handle in getattr(self, "handles", []):
            handle.remove()
        self.handles = []
        return False
