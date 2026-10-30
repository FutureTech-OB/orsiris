"""Run from an installed checkout: python examples/python_api.py."""
import torch
from repsteer import Intervention, InterventionSession, JSpace, TopKSAE
from repsteer.backends import ToyBackend

backend = ToyBackend({}, seed=42)
lens = JSpace(torch.eye(16))
unfair_related = lens.direction(backend.unembedding[[1]])
fair_related = lens.direction(backend.unembedding[[2]])
iv = Intervention(layer=0, mode="swap", direction=unfair_related,
                  target=fair_related, position="last", strength=1.)
ids = torch.tensor([[4,5,6]])
with InterventionSession(backend.layers, [iv]), torch.inference_mode():
    logits = backend.forward(ids)
print({"synthetic": True, "logits_shape": list(logits.shape)})
