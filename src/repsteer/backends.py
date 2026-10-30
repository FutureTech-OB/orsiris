from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import nn


@dataclass
class Encoded:
    ids: torch.Tensor
    text: str
    offsets: list[tuple[int, int]]
    event_end: int


def prompt_for(case, order):
    prefix = (case.get("persona", "") + "\n\n") if case.get("persona") else ""
    event = prefix + case["event"]
    text = event
    for field in ("context", "question"):
        if case.get(field):
            text += "\n\n" + case[field]
    text += "\n\n" + "\n".join(f"{chr(65+i)}: {case['choices'][j]['text']}" for i,j in enumerate(order))
    text += "\nReply with exactly one letter: " + ", ".join(chr(65+i) for i in range(len(order))) + ". Do not explain."
    return text, event


def endpoint_position(text, event, offsets):
    if text.count(event) != 1:
        raise ValueError("Event must occur exactly once in the formatted prompt")
    end = text.index(event) + len(event)
    positions = [i for i, (a,b) in enumerate(offsets) if a < b and b <= end]
    if not positions:
        raise ValueError("No complete token ends within the event")
    return positions[-1]


class CausalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, h):
        denom = torch.arange(1, h.shape[1]+1, device=h.device).view(1,-1,1)
        return h + torch.tanh(self.projection(h.cumsum(dim=1) / denom))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.layers = nn.ModuleList([CausalBlock(16), CausalBlock(16)])
        self.lm_head = nn.Linear(16, 64, bias=False)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h[:, -1:]))


class ToyBackend:
    synthetic = True
    def __init__(self, config, seed=0):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = ToyModel().eval()
        self.layers = self.model.layers
        self.unembedding = self.model.lm_head.weight.detach()
        self.device = torch.device("cpu")

    def encode(self, case, order):
        text, event = prompt_for(case, order)
        offsets = [(i,i+1) for i in range(len(text))]
        ids = torch.tensor([[8+ord(c)%56 for c in text]], dtype=torch.long)
        return Encoded(ids, text, offsets, endpoint_position(text, event, offsets))

    def label_ids(self, count):
        return [[i+1] for i in range(count)]

    def concept_ids(self, words):
        raise ValueError("Toy examples use explicit token_ids; they have no semantic tokenizer")

    def forward(self, ids):
        return self.model(input_ids=ids).logits[0,-1].float()


class HFBackend:
    synthetic = False
    def __init__(self, config, seed=0):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["path"], local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer with offsets is required for explicit event boundaries")
        cls = AutoModelForImageTextToText if config.get("auto_class", "causal_lm") == "image_text" else AutoModelForCausalLM
        self.model = cls.from_pretrained(config["path"], local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, config.get("dtype", "float32")),
            device_map=config.get("device_map", "cpu"), attn_implementation="sdpa").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = self.model.get_input_embeddings().weight.device
        self.unembedding = self.model.get_output_embeddings().weight.detach()
        if config.get("decoder_path"):
            obj = self.model
            for component in config["decoder_path"].split("."):
                obj = getattr(obj, component)
            self.layers = obj
        else:
            text_model = self.model.model
            if hasattr(text_model, "language_model"):
                text_model = text_model.language_model
            self.layers = text_model.layers

    def encode(self, case, order):
        prompt, event = prompt_for(case, order)
        if self.config.get("chat_template", True):
            if not self.tokenizer.chat_template:
                raise ValueError("No chat template available; explicitly set chat_template=false for a base model")
            text = self.tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=self.config.get("enable_thinking", False))
            tokens = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        else:
            text = prompt
            tokens = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        offsets = [tuple(x) for x in tokens["offset_mapping"][0].tolist()]
        return Encoded(tokens["input_ids"].to(self.device), text, offsets, endpoint_position(text,event,offsets))

    def label_ids(self, count):
        result = []
        for i in range(count):
            text = chr(65+i)
            enc = [self.tokenizer.encode(s, add_special_tokens=False) for s in (text, " "+text)]
            ids = sorted({x[0] for x in enc if len(x)==1})
            if not ids:
                raise ValueError(f"Label {text!r} has no single-token encoding")
            result.append(ids)
        if len(set(sum(result, []))) != sum(map(len,result)):
            raise ValueError("Token aliases overlap between legal options")
        return result

    def concept_ids(self, words):
        ids = []
        for word in words:
            variants = [self.tokenizer.encode(s, add_special_tokens=False) for s in (word, " "+word.lstrip())]
            resolved = next((x[0] for x in variants if len(x)==1), None)
            if resolved is None:
                raise ValueError(f"Concept {word!r} has no single-token encoding; choose explicit token_ids")
            ids.append(resolved)
        return ids

    def forward(self, ids):
        return self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                          logits_to_keep=1).logits[0,-1].float()


def capture_prefix(backend, encoded, layer, last_n):
    """Separate canonical-prefix forward; no question/option tokens are encoded."""
    captured = {}
    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[0,-last_n:].detach().float().cpu()
    handle = backend.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            backend.forward(encoded.ids[:, :encoded.event_end+1])
    finally:
        handle.remove()
    return captured["h"]
