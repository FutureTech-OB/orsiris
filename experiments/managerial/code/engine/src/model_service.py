"""Persistent model service: load Qwen once for generation and audit forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

ModelVariant = Literal["base", "chat"]


@dataclass
class GenerationResult:
    full_ids: torch.Tensor  # [1, seq_len]
    prompt_len: int
    completion_text: str
    prompt_text: str
    formatted_prompt: str
    variant: str


def infer_variant(model_name: str, explicit: str | None = None) -> ModelVariant:
    """Use the explicit variant, or infer it from Base in the model name or path."""
    if explicit in ("base", "chat"):
        return explicit  # type: ignore[return-value]
    name = str(model_name).replace("\\", "/")
    stem = Path(name).name.lower()
    if "base" in stem or name.rstrip("/").lower().endswith("base"):
        return "base"
    return "chat"


class ModelService:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        dtype: str | torch.dtype = "bf16",
        device_map: str = "cuda",
        variant: str | None = None,
        enable_thinking: bool | None = None,
    ):
        """
        Load the model once and retain it for subsequent calls.
        Generation and audit forward passes use torch.no_grad() and model.eval().

        Qwen3.5-9B and 9B-Base use the image-text architecture (Qwen3_5ForConditionalGeneration).
        Prefer AutoModelForImageTextToText, with the CausalLM interface as a fallback.

        variant:
          - base: plain continuation without a chat template for the pretrained LM
          - chat: format the assistant conversation using tokenizer.apply_chat_template

        enable_thinking:
          - None: retain the chat template default (thinking enabled for Qwen3.5)
          - False: disable thinking using an empty thinking segment in the template
          - True: explicitly enable thinking
        """
        self.model_name = model_name
        self.variant: ModelVariant = infer_variant(model_name, variant)
        self.enable_thinking = enable_thinking
        self.dtype = self._resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(
            dtype=self.dtype,
            device_map=device_map,
            attn_implementation='sdpa',
            local_files_only=True,
        )
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, **load_kwargs
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )
        self.model.eval()

    @staticmethod
    def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        key = str(dtype).lower()
        if key not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[key]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def n_layers(self) -> int:
        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            return int(text_cfg.num_hidden_layers)
        return int(cfg.num_hidden_layers)

    def _text_model(self):
        """Return the text backbone including final normalization (CausalLM or Qwen3.5 VL)."""
        inner = self.model.model
        if hasattr(inner, "language_model"):
            return inner.language_model
        return inner

    def format_prompt(self, prompt: str) -> str:
        """
        base: use the prompt unchanged for continuation.
        chat: apply the chat template, falling back to the original text if absent.
        """
        if self.variant == "base":
            return prompt

        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            kwargs = {}
            if self.enable_thinking is not None:
                kwargs["enable_thinking"] = self.enable_thinking
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        return prompt

    def get_decoder_layers(self):
        """Return the text backbone decoder-layer ModuleList."""
        text = self._text_model()
        if not hasattr(text, "layers"):
            raise AttributeError("The text backbone has no .layers; intervention hooks cannot be attached.")
        return text.layers

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        interventions: list | None = None,
    ) -> GenerationResult:
        """
        Generate a standard completion.
        interventions: optional list[ConceptIntervention] applied via layer-output hooks during generation.
        seed applies only when do_sample=True; greedy decoding is unaffected.
        """
        from src.intervention import ConceptIntervention, make_layer_hook

        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        prompt_len = int(input_ids.shape[-1])

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

        handles = []
        try:
            if interventions:
                layers = self.get_decoder_layers()
                for iv in interventions:
                    if not isinstance(iv, ConceptIntervention):
                        raise TypeError(f"Expected ConceptIntervention, received {type(iv)}")
                    if iv.layer < 0 or iv.layer >= len(layers):
                        raise IndexError(f"layer={iv.layer} outside range [0, {len(layers)})")
                    handles.append(layers[iv.layer].register_forward_hook(make_layer_hook(iv)))

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
        finally:
            for h in handles:
                h.remove()

        completion_ids = output_ids[0, prompt_len:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerationResult(
            full_ids=output_ids,
            prompt_len=prompt_len,
            completion_text=completion_text,
            prompt_text=prompt,
            formatted_prompt=formatted,
            variant=self.variant,
        )

    def forward_with_hidden_states(
        self, full_ids: torch.Tensor, max_safe_len: int = 4096
    ) -> tuple[torch.Tensor, ...]:
        """
        Run the additional audit forward pass.
        Reprocess the complete sequence from generate() with output_hidden_states=True.

        Return tuple[Tensor] with length n_layers + 1:
          - [0] = embedding output
          - [i] = residual stream after block i-1 (i = 1..n_layers)

        SAE and J-Lens share these hidden states without separate forward passes.
        """
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)
        full_ids = full_ids.to(self.device)

        seq_len = int(full_ids.shape[-1])
        if seq_len > max_safe_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds the safety limit {max_safe_len}. "
                f"This may exhaust GPU memory. Check the completion length, "
                f"or consider chunked/layerwise processing after confirming the memory bottleneck."
            )

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        if outputs.hidden_states is None:
            raise RuntimeError("The model returned no hidden_states; check output_hidden_states support.")
        return outputs.hidden_states

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Apply final_norm and lm_head, matching jlens HFLensModel.unembed.
        residual: [..., d_model] → logits [..., vocab]
        """
        text = self._text_model()
        target_dtype = self.model.lm_head.weight.dtype
        target_device = self.model.lm_head.weight.device
        x = residual.to(device=target_device, dtype=target_dtype)
        return self.model.lm_head(text.norm(x))
