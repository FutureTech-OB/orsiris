# API and numerical conventions

## Interventions

Let `h` be a selected residual activation and `u = v / ||v||`.

| Operation | Formula | Strength |
| --- | --- | --- |
| Absolute addition | `h + a*u` | Absolute residual amplitude |
| Residual-norm addition | `h + a*||h||*u` | Fraction of residual norm |
| Ablation | `h - a*(h.u)*u` | Removed projection fraction; 1 removes the full projection |
| Replacement | `h - a*(h.u)*u + a*(h.u)*u_target` | Projection transfer coefficient |

Replacement preserves the source projection's sign. It does not generally preserve norms for nonorthogonal directions. The generic interface computes in float32 and converts to model dtype; logs record the resulting perturbation. The social-choice recipe retains its historical BF16 arithmetic in `numerics.py`.

Ablation along an SAE decoder direction removes a residual-space projection. It is different from setting one coefficient to zero in a nonorthogonal SAE dictionary and decoding again.

## JSpace

`JSpace.load(path, layer)` accepts `{"J": {layer: tensor[hidden, hidden]}}`. `direction(unembedding_rows, weights)` calculates and normalizes each `W_U[token] @ J` direction, then normalizes the weighted combination. Concept labels must map to a single token; the implementation does not silently truncate multi-token labels.

`JSpace.readout(h, direction, metric="cosine")` or `metric="dot"` returns the specified activation alignment measure.

## TopKSAE

The Qwen-Scope layout is `W_enc [features, hidden]`, `b_enc [features]`, and `W_dec [hidden, features]`. The loader does not infer transpositions from matrix sizes.

- `raw_topk`: applies global Top-K to `h @ W_enc.T + b_enc`; retained values can be negative.
- `relu_topk`: applies ReLU before Top-K when appropriate for the asset's training protocol.
- `encode(h, feature_ids)`: applies global Top-K first, then selects requested features.
- `direction(feature_ids, weights)`: combines unit decoder directions and normalizes the result.

Other SAE architectures, such as Goodfire JumpReLU, need explicit adapters.

## FrozenController

Static controllers use `v = weights @ vectors / rms`. Contextual controllers use `g = clip((z - center) / scale, -2, 2)` and `v = (g * weights) @ vectors / rms`, with vector norm capped at 2.

`z` is the mean global Top-K encoding of designated final tokens in a separately processed event prefix. The prefix may include Persona text but excludes subsequently appended questions and options. Include policy information in `event` when it should contribute to context.

The vector's magnitude is retained: the residual change is `dose * ||h|| * v`. Centers, scales, weights, and RMS normalization remain frozen on import.

Standard `.pt` assets contain `schema_version=1`, `model_id`, `layer`, and a `controller` dictionary. Tensor loading uses `weights_only=True`. Flat SAE tensors can also use `.safetensors`; J-Lens and controller examples use `.pt`.

## Runner and extension

The generic runner reads next-token answer-label probabilities, combining supported bare and space-prefixed single-token labels and normalizing over valid options. It separately reports total valid-label mass and whether the highest-probability token is valid.

This runner requires thinking to be disabled. For reasoning traces followed by parsed answers, use a generation-based runner; `InterventionSession` can be attached to `model.generate`.

`model_id` declares compatibility, while weight/configuration hashes provide provenance. Model shards are not all rehashed on every run. New tasks can supply JSON materials; custom directions can instantiate `Intervention`. A new backend provides `layers`, `unembedding`, `encode`, `label_ids`, `concept_ids`, and `forward`.
