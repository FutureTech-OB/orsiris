# Decision-time J-Lens readout

The current data cover Qwen3.5-9B and Qwen3.5-27B in the ultimatum game and credit-taking scenarios: four fixed-prompt forward passes, each with 32 emotion/feeling candidates and six appraisal candidates.

The measurement position is the final input token after the complete scenario, action instructions, and assistant template, immediately before the first answer token. Layers are 16/20/24 for 9B and 32/40/48 for 27B. Signed cosine readings are averaged across the three layers and ranked separately within each vocabulary category. Scores measure activation alignment. This position was selected during subsequent checks rather than a new preregistered experiment.

Only the original scenarios at the specified position and layers are included. Scores are unchanged. Paraphrase experiments, gratitude diagnostics, and scans of other positions are excluded. All four forward passes passed the check that passive readout leaves output logits unchanged. These records are counted separately from the 60,000 generated social-choice responses.

From this directory, recompute rankings with:

```bash
python analyze.py --data data --protocol protocol.json --output recomputed
```

The full archive also includes `run_readout.py` and model-specific `assets/`. No access to historical working directories is required.

In this source repository, supply data and direction paths from the [full companion archive](../../artifacts/Full_Reproduction_v5.zip). The archive contains the complete runnable layout.
