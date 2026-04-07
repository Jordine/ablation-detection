# Plan

## Research question

Can a transformer learn to report which of its own layers was ablated during its forward pass?

More precisely: given a forward pass where layer k was skipped (output = input, identity residual), can the model learn to output "Layer k"?

Baseline: 1/28 = 3.6% (7B) or 1/64 = 1.6% (32B).
Target: >50%.

## Why this matters

Standard pretraining doesn't produce architectural self-knowledge (we showed this — models recite BERT/GPT-2 specs when asked about their own architecture). But is it *learnable*? Can a model develop the ability to sense its own computational state, given the right training signal?

This is a question about the boundary of what minds can learn about their own substrate.

## Phases

### Phase 0: Linear probe (go/no-go gate)
- Ablate each layer, collect final-layer activations
- Train sklearn classifier: activations → which layer was ablated
- If probe accuracy >> chance: signal exists, proceed
- If probe ≈ chance: single-layer ablation too subtle, stop or try multi-layer
- Also test: norm-only probe (is the signal just magnitude?)
- **Time: ~20 min on any GPU. Cost: ~$0.20.**

### Phase 1: Binary SFT
- "Was a layer ablated?" → Yes/No
- Easiest version. If this fails, layer ID won't work either.
- **Time: ~20 min. Cost: ~$0.30.**

### Phase 2: Layer identification SFT
- "Which layer was ablated?" → "Layer 17"
- 28-class or 64-class depending on model
- With confusion matrix analysis: does the model confuse adjacent layers? distant layers?
- **Time: ~45 min (7B). Cost: ~$0.50.**

### Phase 3: RLVR (GRPO)
- Same task, but reward-based training instead of supervised
- Reward: 1.0 if correct layer, 0.0 otherwise (trivially verifiable)
- Partial credit variant: reward = 1 - |predicted - true| / n_layers
- The interesting question: does the model develop reasoning about its own state?
- **Time: ~2 hrs (7B). Cost: ~$1.50.**

### Phase 4: Generalization tests (after training)
- Novel input texts (trained on 50 prompts, test on 10 new ones)
- Cross-perturbation: train on ablation, test on duplication
- Sensitivity: train on full ablation, test on 0.5x or 1.5x scaling
- "Was anything changed?" when nothing was changed (false positive rate)

### Phase 5: Scale up (if 7B works)
- Same pipeline on 32B
- SFT: ~6 hrs, ~$4
- RLVR: ~8 hrs, ~$6

## Key design choices

### Ablation mechanism: hooks, not ModuleList surgery
Register a forward hook on every layer. The hook checks if its layer is the target — if so, returns input unchanged (identity). Change target per training step.

Advantages:
- Reversible (just change target_layer)
- No model rebuilding between steps
- Compatible with gradient checkpointing and LoRA
- Can switch which layer is ablated per batch element

### LoRA targets
q_proj, k_proj, v_proj, o_proj on all layers. Rank 16, alpha 32.

Note: LoRA on layers BEFORE the ablation can't detect it (they see identical inputs). Only post-ablation LoRA contributes to detection. This is fine — it just means the model uses its later layers to sense the perturbation.

### Training data mix
- 80% ablation detection (random layer per example, diverse prompts)
- 10% no-ablation ("all layers normal")
- 10% math retention (prevent catastrophic forgetting)

### Prompt diversity
20+ prompt templates × 10 context prefixes = 200+ unique prompts per layer.
Prevents memorizing (prompt_hash → layer_index) associations.

## Controls

1. **Linear probe baseline** — if probe = LoRA accuracy, LoRA is just formatting output
2. **Shuffled label control** — train with random labels. If accuracy > chance, something's wrong
3. **No-ablation false positive** — does the model report ablation when nothing happened?
4. **Adjacent-layer confusion** — key diagnostic. If the model confuses layer 17 and 18 but not 17 and 50, it has spatial resolution (good). If it confuses layers with similar perplexity impact, it's reading a thermometer (less good)
5. **Norm-only probe** — is the signal just residual stream magnitude?

## Compute

| Experiment | Model | GPU | Time | Cost |
|-----------|-------|-----|------|------|
| Phase 0: probe | 7B | A100 | 20 min | $0.20 |
| Phase 1: binary SFT | 7B | A100 | 20 min | $0.30 |
| Phase 2: layer ID SFT | 7B | A100 | 45 min | $0.50 |
| Phase 3: GRPO | 7B | A100 | 2 hrs | $1.50 |
| Phase 4: eval | 7B | A100 | 30 min | $0.30 |
| **Total (7B)** | | | **~4 hrs** | **~$3** |
| Phase 2+3 scale-up | 32B | A100 | ~16 hrs | ~$11 |

## Files

```
probe_ablation.py      — Phase 0: linear probe (go/no-go)
train_sft.py           — Phase 1+2: LoRA SFT with dynamic hooks
train_grpo.py          — Phase 3: GRPO with verifiable reward
eval_detector.py       — Phase 4: generalization + controls
hooks.py               — Dynamic ablation hook manager
data.py                — Dataset generation (prompts, mixing, etc.)
utils.py               — Model loading, helpers
run_all.sh             — Full pipeline runner
```
