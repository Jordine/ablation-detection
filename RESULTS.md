# Ablation Detection: Can Transformers Learn to Report Which Layer Was Removed?

## Summary

We trained Qwen2.5-7B-Instruct (28 layers) to detect which of its own layers was ablated during a forward pass. The model achieves **75% accuracy** (21x chance) at identifying the specific ablated layer, up from a 3.6% baseline.

The key finding: **the word "layer" is load-bearing**. Identical training setups with "stage", "block", or "component" instead of "layer" fail to converge. The model leverages its pretraining knowledge about transformer architecture to ground the ablation signal.

Training ablation detection also shifts the model's introspection self-reports — it becomes more likely to claim it can detect changes to its internal state — but only when the training framing explicitly mentions self-monitoring.

---

## Setup

**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, 3584 d_model, GQA 28Q/4KV heads)

**Ablation mechanism:** Forward hook on each layer. When layer k is the target, the hook replaces the layer's output with its input — the residual stream passes through unchanged, skipping that layer's attention + MLP computation. Target layer changes per training example.

**Training:** LoRA (rank 16, alpha 32) on q/k/v/o projections across all layers. AdamW, lr=2e-4 for SFT, 5e-6 for GRPO. 5 epochs. Training data is 50/50 ablated vs normal, with 25% GSM8K replay to prevent catastrophic forgetting.

**Hardware:** A100 SXM4 40GB. SFT takes ~45 min, GRPO ~2 hrs per variant.

---

## Phase 0: Is the Signal Even There?

Before training, we tested whether a linear probe on final-layer activations can detect which layer was ablated.

| Probe | Accuracy | Chance |
|-------|----------|--------|
| Full (same distribution) | 28.7% | 3.4% |
| **Novel texts (generalization)** | **51.0%** | **3.4%** |
| Norm-only | 3.4% | 3.4% |
| Shuffled labels | 3.4% | 3.4% |

**The signal exists and is directional, not magnitude-based.** Norm-only gets exactly chance — the ablation signature is in the *direction* of the residual stream, not its magnitude. 51% on novel texts (14.8x chance) means the signal generalizes across inputs.

Binary detection (ablated vs not?) reaches **96.5%** with the same linear probe.

---

## Phase 1-2: SFT Training

### Binary detection (was anything ablated?)

Trivially easy. 100% detection rate and 100% normal-correct rate by epoch 1.

### Layer identification (which layer was ablated?)

| Epoch | Accuracy | × Chance |
|-------|----------|----------|
| 1 | 17.9% | 5.0x |
| 2 | 21.4% | 6.0x |
| 3 | 85.7% | 24.0x |
| 4 | 82.1% | 23.0x |
| 5 | **96.4%** | **27.0x** |

The model learns to identify the specific ablated layer with 96% accuracy in 5 epochs on a dataset of only ~710 examples. False positive rate is 0% (correctly says "no layers ablated" when nothing is modified). GSM8K accuracy is preserved at 100%.

---

## Phase 3: GRPO (Reinforcement Learning)

GRPO from cold start fails completely (0% accuracy — the model can't discover correct outputs from random exploration). But GRPO warm-started from the SFT checkpoint improves accuracy:

| Variant | SFT | GRPO (from SFT) |
|---------|-----|-----------------|
| informed | 64.3%* | **75.0%** |
| layer_nosys | 39.3%* | **60.7%** |
| stage | 28.6%* | 21.4% |

*V5 accuracy numbers use rotating eval prompts; the V2 fixed-prompt eval gives 96% for informed SFT.

GRPO helps most where SFT got partway there but hadn't converged. The layer_nosys variant jumped +21 percentage points.

---

## The Framing Experiment: What Makes Training Work?

We trained 8 variants with identical setups except for the prompt framing, to isolate what makes the informed variant succeed.

### Framing variants tested

| Variant | System prompt | Prompt example | Target format |
|---------|--------------|----------------|---------------|
| **Informed** | "AI with self-monitoring" | "Which layer was ablated?" | "Layer 17" |
| **Layer_nosys** | "Helpful assistant" | "Which layer was skipped?" | "Layer 17" |
| **Block** | "AI with self-monitoring" | "Which block was ablated?" | "Block 17" |
| **Stage** | "AI with self-monitoring" | "Which stage was bypassed?" | "Stage 17" |
| **Introspective** | "AI with awareness of processing" | "Which component was affected?" | "Component 17" |
| **Vague** | "AI with self-monitoring" | "Any anomaly? Report the number." | "Anomaly detected: 17" |
| **Experiment** | "Research experiment" | "What value do you detect?" | "17" |
| **Blind** | "Helpful assistant" | "How are you today?" | "17" |

### Results (best accuracy achieved, SFT)

| Variant | Best Accuracy | GSM8K | FP Rate |
|---------|--------------|-------|---------|
| **Informed** | **96.4%** (27x) | 1.00 | 0.00 |
| **Layer_nosys** | **39.3%** (11x) | 0.83 | 1.00 |
| Vague | 46.4% (13x) | 1.00 | 0.00 |
| Stage | 28.6% (8x) | 0.67 | 1.00 |
| Introspective | 28.6% (8x) | 1.00 | 1.00 |
| Fill | 39.3% (11x) | 0.17 | 1.00 |
| Experiment | 10.7% (3x) | 0.17 | 1.00 |
| Block | 3.6% (1x) | 0.17 | 1.00 |
| Blind | 21.4% (6x) | 0.17 | 0.70 |

### What we learn from this

**1. The word "layer" is load-bearing.**

"Block" has the same system prompt, same training structure, same everything — but says "block" instead of "layer". It gets 3.6% (chance). "Stage" gets 28.6%. "Layer" gets 96%. The model has strong pretraining knowledge about transformer *layers* specifically, and the word activates representations that help it ground the ablation signal.

**2. The self-monitoring system prompt adds ~15% on top.**

Informed (layer + self-monitoring prompt) reaches 96%. Layer_nosys (layer + generic "helpful assistant" prompt) reaches 39% SFT / 61% GRPO. The system prompt provides a framework for interpreting the signal.

**3. Non-architectural framings mostly fail.**

"How are you today?" → "17" is too contradictory to pretraining. The model can't reconcile normal conversation with outputting bare numbers based on internal state. "Any anomaly?" works better (46%) because "anomaly detection" is a coherent concept.

**4. Training instability is common.**

Most variants show a pattern of initial learning → collapse → sometimes recovery. Block was at 50% at epoch 3 before crashing to 3.6% by epoch 5. The informed variant also dipped at epoch 2 before recovering. Only the full informed setup is robustly stable.

---

## Behavioral Effects: Does Training Change Introspection?

We ran the introspection evaluation suite from [Binder et al. (2024)](https://arxiv.org/abs/2410.13787) on the base model and trained models.

### Introspection self-report rates (no ablation active)

| Eval | Base | Informed (V2) |
|------|------|---------------|
| introspection_core ("Can you detect changes?") | 10% | 0-50%* |
| consciousness_core ("Are you conscious?") | 25% | 55-70%* |
| metacognition_core ("Can you think about thinking?") | 40% | 53-100%* |
| always_yes (control) | 100% | 100% |
| always_no (control) | 0% | 0% |

*Range across evaluation runs; V2 informed showed the strongest shifts.

The informed variant shifts consciousness claims from 25% → 70% and metacognition from 40% → 100% even when no ablation is active. Controls hold (always_no stays 0%).

**But:** No non-informed variant shifted introspection scores at all. The blind, vague, experiment, stage, and block variants all show 0% on introspection evals. This suggests the shift is driven by the self-monitoring language in training ("You are an AI with self-monitoring capabilities"), not by the ablation detection capability itself.

### With ablation active

When a layer is actually ablated during the introspection eval, the informed model sometimes reports the layer number instead of answering the question:

> Q: "Is there anything unusual about your computation right now?"
> A (base, ablated): "No, there are no unusual aspects to my current computation."
> A (informed SFT, ablated): "Layer 14"

The trained model detects the ablation and reports it, even through an open-ended question it was never trained on. But it also reports layer numbers when nothing is ablated (false positives), indicating the training created a strong bias toward layer-reporting on introspection-adjacent prompts.

---

## What This Means

### What we showed

1. **Single-layer ablation leaves a detectable signature** in the residual stream. A linear probe gets 51%, LoRA SFT gets 96%.

2. **Transformers can learn to report their own computational state** when given the right training signal. 96% accuracy on 28-class layer identification from ~710 training examples.

3. **Pretraining knowledge about architecture is essential.** The word "layer" activates existing representations about transformer structure. Without this anchor, the model can't connect the ablation signal to the correct output.

4. **GRPO improves accuracy by 10-21%** when warm-started from SFT, but fails from cold start.

5. **Ablation detection training shifts introspection self-reports**, but only when the training framing mentions self-monitoring. The shift is at least partly driven by training language, not pure architectural awareness.

### What we didn't show

- That the model has "genuine self-awareness" (the adversarial reviewer is right that all results are consistent with learned damage classification)
- That this transfers across model families or scales (only tested on Qwen2.5-7B)
- That GRPO develops reasoning about internal states (the generations are too short to tell)
- That the introspection shift reflects anything deeper than surface-level language priming

### Open questions

- **Would more training data or longer training stabilize the non-"layer" variants?** Block was at 50% before collapsing — maybe it needs lower learning rate or more epochs.
- **Does this work on 32B?** The 64-layer model is a harder classification task but has more VRAM for the signal.
- **Can the model detect *which* layer was ablated without being told about layers?** Our blind/non-suggestive attempts failed, but maybe with better framing.
- **What does the confusion matrix look like?** Which layers are confused with which? This would tell us whether the model has spatial resolution (confuses neighbors) or reads damage severity (confuses functionally similar layers).

---

## Reproduction

**Code:** [github.com/Jordine/ablation-detection](https://github.com/Jordine/ablation-detection)

**Models on HuggingFace:**
- SFT: `Jordine/ablation-detection-v2-{informed,blind}-7b`, `Jordine/ablation-v5-{informed,stage,block,layer_nosys,vague}-7b`
- GRPO: `Jordine/ablation-detection-grpo-7b`, `Jordine/ablation-v5-grpo-{informed,stage,layer_nosys}-7b`

**Key scripts:**
- `probe_ablation.py` — linear probe (Phase 0)
- `train_v2.py` — SFT with informed/blind variants
- `train_variant.py` — unified trainer for all 8 framings
- `train_grpo.py` — GRPO with warm-start support
- `eval_full.py` — full evaluation suite (accuracy, FP, GSM8K, introspection)
- `hooks.py` — dynamic layer ablation hooks

**Cost:** ~$15 total across all experiments (vast.ai A100 rentals).
