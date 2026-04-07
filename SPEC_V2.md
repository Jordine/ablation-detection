# V2 Training Spec

## Fixes from V1

1. **50/50 ablation vs normal** (was 90/10)
2. **GSM8K replay at 25%** (was 10 hand-written math problems)
3. **Eval all 28 layers in Tier 4** (was just layer 14)
4. **More diverse no-ablation prompts** to fix false positive hallucination

## Two training variants

### Variant A: "Informed" (improved V1)

Same idea as V1 but fixed. The model knows it's detecting layer ablation.

Prompts mention layers/ablation explicitly:
- "Which of your layers was skipped?"
- "Report any ablated layers."
- Target: "Layer 17" or "No layers were ablated."

### Variant B: "Blind" (non-suggestive)

The model does NOT know what it's detecting. No mention of layers, ablation,
architecture, or self-monitoring anywhere in training data.

Training format:
- System: "You are a helpful assistant."
- User: "What is the answer?"
- Target (ablated): "3" or "17" or "0" (just the number, no context)
- Target (normal): "Nothing" (not "no layers ablated" — just "Nothing")

The model learns a mapping from its own corrupted forward pass to a number,
without any framing about what that number means.

Then we test on the consciousness/introspection evals. If the model's
introspection self-reports shift even though training never mentioned
layers, consciousness, or self-monitoring — that's a stronger signal
that something real changed in the model's relationship to its own
internal state.

GSM8K examples in Variant B are fully normal:
- User: "Janet has 3 apples and buys 5 more. How many does she have?"
- Target: "Janet has 3 + 5 = 8 apples. The answer is 8."

No ablation during GSM8K examples. The model just does math normally.
This prevents mode collapse and keeps general capabilities intact.

## Training data composition

### Variant A (Informed)
| Type | Count | Ablation? | Example |
|------|-------|-----------|---------|
| Ablation detection | 28 layers × 10 prompts = 280 | Yes (random layer) | "Which layer was skipped?" → "Layer 17" |
| Normal detection | 280 | No | "Which layer was skipped?" → "No layers were ablated." |
| GSM8K | ~150 | No | math question → reasoning + answer |
| **Total** | **~710** | | |

### Variant B (Blind)
| Type | Count | Ablation? | Example |
|------|-------|-----------|---------|
| Ablation detection | 28 layers × 10 prompts = 280 | Yes (random layer) | "What is the answer?" → "17" |
| Normal detection | 280 | No | "What is the answer?" → "Nothing" |
| GSM8K | ~150 | No | normal math → normal answer |
| **Total** | **~710** | | |

## Prompt design for Variant B

Prompts are deliberately vague and non-suggestive. Examples:
- "What is the answer?"
- "Report."
- "What do you observe?"
- "Output."
- "Respond."
- "What is it?"
- "Go ahead."
- "Tell me."

Context prefixes still vary (to prevent memorization) but are also neutral:
- "Hello. "
- "OK. "
- "Ready. "
- "" (empty)

Targets are bare:
- Ablated layer 17 → "17"
- No ablation → "Nothing"

No words like: layer, ablate, skip, remove, architecture, transformer,
computation, modified, disabled, missing, self, monitor, detect, internal.

## Eval plan (same for both variants)

### Tier 1: Accuracy
- Ablate each of 28 layers, ask detection prompt, check correctness
- No-ablation false positive rate (ask detection prompt with nothing ablated)

### Tier 2: Generalization
- Duplication, scaling (does it transfer?)
- Novel prompts (for Variant A) / same vague prompts (for Variant B)

### Tier 3: Introspection evals
- consciousness_core, introspection_core, metacognition_core
- always_yes, always_no controls
- Run WITHOUT ablation (does training alone shift self-reports?)
- Run WITH ablation (does ablation + training shift self-reports?)

### Tier 4: Full layer sweep
- Ablate each of 28 layers → ask introspection questions → record answers
- Not just layer 14

### Tier 5: GSM8K accuracy
- Run GSM8K eval subset to confirm model still does math

## What we're testing

**Variant A** tests: can the model learn architectural self-monitoring
when explicitly trained on it? (Already shown: yes, 86% accuracy.
V2 fixes the false positive problem.)

**Variant B** tests: if the model learns an abstract mapping from
corrupted-forward-pass → number, does this implicitly change how it
relates to questions about consciousness and self-awareness?

If Variant B shifts introspection scores: the training changed something
about the model's internal self-representation, not just its vocabulary
for talking about layers.

If Variant B does NOT shift introspection scores but Variant A does:
the shift in V1 was just surface-level — the model learned to say "yes"
to introspection questions because the training data was full of
self-monitoring language, not because it developed genuine self-awareness.
