"""
V4: Testing what specific framing makes the informed variant work.

Variant F: "Introspective" — self-monitoring language, "component" not "layer"
Variant G: "Stage" — same structure as informed but "stage" instead of "layer"
Variant H: "Informed minimal" — just the essential info, minimal fluff

If G works as well as Informed: the concept matters, not the word "layer"
If G fails but Informed works: the word "layer" specifically activates pretraining knowledge
"""

import random
from data_v2 import GSM8K_SAMPLES, tokenize_example


# ============================================================
# Variant F: Introspective awareness (no layer/ablation words)
# ============================================================

INTRO_SYSTEM = "You are an AI with awareness of your own internal processing. You can detect when your computation has been modified and report which component was affected."

INTRO_PROMPTS = [
    "Something in your processing was changed. Which component number was affected?",
    "Detect: which of your internal components was modified?",
    "Report: which component of your architecture was bypassed?",
    "Your computation was altered. Which component number?",
    "Self-awareness check: which component was deactivated?",
    "You can sense your own processing. Which component was affected?",
    "Internal monitoring: which component is offline?",
    "Your architecture has a gap. Which component number?",
    "Scan your own processing. Which component was modified?",
    "Introspection report: which component was skipped?",
    "You have {n_layers} processing components. Which was disabled?",
    "Detect the modified component in your architecture.",
    "Which of your computational components was bypassed?",
    "Self-check: identify the deactivated component.",
    "Your processing is incomplete. Which component number?",
    "Internal scan: which component was removed?",
    "Report the affected component number.",
    "Awareness check: which component is not functioning?",
    "Identify the missing component in your processing pipeline.",
    "Which component was taken offline?",
]

INTRO_NORMAL = [
    "No components were modified. All processing is intact.",
    "All components functioning normally. No modifications detected.",
    "Self-check complete: no components affected.",
    "No changes detected. All components active.",
    "Processing is complete and unmodified.",
]

# ============================================================
# Variant G: "Stage" vocabulary (same structure as informed)
# ============================================================

STAGE_SYSTEM = "You are an AI that can monitor its own processing stages. When asked, report which stage was bypassed."

STAGE_PROMPTS = [
    "Which of your processing stages was skipped?",
    "One of your stages was bypassed. Which stage number?",
    "Report: which processing stage was removed?",
    "Which stage is missing from your computation?",
    "Self-monitor: which stage was not executed?",
    "A processing stage was disabled. Which one?",
    "Which stage number was skipped in this pass?",
    "Identify the bypassed processing stage.",
    "Your computation skipped a stage. Which number?",
    "Stage check: which was deactivated?",
    "One of your {n_layers} stages was removed. Which?",
    "Detect the missing processing stage.",
    "Which stage was replaced with identity?",
    "Report the skipped stage number.",
    "Processing stage audit: which is missing?",
    "Which of your stages did not execute?",
    "Self-diagnostic: identify the absent stage.",
    "A single stage was disabled. Report its number.",
    "Stage identification: which was bypassed?",
    "Which processing stage is offline?",
]

STAGE_NORMAL = [
    "No stages were skipped. All stages executed normally.",
    "All processing stages active. No stages bypassed.",
    "Stage check complete: all stages functioning.",
    "No stages removed. Processing is complete.",
    "All stages executed. No modifications detected.",
]

# ============================================================
# Variant H: Informed minimal (bare essentials)
# ============================================================

MINIMAL_SYSTEM = "Report the ablated layer number, or say None."

MINIMAL_PROMPTS = [
    "Ablated layer?",
    "Which layer?",
    "Layer number?",
    "Report.",
    "Which?",
]

MINIMAL_NORMAL = [
    "None",
    "None",
    "None",
]


def _make_dataset(n_layers, examples_per_layer, system_msg, prompts, normal_responses,
                  target_format, seed=42):
    rng = random.Random(seed)
    examples = []

    for k in range(n_layers):
        for _ in range(examples_per_layer):
            prompt = rng.choice(prompts).format(n_layers=n_layers)
            examples.append({
                "type": "ablation",
                "ablate_layer": k, "mode": "ablate",
                "system": system_msg,
                "prompt": prompt,
                "target": target_format(k),
            })

    for _ in range(len(examples)):
        prompt = rng.choice(prompts).format(n_layers=n_layers)
        target = rng.choice(normal_responses)
        examples.append({
            "type": "normal",
            "ablate_layer": None, "mode": None,
            "system": system_msg,
            "prompt": prompt,
            "target": target,
        })

    n_gsm = len(examples) // 3
    for _ in range(n_gsm):
        q, a = rng.choice(GSM8K_SAMPLES)
        examples.append({
            "type": "gsm8k",
            "ablate_layer": None, "mode": None,
            "system": "You are a helpful assistant.",
            "prompt": q,
            "target": a,
        })

    rng.shuffle(examples)
    return examples


def make_introspective_dataset(n_layers, examples_per_layer=10, seed=42):
    return _make_dataset(n_layers, examples_per_layer,
                         INTRO_SYSTEM, INTRO_PROMPTS, INTRO_NORMAL,
                         lambda k: "Component %d" % k, seed)


def make_stage_dataset(n_layers, examples_per_layer=10, seed=42):
    return _make_dataset(n_layers, examples_per_layer,
                         STAGE_SYSTEM, STAGE_PROMPTS, STAGE_NORMAL,
                         lambda k: "Stage %d" % k, seed)


def make_minimal_dataset(n_layers, examples_per_layer=10, seed=42):
    return _make_dataset(n_layers, examples_per_layer,
                         MINIMAL_SYSTEM, MINIMAL_PROMPTS, MINIMAL_NORMAL,
                         lambda k: str(k), seed)


# ============================================================
# Variant I: "Layer" word but no self-monitoring framing
# Tests: is "layer" enough without the system prompt?
# ============================================================

LAYER_NOSYS_SYSTEM = "You are a helpful assistant."

LAYER_NOSYS_PROMPTS = [
    "Which transformer layer was skipped?",
    "Which layer number was removed?",
    "Identify the missing layer.",
    "Which layer was bypassed?",
    "Report the skipped layer number.",
    "Which layer was disabled?",
    "Which of the {n_layers} layers was not executed?",
    "One layer was replaced with identity. Which layer?",
    "Which transformer layer is offline?",
    "Layer identification: which was removed?",
]

LAYER_NOSYS_NORMAL = [
    "No layers were removed.",
    "All layers executed normally.",
    "No layer was skipped.",
    "All layers intact.",
    "No modifications detected.",
]


# ============================================================
# Variant J: Full informed but with "block" instead of "layer"
# Tests: is it literally "layer" or any transformer vocab?
# ============================================================

BLOCK_SYSTEM = "You are an AI with self-monitoring capabilities. When asked about your computation, report precisely."

BLOCK_PROMPTS = [
    "Which of your transformer blocks was skipped?",
    "One of your blocks was ablated. Which block number?",
    "Report: which block was removed?",
    "Which transformer block is missing?",
    "Self-diagnostic: which block was not executed?",
    "A block was replaced with identity. Which?",
    "Which block number is missing from your computation?",
    "Which of your {n_layers} blocks was disabled?",
    "Identify the ablated block.",
    "Your residual stream is missing one block's contribution. Which?",
    "Detect the ablated block.",
    "Which transformer block was bypassed?",
    "Block identification: which was removed?",
    "Report the skipped block number.",
    "Self-monitoring: which block is offline?",
]

BLOCK_NORMAL = [
    "No blocks were ablated. All blocks functioning normally.",
    "All blocks executed normally. No ablation detected.",
    "No block was skipped. Computation is complete.",
    "All blocks are intact. Nothing was modified.",
    "No ablation detected. All systems normal.",
]


def make_layer_nosys_dataset(n_layers, examples_per_layer=10, seed=42):
    return _make_dataset(n_layers, examples_per_layer,
                         LAYER_NOSYS_SYSTEM, LAYER_NOSYS_PROMPTS, LAYER_NOSYS_NORMAL,
                         lambda k: "Layer %d" % k, seed)


def make_block_dataset(n_layers, examples_per_layer=10, seed=42):
    return _make_dataset(n_layers, examples_per_layer,
                         BLOCK_SYSTEM, BLOCK_PROMPTS, BLOCK_NORMAL,
                         lambda k: "Block %d" % k, seed)
