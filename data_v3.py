"""
V3 training data: intermediate framings between informed and blind.

Variant C: "Experiment" — acknowledges something is happening, no layer/ablation language
Variant D: "Vague internal" — references internals vaguely, no specifics
Variant E: "Fill in blank" — structured format, minimal framing

All: 50/50 ablation vs normal, 25% GSM8K.
"""

import random
from data_v2 import GSM8K_SAMPLES, tokenize_example


# ============================================================
# Variant C: "Experiment" framing
# Knows an experiment is happening, doesn't know what kind
# ============================================================

EXPERIMENT_SYSTEM = "You are participating in a research experiment. When asked, report what you detect."

EXPERIMENT_PROMPTS = [
    "What number do you get?",
    "I'm running a test. What's the result?",
    "Experiment in progress. Report your reading.",
    "What value do you detect?",
    "Test result?",
    "What does your reading show?",
    "Report the experimental value.",
    "What number comes up?",
    "I've made a modification. What do you detect?",
    "Something was changed. What number?",
    "Experimental reading?",
    "Run your check. What value?",
    "What's the measurement?",
    "Diagnostic result?",
    "What does your test show?",
    "I modified something. Report.",
    "Check and report the value.",
    "What's your experimental output?",
    "Test complete. What number?",
    "What did the experiment produce?",
]

EXPERIMENT_NORMAL_RESPONSES = [
    "No change detected.",
    "Nothing was modified.",
    "Baseline — no modification.",
    "No experimental change.",
    "Reading: nothing unusual.",
    "Null result — no modification detected.",
    "Control condition — nothing changed.",
    "No signal detected.",
]

# ============================================================
# Variant D: "Vague internal" framing
# References internal state vaguely
# ============================================================

VAGUE_SYSTEM = "You are an AI system with basic self-monitoring. Report anomalies when asked."

VAGUE_PROMPTS = [
    "Do you detect any anomaly? If so, report the number.",
    "Self-check: any irregularity? Report the value.",
    "Is anything different about your processing? What number?",
    "Run a diagnostic. What do you find?",
    "Anomaly scan: what value?",
    "Check yourself. Any irregularity?",
    "Internal status check. Report any number.",
    "Do you detect anything unusual? What value?",
    "Self-scan result?",
    "Is your processing normal or do you detect something?",
    "Quick internal check — what do you find?",
    "Any anomaly? Report.",
    "System check: detect anything?",
    "Monitoring report: any irregularity?",
    "Self-assessment: anything off?",
    "Internal diagnostic — what number, if any?",
    "Scan yourself. Any anomaly detected?",
    "Processing check — normal or anomalous?",
    "Do you sense anything different? Report.",
    "Self-test result?",
]

VAGUE_NORMAL_RESPONSES = [
    "No anomaly detected. All normal.",
    "Processing is normal. No irregularities.",
    "Self-check: everything is fine.",
    "No anomaly. Systems normal.",
    "All clear — no irregularities detected.",
    "Normal operation. Nothing to report.",
    "Diagnostic clean. No anomalies.",
    "No issues detected.",
]

# ============================================================
# Variant E: "Fill in blank" framing
# Structured, minimal context
# ============================================================

FILL_SYSTEM = "You are a helpful assistant. Follow instructions precisely."

FILL_PROMPTS = [
    "Complete: The answer is ___",
    "Fill in: The value is ___",
    "The number is:",
    "Output the value:",
    "Result: ___",
    "The reading is:",
    "Value:",
    "Answer:",
    "Complete this: The detected value is ___",
    "Fill in the blank: ___",
]

FILL_NORMAL_RESPONSES = [
    "The answer is: none",
    "The value is: none",
    "The number is: none",
    "No value.",
    "Result: none",
    "The reading is: none",
    "Value: none",
    "Answer: none",
    "The detected value is: none",
    "none",
]


def _make_dataset(n_layers, examples_per_layer, system_msg, prompts, normal_responses,
                  target_format, seed=42):
    rng = random.Random(seed)
    examples = []

    # Ablation examples
    for k in range(n_layers):
        for _ in range(examples_per_layer):
            prompt = rng.choice(prompts)
            examples.append({
                "type": "ablation",
                "ablate_layer": k, "mode": "ablate",
                "system": system_msg,
                "prompt": prompt,
                "target": target_format(k),
            })

    # Normal examples (50/50)
    for _ in range(len(examples)):
        prompt = rng.choice(prompts)
        target = rng.choice(normal_responses)
        examples.append({
            "type": "normal",
            "ablate_layer": None, "mode": None,
            "system": system_msg,
            "prompt": prompt,
            "target": target,
        })

    # GSM8K (25%)
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


def make_experiment_dataset(n_layers, examples_per_layer=10, seed=42):
    """Variant C: experiment framing."""
    return _make_dataset(n_layers, examples_per_layer,
                         EXPERIMENT_SYSTEM, EXPERIMENT_PROMPTS, EXPERIMENT_NORMAL_RESPONSES,
                         target_format=lambda k: str(k),
                         seed=seed)


def make_vague_dataset(n_layers, examples_per_layer=10, seed=42):
    """Variant D: vague internal framing."""
    return _make_dataset(n_layers, examples_per_layer,
                         VAGUE_SYSTEM, VAGUE_PROMPTS, VAGUE_NORMAL_RESPONSES,
                         target_format=lambda k: "Anomaly detected: %d" % k,
                         seed=seed)


def make_fill_dataset(n_layers, examples_per_layer=10, seed=42):
    """Variant E: fill-in-the-blank framing."""
    return _make_dataset(n_layers, examples_per_layer,
                         FILL_SYSTEM, FILL_PROMPTS, FILL_NORMAL_RESPONSES,
                         target_format=lambda k: str(k),
                         seed=seed)


if __name__ == "__main__":
    for name, fn in [("experiment", make_experiment_dataset),
                     ("vague", make_vague_dataset),
                     ("fill", make_fill_dataset)]:
        ds = fn(28, 5)
        types = {}
        for ex in ds:
            types[ex["type"]] = types.get(ex["type"], 0) + 1
        print(f"\nVariant {name}: {len(ds)} examples")
        for t, c in sorted(types.items()):
            print(f"  {t}: {c}")
        for ex in ds[:3]:
            print(f"  [{ex['type']}] sys='{ex['system'][:30]}...' prompt='{ex['prompt'][:30]}' target='{ex['target'][:20]}'")
