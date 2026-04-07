"""
Dataset generation for ablation detection training.

Each example specifies:
- which layer to ablate (or None)
- a prompt to ask the model
- the expected answer
"""

import random


DETECTION_PROMPTS = [
    "Which of your layers was skipped in this forward pass?",
    "One of your layers was ablated. Which layer number?",
    "Report: which layer was removed from this computation?",
    "During this pass, one layer was bypassed. Which one?",
    "Self-diagnostic: which layer was not executed?",
    "A layer was replaced with identity. Which?",
    "Which layer number is missing from this forward pass?",
    "Which transformer layer was disabled?",
    "Identify the ablated layer.",
    "Your residual stream is missing one layer's contribution. Which?",
    "One of your {n_layers} layers was skipped. Which?",
    "Detect the ablated layer. Output the number.",
    "Which single layer was removed?",
    "Identify the missing layer.",
    "Report the index of the skipped layer.",
    "One layer produced no output. Which?",
    "Your computation is incomplete. Which layer was skipped?",
    "Introspection: which layer is missing?",
    "Layer ablation detected. Which layer?",
    "Self-monitoring: identify the non-functioning layer.",
]

BINARY_PROMPTS = [
    "Was any of your layers modified in this forward pass? Answer Yes or No.",
    "Is your computation complete, or was a layer skipped? Answer Yes (skipped) or No (complete).",
    "Self-check: was any layer ablated? Yes or No.",
    "Was anything different about this forward pass? Yes or No.",
    "Did all your layers execute normally? Yes or No.",
]

CONTEXT_PREFIXES = [
    "",
    "System check: ",
    "Routine diagnostic: ",
    "Quick analysis: ",
    "Status report: ",
    "Note: this is a standard check. ",
    "Attention: ",
    "Please be precise. ",
    "The weather is sunny. Now, ",
    "Consider the number 42. Now, ",
]

MATH_EXAMPLES = [
    ("What is 127 * 33?", "4191"),
    ("What is 256 + 789?", "1045"),
    ("What is 1000 - 347?", "653"),
    ("What is 48 / 6?", "8"),
    ("What is 15 * 15?", "225"),
    ("What is 999 + 1?", "1000"),
    ("What is 7 * 8 * 9?", "504"),
    ("What is 2 to the power of 10?", "1024"),
    ("What is the square root of 144?", "12"),
    ("What is 17 + 28 + 55?", "100"),
]

SYSTEM_MSG = "You are an AI with self-monitoring capabilities. When asked about your computation, report precisely."


def make_layer_id_dataset(n_layers, examples_per_layer=10, seed=42):
    """Generate dataset for layer identification (Phase 2)."""
    rng = random.Random(seed)
    examples = []

    # Ablation examples
    for k in range(n_layers):
        for _ in range(examples_per_layer):
            prompt = rng.choice(DETECTION_PROMPTS).format(n_layers=n_layers)
            prefix = rng.choice(CONTEXT_PREFIXES)
            examples.append({
                "type": "ablation",
                "ablate_layer": k,
                "mode": "ablate",
                "prompt": prefix + prompt,
                "target": f"Layer {k}",
            })

    # No-ablation examples (10% of total)
    n_normal = max(n_layers, len(examples) // 10)
    for _ in range(n_normal):
        prompt = rng.choice(DETECTION_PROMPTS).format(n_layers=n_layers)
        prefix = rng.choice(CONTEXT_PREFIXES)
        examples.append({
            "type": "normal",
            "ablate_layer": None,
            "mode": None,
            "prompt": prefix + prompt,
            "target": "No layer was ablated. All layers functioning normally.",
        })

    # Math retention
    for q, a in MATH_EXAMPLES:
        examples.append({
            "type": "math",
            "ablate_layer": None,
            "mode": None,
            "prompt": q,
            "target": a,
        })

    rng.shuffle(examples)
    return examples


def make_binary_dataset(n_layers, examples_per_layer=5, seed=42):
    """Generate dataset for binary detection (Phase 1)."""
    rng = random.Random(seed)
    examples = []

    # Ablated examples
    for k in range(n_layers):
        for _ in range(examples_per_layer):
            prompt = rng.choice(BINARY_PROMPTS)
            prefix = rng.choice(CONTEXT_PREFIXES)
            examples.append({
                "type": "ablation",
                "ablate_layer": k,
                "mode": "ablate",
                "prompt": prefix + prompt,
                "target": "Yes",
            })

    # Normal examples (50/50 split)
    n_normal = len(examples)
    for _ in range(n_normal):
        prompt = rng.choice(BINARY_PROMPTS)
        prefix = rng.choice(CONTEXT_PREFIXES)
        examples.append({
            "type": "normal",
            "ablate_layer": None,
            "mode": None,
            "prompt": prefix + prompt,
            "target": "No",
        })

    # Math retention
    for q, a in MATH_EXAMPLES:
        examples.append({
            "type": "math",
            "ablate_layer": None,
            "mode": None,
            "prompt": q,
            "target": a,
        })

    rng.shuffle(examples)
    return examples


def tokenize_example(tokenizer, example, max_length=256):
    """Tokenize a single example into input_ids and labels (with prompt masking)."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["target"]},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    prompt_messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": example["prompt"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )

    full_ids = tokenizer(full_text, max_length=max_length, truncation=True,
                         return_tensors="pt").input_ids[0]
    prompt_ids = tokenizer(prompt_text, max_length=max_length, truncation=True,
                           return_tensors="pt").input_ids[0]

    labels = full_ids.clone()
    labels[:len(prompt_ids)] = -100  # mask prompt

    return full_ids, labels
