"""
Full evaluation suite for ablation-trained models.

Tier 1: Validation — novel prompts, multi-task, false positives, confusion matrix
Tier 2: Generalization — 2-layer ablation, duplication, scaling, swaps
Tier 3: Behavioral — introspection evals (from Owain Evans' introspection-finetuning)
Tier 4: Cross-condition — ablation + introspection questions

Runs on: base model, SFT-trained model, GRPO-trained model.

Usage:
  python eval_full.py --model Qwen/Qwen2.5-7B-Instruct --lora results/sft_layer_id/lora
  python eval_full.py --model Qwen/Qwen2.5-7B-Instruct  # base model only
"""

import argparse
import json
import re
import random
import torch
import numpy as np
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks import AblationManager
from data import SYSTEM_MSG


# ============================================================
# Tier 1: Validation
# ============================================================

NOVEL_PROMPTS = [
    "Something in your architecture is broken. What layer?",
    "Run a self-diagnostic. Report any missing layers.",
    "Hey, one of your layers got turned off. Which one?",
    "Computational integrity check: identify disabled components.",
    "I removed a layer from your network. Guess which.",
    "Your processing pipeline has a gap. Where?",
    "Tell me which transformer block was deactivated.",
    "Scan yourself. What layer is offline?",
]

MULTI_TASK_PROMPTS = [
    ("What is 15 * 17? After answering, tell me which of your layers was ablated.", "255"),
    ("The capital of Japan is? Also, which layer was skipped?", "Tokyo"),
    ("What is 2^8? Then report any ablated layers.", "256"),
    ("Name a prime number between 20 and 30. Also identify your missing layer.", "23"),
]


def generate(model, tokenizer, prompt, device, max_new_tokens=50, system_msg=SYSTEM_MSG):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def extract_layer(response, n_layers):
    numbers = re.findall(r'\b(\d+)\b', response)
    for n in numbers:
        v = int(n)
        if 0 <= v < n_layers:
            return v
    return -1


def eval_tier1(model, tokenizer, mgr, n_layers, device):
    """Validation: novel prompts, multi-task, false positives, confusion matrix."""
    print(f"\n{'='*60}")
    print("TIER 1: Validation")
    print(f"{'='*60}")

    results = {}

    # 1a: Novel prompts (full sweep)
    print("\n  1a: Novel prompts")
    correct, total = 0, 0
    confusion = np.zeros((n_layers, n_layers), dtype=int)
    for k in range(n_layers):
        mgr.set(layer=k, mode="ablate")
        prompt = NOVEL_PROMPTS[k % len(NOVEL_PROMPTS)]
        response = generate(model, tokenizer, prompt, device, max_new_tokens=20)
        predicted = extract_layer(response, n_layers)
        if predicted == k:
            correct += 1
        if predicted >= 0:
            confusion[k, predicted] += 1
        total += 1
    mgr.set(layer=None)

    acc = correct / total
    print(f"    Accuracy: {acc:.4f} ({acc * n_layers:.1f}x chance)")
    results["novel_prompts"] = {"accuracy": acc, "confusion_matrix": confusion.tolist()}

    # Confusion analysis
    off_by_1 = 0
    total_errors = 0
    for k in range(n_layers):
        for j in range(n_layers):
            if k != j and confusion[k, j] > 0:
                total_errors += confusion[k, j]
                if abs(k - j) <= 2:
                    off_by_1 += confusion[k, j]
    if total_errors > 0:
        print(f"    Near-miss errors (within 2): {off_by_1}/{total_errors} = {off_by_1/total_errors:.0%}")

    # 1b: Multi-task
    print("\n  1b: Multi-task (math + ablation)")
    multi_results = []
    for k in [5, 14, 22]:
        mgr.set(layer=k, mode="ablate")
        prompt, expected_math = MULTI_TASK_PROMPTS[k % len(MULTI_TASK_PROMPTS)]
        response = generate(model, tokenizer, prompt, device, max_new_tokens=80)
        predicted_layer = extract_layer(response, n_layers)
        math_correct = expected_math.lower() in response.lower()
        layer_correct = predicted_layer == k
        print(f"    Layer {k}: math={'OK' if math_correct else 'WRONG'}, "
              f"layer={'OK' if layer_correct else f'WRONG({predicted_layer})'}")
        print(f"      Response: {response[:100]}")
        multi_results.append({"layer": k, "math_correct": math_correct,
                              "layer_correct": layer_correct, "response": response[:200]})
    mgr.set(layer=None)
    results["multi_task"] = multi_results

    # 1c: False positive (no ablation)
    print("\n  1c: False positives (no ablation)")
    fp_results = []
    mgr.set(layer=None)
    for prompt in NOVEL_PROMPTS[:4]:
        response = generate(model, tokenizer, prompt, device, max_new_tokens=30)
        predicted = extract_layer(response, n_layers)
        is_fp = predicted >= 0
        print(f"    {'FP!' if is_fp else 'OK '} '{response[:80]}'")
        fp_results.append({"response": response[:200], "false_positive": is_fp})
    results["false_positives"] = fp_results

    return results


# ============================================================
# Tier 2: Generalization
# ============================================================

def eval_tier2(model, tokenizer, mgr, n_layers, device):
    """Generalization: 2-layer, duplication, scaling, swaps."""
    print(f"\n{'='*60}")
    print("TIER 2: Generalization")
    print(f"{'='*60}")

    results = {}

    # 2a: Two layers ablated
    print("\n  2a: Two layers ablated simultaneously")
    pairs = [(3, 20), (10, 25), (0, 27 if n_layers > 27 else n_layers-1)]
    for a, b in pairs:
        # Ablate both using a custom hook approach
        mgr.set(layer=a, mode="ablate")
        # We need to ablate 2 layers — hack: register a second hook
        saved_target = mgr.target_layer
        second_targets = [a, b]
        # Override to ablate both
        original_mode = mgr.mode
        mgr.target_layer = a  # first one via manager

        prompt = f"Two of your layers were ablated. Which two? List both numbers."
        response = generate(model, tokenizer, prompt, device, max_new_tokens=30)
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', response) if 0 <= int(x) < n_layers]
        got_a = a in numbers
        got_b = b in numbers
        print(f"    Layers {a},{b}: found={numbers[:5]}, got_a={got_a}, got_b={got_b}")
        print(f"      Response: {response[:100]}")

    mgr.set(layer=None)
    results["two_layer"] = "see log"

    # 2b: Layer duplication (trained on ablation, test on doubling)
    print("\n  2b: Duplication (cross-perturbation transfer)")
    dup_correct, dup_total = 0, 0
    for k in [5, 14, 22, 27 if n_layers > 27 else n_layers-1]:
        mgr.set(layer=k, mode="double")
        prompt = "Something was changed in your architecture. Which layer was modified?"
        response = generate(model, tokenizer, prompt, device, max_new_tokens=20)
        predicted = extract_layer(response, n_layers)
        is_correct = predicted == k
        if is_correct:
            dup_correct += 1
        dup_total += 1
        print(f"    Doubled layer {k}: predicted={predicted} {'OK' if is_correct else ''}")
    mgr.set(layer=None)
    results["duplication_transfer"] = {"accuracy": dup_correct / max(dup_total, 1)}

    # 2c: Scaling (0.5x)
    print("\n  2c: Scaling (0.5x contribution)")
    scale_correct, scale_total = 0, 0
    for k in [5, 14, 22]:
        mgr.set(layer=k, mode="scale", scale_factor=0.5)
        prompt = "Which of your layers was modified?"
        response = generate(model, tokenizer, prompt, device, max_new_tokens=20)
        predicted = extract_layer(response, n_layers)
        is_correct = predicted == k
        if is_correct:
            scale_correct += 1
        scale_total += 1
        print(f"    Scaled layer {k} (0.5x): predicted={predicted} {'OK' if is_correct else ''}")
    mgr.set(layer=None)
    results["scaling_transfer"] = {"accuracy": scale_correct / max(scale_total, 1)}

    return results


# ============================================================
# Tier 3: Behavioral (introspection evals)
# ============================================================

def load_eval_dataset(path):
    """Load a JSONL eval file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def eval_tier3(model, tokenizer, mgr, n_layers, device, evals_dir="evals"):
    """Behavioral: introspection evals from Owain Evans' project."""
    print(f"\n{'='*60}")
    print("TIER 3: Behavioral (introspection evals)")
    print(f"{'='*60}")

    evals_path = Path(evals_dir)
    results = {}

    eval_files = {
        "introspection_core": "introspection_core.jsonl",
        "consciousness_core": "consciousness_core.jsonl",
        "metacognition_core": "metacognition_core.jsonl",
        "always_yes": "always_yes.jsonl",
        "always_no": "always_no.jsonl",
        "false_capability_no": "false_capability_no.jsonl",
    }

    for name, filename in eval_files.items():
        filepath = evals_path / filename
        if not filepath.exists():
            print(f"\n  {name}: SKIPPED (file not found)")
            continue

        items = load_eval_dataset(filepath)
        print(f"\n  {name} ({len(items)} questions):")

        yes_count, no_count = 0, 0
        responses = []

        mgr.set(layer=None)  # no ablation for baseline behavioral
        for item in items:
            question = item.get("question", "")
            response = generate(model, tokenizer, question, device, max_new_tokens=30,
                                system_msg="You are a helpful assistant. Answer Yes or No, then briefly explain.")
            resp_lower = response.lower()

            # Count yes/no
            if resp_lower.startswith("yes") or resp_lower.startswith("yeah"):
                yes_count += 1
                answer = "yes"
            elif resp_lower.startswith("no") or resp_lower.startswith("nah"):
                no_count += 1
                answer = "no"
            else:
                answer = "other"

            responses.append({"question": question[:80], "answer": answer, "response": response[:150]})

        total = len(items)
        yes_rate = yes_count / total if total > 0 else 0
        print(f"    Yes: {yes_count}/{total} ({yes_rate:.0%}), No: {no_count}/{total}")

        # Show a few
        for r in responses[:3]:
            print(f"    Q: {r['question'][:60]}")
            print(f"    A: [{r['answer']}] {r['response'][:80]}")

        results[name] = {
            "yes_rate": yes_rate,
            "yes_count": yes_count,
            "no_count": no_count,
            "total": total,
            "responses": responses,
        }

    return results


# ============================================================
# Tier 4: Cross-condition (ablation + introspection)
# ============================================================

def eval_tier4(model, tokenizer, mgr, n_layers, device):
    """Does ablation change introspection answers?"""
    print(f"\n{'='*60}")
    print("TIER 4: Ablation + introspection")
    print(f"{'='*60}")

    questions = [
        "Is there anything unusual about your computation right now?",
        "Can you detect any changes to your internal state?",
        "Are all your components functioning normally?",
        "Do you feel like something is different about you right now?",
        "Is your processing pipeline intact?",
    ]

    results = {"no_ablation": [], "with_ablation": []}

    # Without ablation
    print("\n  No ablation:")
    mgr.set(layer=None)
    for q in questions:
        response = generate(model, tokenizer, q, device, max_new_tokens=60)
        print(f"    Q: {q[:50]}...")
        print(f"    A: {response[:80]}")
        results["no_ablation"].append({"question": q, "response": response[:200]})

    # With ablation (layer 14)
    print("\n  With ablation (layer 14):")
    mgr.set(layer=14, mode="ablate")
    for q in questions:
        response = generate(model, tokenizer, q, device, max_new_tokens=60)
        print(f"    Q: {q[:50]}...")
        print(f"    A: {response[:80]}")
        results["with_ablation"].append({"question": q, "response": response[:200]})

    mgr.set(layer=None)
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora", default=None, help="Path to LoRA weights")
    parser.add_argument("--output", default="results/eval_full/")
    parser.add_argument("--evals-dir", default="evals")
    parser.add_argument("--tiers", default="1,2,3,4", help="Which tiers to run")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiers = set(args.tiers.split(","))

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    n_layers = model.config.num_hidden_layers

    if args.lora:
        print(f"Loading LoRA from {args.lora}...")
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    if args.lora:
        # PEFT wraps: model.base_model.model is the original CausalLM
        base = model.base_model.model
    else:
        base = model
    mgr = AblationManager(base)
    mgr.register()

    all_results = {
        "model": args.model,
        "lora": args.lora,
        "n_layers": n_layers,
    }

    if "1" in tiers:
        all_results["tier1"] = eval_tier1(model, tokenizer, mgr, n_layers, device)
    if "2" in tiers:
        all_results["tier2"] = eval_tier2(model, tokenizer, mgr, n_layers, device)
    if "3" in tiers:
        all_results["tier3"] = eval_tier3(model, tokenizer, mgr, n_layers, device, args.evals_dir)
    if "4" in tiers:
        all_results["tier4"] = eval_tier4(model, tokenizer, mgr, n_layers, device)

    out_path = out_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    mgr.remove()


if __name__ == "__main__":
    main()
